from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from src.models.components.rank_utils import rank_norm, spearman_corr
from src.models.components.rnr_gad import (
    CrossFittedResult,
    RankSafeProjector,
    _cfg_get,
    _section_get,
)
from src.models.components.rnr_gad_v3 import RNRv3Result, RNRv3Scorer
from src.models.components.rnr_gad_v31 import (
    RNRv31AttentionInputs,
    RNRv31AttentionOutput,
    attention_self_supervised_loss,
)


@dataclass
class RNRv31ScalarGateInputs:
    """The five pre-arbitration features available to the scalar gate."""

    stage1_rank: torch.Tensor
    graph_rank: torch.Tensor
    stage1_confidence: torch.Tensor
    graph_confidence: torch.Tensor
    features: torch.Tensor

    def as_loss_inputs(self) -> RNRv31AttentionInputs:
        token_scores = torch.stack([self.stage1_rank, self.graph_rank], dim=1)
        # Repeat the five features for the common v31 loss's two-source input.
        token_features = self.features.unsqueeze(1).expand(-1, 2, -1)
        return RNRv31AttentionInputs(
            token_names=["stage1", "graph_candidate"],
            token_scores=token_scores,
            token_features=token_features,
            source_ids=torch.tensor([0, 1], dtype=torch.long, device=token_scores.device),
            orientation_ids=torch.zeros(2, dtype=torch.long, device=token_scores.device),
        )


@dataclass
class RNRv31ScalarGateResult:
    score: torch.Tensor
    candidate_score: torch.Tensor
    score_raw: torch.Tensor
    lambda_stage1: torch.Tensor
    confidence: torch.Tensor
    stage2_result: RNRv3Result
    diagnostics: Dict[str, torch.Tensor]


class LowCapacityScalarGate(nn.Module):
    """Two-layer MLP producing one Stage-1 mixing coefficient per node."""

    feature_dim = 5

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        gate_cfg = _cfg_get(cfg, "scalar_gate", None)
        attention_cfg = _cfg_get(cfg, "attention", None)
        hidden_dim = max(1, int(_cfg_get(gate_cfg, "hidden_dim", 8)))
        dropout = float(
            _cfg_get(
                gate_cfg,
                "dropout",
                _cfg_get(attention_cfg, "dropout", 0.1),
            )
        )
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_output_bias(cfg)

    def _initialize_output_bias(self, cfg: Any) -> None:
        gate_cfg = _cfg_get(cfg, "scalar_gate", None)
        if not bool(_cfg_get(gate_cfg, "initialize_from_source_prior", True)):
            return
        attention_cfg = _cfg_get(cfg, "attention", None)
        source_prior = [
            float(item)
            for item in list(_cfg_get(attention_cfg, "source_prior", [0.45, 0.35, 0.20]))
        ]
        stage1_prior = max(source_prior[0], 0.0) if source_prior else 1.0
        graph_prior = max(source_prior[1], 0.0) if len(source_prior) > 1 else 1.0
        denominator = stage1_prior + graph_prior
        probability = stage1_prior / denominator if denominator > 0.0 else 0.5
        probability = min(max(probability, 1e-4), 1.0 - 1e-4)

        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, math.log(probability / (1.0 - probability)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


class RNRv31ScalarGateScorer(nn.Module):
    """Low-capacity Stage-1/graph-candidate convex arbitration baseline."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.direct_anchor_enabled = True
        self.graph_evidence_enabled = True
        self.reliability_enabled = True
        self.v3_scorer = RNRv3Scorer(cfg)
        self.gate = LowCapacityScalarGate(cfg)
        self.projector = RankSafeProjector(
            epsilon_max=float(
                _section_get(
                    cfg,
                    "attention",
                    "epsilon_max",
                    _section_get(cfg, "rank_safety", "epsilon_max", 0.05),
                )
            ),
            enabled=bool(_section_get(cfg, "attention", "project_to_stage1", False)),
            max_pairs=int(_section_get(cfg, "rank_safety", "max_pairs", 200_000)),
        )

    def build_inputs(
        self,
        stage1: CrossFittedResult,
        stage2: RNRv3Result,
    ) -> RNRv31ScalarGateInputs:
        stage1_rank = rank_norm(stage1.score.detach()).reshape(-1).float()
        graph_rank = rank_norm(stage2.candidate_score.detach()).reshape(-1).float()
        stage1_confidence = (
            stage1.confidence.detach().reshape(-1).float().to(stage1_rank).clamp(0.0, 1.0)
        )
        graph_confidence = (
            stage2.confidence.detach().reshape(-1).float().to(stage1_rank).clamp(0.0, 1.0)
        )
        features = torch.stack(
            [
                stage1_rank,
                graph_rank,
                (stage1_rank - graph_rank).abs(),
                stage1_confidence,
                graph_confidence,
            ],
            dim=1,
        )
        return RNRv31ScalarGateInputs(
            stage1_rank=stage1_rank,
            graph_rank=graph_rank,
            stage1_confidence=stage1_confidence,
            graph_confidence=graph_confidence,
            features=features.detach(),
        )

    def forward_inputs(self, inputs: RNRv31ScalarGateInputs) -> RNRv31AttentionOutput:
        lambda_stage1 = self.gate(inputs.features)
        lambda_graph = 1.0 - lambda_stage1
        score_raw = lambda_stage1 * inputs.stage1_rank + lambda_graph * inputs.graph_rank
        source_pair = torch.stack([lambda_stage1, lambda_graph], dim=1)
        source_weights = torch.cat(
            [source_pair, source_pair.new_zeros(source_pair.size(0), 1)],
            dim=1,
        )
        global_reliability = source_weights.mean(dim=0)
        global_reliability = global_reliability / global_reliability.sum().clamp_min(1e-8)
        confidence = (
            lambda_stage1 * inputs.stage1_confidence
            + lambda_graph * inputs.graph_confidence
        ).clamp(0.0, 1.0)
        single_source_credibility = source_pair.max(dim=1).values

        return RNRv31AttentionOutput(
            score_raw=score_raw,
            score=rank_norm(score_raw.detach()),
            base_score_raw=score_raw,
            rerank_delta=torch.zeros_like(score_raw),
            attention_weights=source_pair,
            source_weights=source_weights,
            source_reliability=global_reliability,
            source_reliability_logits=global_reliability.clamp_min(1e-8).log(),
            single_source_credibility=single_source_credibility,
            single_source_gap=(inputs.stage1_rank - inputs.graph_rank).abs(),
            confidence=confidence,
            token_logits=source_pair.clamp_min(1e-8).log(),
        )

    def compute(
        self,
        data: Any,
        stage1: CrossFittedResult,
        stage2: Optional[RNRv3Result] = None,
    ) -> RNRv31ScalarGateResult:
        stage2_result = stage2 if stage2 is not None else self.v3_scorer.compute(data, stage1)
        inputs = self.build_inputs(stage1, stage2_result)
        output = self.forward_inputs(inputs)
        projected = self.projector.project(stage1.score, output.score_raw, output.confidence)
        final_score = projected["score"] if self.projector.enabled else output.score
        diagnostics = {
            "rnrv31_scalar_gate/lambda_stage1_mean": output.attention_weights[:, 0].mean(),
            "rnrv31_scalar_gate/lambda_stage1_std": output.attention_weights[:, 0].float().std(
                unbiased=False
            ),
            "rnrv31_scalar_gate/lambda_graph_mean": output.attention_weights[:, 1].mean(),
            "rnrv31_scalar_gate/confidence_mean": output.confidence.mean(),
            "rnrv31_scalar_gate/stage1_candidate_disagreement_mean": (
                inputs.stage1_rank - inputs.graph_rank
            ).abs().mean(),
            "rnrv31_scalar_gate/stage1_final_spearman": spearman_corr(
                stage1.score,
                final_score,
            ),
            "rnrv31_scalar_gate/candidate_final_spearman": spearman_corr(
                stage2_result.candidate_score,
                final_score,
            ),
            "rnrv31_scalar_gate/rank_safe_spearman_base": projected["spearman_base"],
            "rnrv31_scalar_gate/rank_safe_max_rank_displacement": projected[
                "max_rank_displacement"
            ],
            "rnrv31_scalar_gate/rank_safe_flip_ratio": projected["rank_flip_ratio"],
            "rnrv31_scalar_gate/rank_safe_protected_pair_violations": projected[
                "protected_pair_violations"
            ],
        }
        return RNRv31ScalarGateResult(
            score=final_score.detach(),
            candidate_score=output.score.detach(),
            score_raw=output.score_raw.detach(),
            lambda_stage1=output.attention_weights[:, 0].detach(),
            confidence=output.confidence.detach(),
            stage2_result=stage2_result,
            diagnostics=diagnostics,
        )

    def target_score(
        self,
        stage1_target: CrossFittedResult,
        stage2_target: Optional[RNRv3Result],
    ) -> torch.Tensor:
        mode = str(_section_get(self.cfg, "attention", "target_source", "heldout_consensus"))
        stage_score = rank_norm(stage1_target.score.detach()).reshape(-1)
        if stage2_target is None:
            return stage_score

        graph_score = rank_norm(stage2_target.candidate_score.detach()).reshape(-1)
        if mode == "heldout_stage1":
            return stage_score
        if mode == "heldout_v3":
            return graph_score

        non_stage = [
            idx
            for idx, name in enumerate(stage2_target.component_names)
            if name != "stage1"
        ]
        parts = [stage_score, graph_score]
        if non_stage:
            ranked = torch.stack(
                [
                    rank_norm(stage2_target.component_scores[:, idx].detach())
                    for idx in non_stage
                ],
                dim=1,
            )
            reducer = str(
                _section_get(self.cfg, "attention", "target_component_reducer", "median")
            )
            parts.append(
                ranked.mean(dim=1)
                if reducer == "mean"
                else ranked.median(dim=1).values
            )
        return rank_norm(torch.stack(parts, dim=1).median(dim=1).values)


def scalar_gate_self_supervised_loss(
    output_a: RNRv31AttentionOutput,
    output_b: RNRv31AttentionOutput,
    target_score: torch.Tensor,
    target_confidence: torch.Tensor,
    cfg: Any,
    stage1_score: torch.Tensor,
    stage2_score: torch.Tensor,
    inputs: RNRv31ScalarGateInputs,
    target_inputs: Optional[RNRv31ScalarGateInputs],
) -> Dict[str, torch.Tensor]:
    """Apply the v31 objective to the two aggregate sources.

    Architecture-specific quantities are represented directly by the scalar
    gate: the two source probabilities replace token attention, source
    reliability is their global usage, and the residual penalty is exactly
    zero because this baseline has no residual head.
    """

    return attention_self_supervised_loss(
        output_a,
        output_b,
        target_score,
        target_confidence,
        cfg,
        stage1_score=stage1_score,
        stage2_score=stage2_score,
        inputs=inputs.as_loss_inputs(),
        target_inputs=(
            target_inputs.as_loss_inputs()
            if target_inputs is not None
            else None
        ),
    )
