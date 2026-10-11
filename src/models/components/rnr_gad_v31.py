from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.rank_utils import rank_norm, spearman_corr
from src.models.components.rnr_gad import CrossFittedResult, RankSafeProjector, _cfg_get, _section_get
from src.models.components.rnr_gad_v3 import RNRv3Result, RNRv3Scorer


@dataclass
class RNRv31AttentionInputs:
    token_names: List[str]
    token_scores: torch.Tensor
    token_features: torch.Tensor
    source_ids: torch.Tensor
    orientation_ids: torch.Tensor


@dataclass
class RNRv31AttentionOutput:
    score_raw: torch.Tensor
    score: torch.Tensor
    base_score_raw: torch.Tensor
    rerank_delta: torch.Tensor
    attention_weights: torch.Tensor
    source_weights: torch.Tensor
    source_reliability: torch.Tensor
    source_reliability_logits: torch.Tensor
    single_source_credibility: torch.Tensor
    single_source_gap: torch.Tensor
    confidence: torch.Tensor
    token_logits: torch.Tensor


@dataclass
class RNRv31Result:
    score: torch.Tensor
    candidate_score: torch.Tensor
    score_raw: torch.Tensor
    base_score_raw: torch.Tensor
    rerank_delta: torch.Tensor
    confidence: torch.Tensor
    attention_weights: torch.Tensor
    token_names: List[str]
    token_scores: torch.Tensor
    token_features: torch.Tensor
    source_ids: torch.Tensor
    orientation_ids: torch.Tensor
    stage2_result: RNRv3Result
    diagnostics: Dict[str, torch.Tensor]


def _top_tail_overlap(scores_a: torch.Tensor, scores_b: torch.Tensor, q: float) -> torch.Tensor:
    n = int(scores_a.numel())
    if n == 0:
        return scores_a.new_tensor(0.0)
    k = max(1, int(math.ceil(float(q) * n)))
    top_a = torch.topk(scores_a.reshape(-1), k).indices
    top_b = torch.topk(scores_b.reshape(-1), k).indices
    marker = torch.zeros(n, dtype=torch.bool, device=scores_a.device)
    marker[top_a] = True
    return marker[top_b].float().mean()


def _signal(values: torch.Tensor) -> torch.Tensor:
    return torch.quantile(values.float(), 0.95) - values.float().median()


def _top_fraction_mean(values: torch.Tensor, q: float) -> torch.Tensor:
    n = int(values.numel())
    if n == 0:
        return values.new_tensor(0.0)
    k = max(1, int(math.ceil(max(0.0, min(1.0, float(q))) * n)))
    return torch.topk(values.reshape(-1).float(), k=k).values.mean().to(values)


def _orientation_id(name: str) -> int:
    if name.endswith("_high"):
        return 1
    if name.endswith("_low"):
        return 2
    if name.endswith("_two"):
        return 3
    return 0


class StageEvidenceAttention(nn.Module):
    """Self-attention mixer over stage and propagation evidence tokens."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        temperature: float,
        use_source_gate: bool,
        source_prior: List[float],
        source_dropout: float,
        use_rerank_head: bool,
        rerank_residual_scale: float,
        use_source_reliability: bool,
        source_reliability_scale: float,
        use_reliability_evidence: bool,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.use_source_gate = bool(use_source_gate)
        self.source_dropout = max(0.0, min(1.0, float(source_dropout)))
        self.use_rerank_head = bool(use_rerank_head)
        self.rerank_residual_scale = max(0.0, float(rerank_residual_scale))
        self.use_source_reliability = bool(use_source_reliability)
        self.source_reliability_scale = max(0.0, float(source_reliability_scale))
        self.use_reliability_evidence = bool(use_reliability_evidence)
        self.numeric_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.source_embedding = nn.Embedding(3, d_model)
        self.orientation_embedding = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, int(num_heads)),
            dim_feedforward=max(d_model * 2, int(2 * d_model)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.weight_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        self.source_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 3),
        )
        self.source_reliability_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        self.single_source_head = nn.Sequential(
            nn.LayerNorm(d_model + 7),
            nn.Linear(d_model + 7, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        self.rerank_head = nn.Sequential(
            nn.LayerNorm(d_model + 8),
            nn.Linear(d_model + 8, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        prior = torch.tensor(source_prior[:3], dtype=torch.float32)
        if prior.numel() != 3 or float(prior.clamp_min(0.0).sum()) <= 0.0:
            prior = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        prior = prior.clamp_min(1e-6)
        prior = prior / prior.sum()
        self.register_buffer("source_log_prior", prior.log())

    def forward(
        self,
        token_features: torch.Tensor,
        source_ids: torch.Tensor,
        orientation_ids: torch.Tensor,
    ) -> RNRv31AttentionOutput:
        source_ids = source_ids.to(token_features.device).long()
        orientation_ids = orientation_ids.to(token_features.device).long()
        x = self.numeric_projection(token_features)
        x = x + self.source_embedding(source_ids).unsqueeze(0)
        x = x + self.orientation_embedding(orientation_ids).unsqueeze(0)
        if x.is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                encoded = self.encoder(x)
        else:
            encoded = self.encoder(x)
        logits = self.weight_head(encoded).squeeze(-1) / self.temperature
        predicted_reliability_logits = self._source_reliability_logits(encoded, source_ids)
        if self.use_reliability_evidence:
            reliability_logits = (
                predicted_reliability_logits + self.source_log_prior.to(predicted_reliability_logits)
            )
        else:
            reliability_logits = self.source_log_prior.to(encoded) + 0.0 * predicted_reliability_logits
        reliability = torch.softmax(reliability_logits, dim=0)
        if self.use_source_gate:
            weights, source_weights = self._source_gated_weights(logits, encoded, source_ids, reliability_logits)
        else:
            weights = torch.softmax(logits, dim=1)
            source_weights = torch.stack(
                [weights[:, source_ids == idx].sum(dim=1) for idx in range(3)],
                dim=1,
            )
        base_score_raw = (weights * token_features[..., 0]).sum(dim=1)
        pooled = (weights.unsqueeze(-1) * encoded).sum(dim=1)
        source_summary, source_stats = self._node_source_summaries(token_features, source_ids)
        predicted_single_source_credibility = torch.sigmoid(
            self.single_source_head(torch.cat([pooled, source_weights, source_stats], dim=1)).squeeze(-1)
        )
        if self.use_reliability_evidence:
            single_source_credibility = predicted_single_source_credibility
        else:
            single_source_credibility = (
                base_score_raw.new_full(base_score_raw.shape, 0.5) + 0.0 * predicted_single_source_credibility
            )
            base_score_raw = base_score_raw + 0.0 * (
                predicted_reliability_logits.sum() + predicted_single_source_credibility.sum()
            )
        if self.use_rerank_head and self.rerank_residual_scale > 0.0:
            rerank_input = torch.cat(
                [pooled, source_weights, source_stats, single_source_credibility.reshape(-1, 1)],
                dim=1,
            )
            rerank_delta = torch.tanh(self.rerank_head(rerank_input).squeeze(-1)) * self.rerank_residual_scale
        else:
            rerank_delta = base_score_raw.new_zeros(base_score_raw.shape)
        score_raw = base_score_raw + rerank_delta
        if self.use_reliability_evidence:
            confidence = (weights * token_features[..., 4].clamp(0.0, 1.0)).sum(dim=1).clamp(0.0, 1.0)
        else:
            confidence = base_score_raw.new_ones(base_score_raw.shape)
        return RNRv31AttentionOutput(
            score_raw=score_raw,
            score=rank_norm(score_raw.detach()),
            base_score_raw=base_score_raw,
            rerank_delta=rerank_delta,
            attention_weights=weights,
            source_weights=source_weights,
            source_reliability=reliability,
            source_reliability_logits=reliability_logits,
            single_source_credibility=single_source_credibility,
            single_source_gap=source_stats[:, 2],
            confidence=confidence,
            token_logits=logits,
        )

    def _source_gated_weights(
        self,
        token_logits: torch.Tensor,
        encoded: torch.Tensor,
        source_ids: torch.Tensor,
        reliability_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = encoded.mean(dim=1)
        source_logits = self.source_head(pooled) / self.temperature
        source_logits = source_logits + self.source_log_prior.to(source_logits).reshape(1, -1)
        if self.use_source_reliability and self.source_reliability_scale > 0.0:
            source_logits = source_logits + self.source_reliability_scale * reliability_logits.reshape(1, -1)

        present = torch.stack([(source_ids == idx).any() for idx in range(3)]).to(source_logits.device)
        source_logits = source_logits.masked_fill(~present.reshape(1, -1), -1e9)
        if self.training and self.source_dropout > 0.0:
            keep = torch.rand_like(source_logits) >= self.source_dropout
            keep = keep | ~present.reshape(1, -1)
            keep_any = (keep & present.reshape(1, -1)).any(dim=1, keepdim=True)
            keep = torch.where(keep_any, keep, present.reshape(1, -1).expand_as(keep))
            source_logits = source_logits.masked_fill(~keep, -1e9)

        source_weights = torch.softmax(source_logits, dim=1)
        weights = torch.zeros_like(token_logits)
        for idx in range(3):
            mask = source_ids == idx
            if not bool(mask.any().detach().cpu().item()):
                continue
            local_logits = token_logits.masked_fill(~mask.reshape(1, -1), -1e9)
            local_weights = torch.softmax(local_logits, dim=1)
            weights = weights + source_weights[:, idx].reshape(-1, 1) * local_weights
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8), source_weights

    def _source_reliability_logits(self, encoded: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        logits = []
        for idx in range(3):
            mask = source_ids == idx
            if bool(mask.any().detach().cpu().item()):
                context = encoded[:, mask].mean(dim=(0, 1))
                logits.append(self.source_reliability_head(context).reshape(()))
            else:
                logits.append(encoded.new_tensor(-1e9))
        return torch.stack(logits)

    def _node_source_summaries(
        self,
        token_features: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = token_features[..., 0]
        summaries = []
        for idx in range(3):
            mask = source_ids == idx
            if not bool(mask.any().detach().cpu().item()):
                summaries.append(score.new_zeros(score.size(0)))
            elif idx == 2:
                summaries.append(score[:, mask].max(dim=1).values)
            else:
                summaries.append(score[:, mask].mean(dim=1))
        source_summary = torch.stack(summaries, dim=1)
        top_values = torch.topk(source_summary, k=min(2, source_summary.size(1)), dim=1).values
        max_source = top_values[:, 0]
        second_source = top_values[:, -1] if top_values.size(1) > 1 else max_source
        gap = max_source - second_source
        mean_source = source_summary.mean(dim=1)
        stats = torch.stack([max_source, second_source, gap, mean_source], dim=1)
        return source_summary, stats


class RNRv31AttentionScorer(nn.Module):
    """Trainable v3.1 scorer that arbitrates stage1 and v3 evidence with self-attention."""

    feature_dim = 10

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        attn_cfg = _cfg_get(cfg, "attention", None)
        # Keep the legacy feature width and orientation embedding for checkpoint
        # compatibility when the canonical configuration disables these inputs.
        self.include_graph_final_token = bool(
            _cfg_get(attn_cfg, "include_graph_final_token", True)
        )
        self.use_token_delta_features = bool(
            _cfg_get(attn_cfg, "use_token_delta_features", True)
        )
        self.use_orientation_identity = bool(
            _cfg_get(attn_cfg, "use_orientation_identity", True)
        )
        self.direct_anchor_enabled = not bool(_section_get(cfg, "ablation", "disable_direct_anchor", False))
        self.graph_evidence_enabled = not bool(_section_get(cfg, "ablation", "disable_graph_evidence", False))
        self.reliability_enabled = not bool(_section_get(cfg, "ablation", "disable_reliability", False))
        if not self.direct_anchor_enabled and not self.graph_evidence_enabled:
            raise ValueError("RNRv31 requires at least one of direct anchor or graph evidence.")
        self.eps = float(_section_get(cfg, "crossfit", "eps", 1e-8))
        self.v3_scorer = RNRv3Scorer(cfg)
        self.mixer = StageEvidenceAttention(
            feature_dim=self.feature_dim,
            d_model=int(_cfg_get(attn_cfg, "d_model", 32)),
            num_heads=int(_cfg_get(attn_cfg, "num_heads", 4)),
            num_layers=int(_cfg_get(attn_cfg, "num_layers", 1)),
            dropout=float(_cfg_get(attn_cfg, "dropout", 0.1)),
            temperature=float(_cfg_get(attn_cfg, "temperature", 1.0)),
            use_source_gate=bool(_cfg_get(attn_cfg, "use_source_gate", True)),
            source_prior=[float(item) for item in list(_cfg_get(attn_cfg, "source_prior", [0.45, 0.35, 0.20]))],
            source_dropout=(float(_cfg_get(attn_cfg, "source_dropout", 0.0)) if self.reliability_enabled else 0.0),
            use_rerank_head=bool(_cfg_get(attn_cfg, "use_rerank_head", True)),
            rerank_residual_scale=float(_cfg_get(attn_cfg, "rerank_residual_scale", 0.08)),
            use_source_reliability=(
                bool(_cfg_get(attn_cfg, "use_source_reliability", True)) and self.reliability_enabled
            ),
            source_reliability_scale=(
                float(_cfg_get(attn_cfg, "source_reliability_scale", 0.5)) if self.reliability_enabled else 0.0
            ),
            use_reliability_evidence=self.reliability_enabled,
        )
        self.projector = RankSafeProjector(
            epsilon_max=float(_section_get(cfg, "attention", "epsilon_max", _section_get(cfg, "rank_safety", "epsilon_max", 0.05))),
            enabled=(
                bool(_section_get(cfg, "attention", "project_to_stage1", False))
                and self.direct_anchor_enabled
                and self.reliability_enabled
            ),
            max_pairs=int(_section_get(cfg, "rank_safety", "max_pairs", 200_000)),
        )

    def build_inputs(
        self,
        stage1: CrossFittedResult,
        stage2: RNRv3Result,
    ) -> RNRv31AttentionInputs:
        device = stage1.score.device
        token_names = []
        token_scores = []
        source_ids = []
        orientation_ids = []
        scalar_rows = []
        score_rows = []
        confidence_rows = []

        stage1_score = rank_norm(stage1.score.detach()).reshape(-1).float()
        stage2_score = rank_norm(stage2.candidate_score.detach()).reshape(-1).float()
        top_tail_q = float(_section_get(self.cfg, "attention", "top_tail_q", _section_get(self.cfg, "propagation_score", "top_tail_q", 0.05)))

        def add_token(
            name: str,
            score: torch.Tensor,
            source_id: int,
            confidence: torch.Tensor,
            eligible: torch.Tensor,
            prior_weight: torch.Tensor,
        ) -> None:
            score = rank_norm(score.detach()).reshape(-1).float().to(device)
            confidence = confidence.detach().reshape(-1).to(device)
            corr = (
                spearman_corr(score, stage1_score).to(device)
                if self.direct_anchor_enabled
                else score.new_tensor(0.0)
            )
            overlap = (
                _top_tail_overlap(score, stage1_score, top_tail_q).to(device)
                if self.direct_anchor_enabled
                else score.new_tensor(0.0)
            )
            # Neutral constants keep the ablation's feature width unchanged.
            signal = _signal(score).to(device) if self.reliability_enabled else score.new_tensor(0.0)
            eligible_feature = eligible.to(device).float() if self.reliability_enabled else score.new_tensor(1.0)
            prior_feature = prior_weight.to(device).float() if self.reliability_enabled else score.new_tensor(1.0)
            scalars = torch.stack(
                [
                    signal,
                    corr,
                    overlap,
                    eligible_feature,
                    prior_feature,
                ]
            )
            token_names.append(name)
            token_scores.append(score)
            source_ids.append(source_id)
            orientation_ids.append(
                _orientation_id(name) if self.use_orientation_identity else 0
            )
            scalar_rows.append(scalars)
            score_rows.append(score)
            confidence_rows.append(confidence if self.reliability_enabled else torch.ones_like(confidence))

        if self.direct_anchor_enabled:
            add_token(
                "stage1",
                stage1_score,
                0,
                stage1.confidence,
                stage1.score.new_tensor(1.0),
                stage1.score.new_tensor(1.0),
            )

        if self.graph_evidence_enabled:
            add_token(
                "rnrv3_candidate",
                stage2_score,
                1,
                stage2.confidence,
                stage2.diagnostics.get("rnrv3/non_stage_eligible", stage2.score.new_tensor(1.0)),
                stage2.score.new_tensor(1.0),
            )
            if self.include_graph_final_token:
                add_token(
                    "rnrv3_final",
                    stage2.score,
                    1,
                    stage2.confidence,
                    stage2.diagnostics.get("rnrv3/non_stage_eligible", stage2.score.new_tensor(1.0)),
                    stage2.score.new_tensor(1.0),
                )

            max_components = int(_section_get(self.cfg, "attention", "max_component_tokens", 24))
            non_stage_indices = [idx for idx, name in enumerate(stage2.component_names) if name != "stage1"]
            if len(non_stage_indices) > max_components:
                component_scores = stage2.component_scores[:, non_stage_indices]
                signals = torch.stack([_signal(component_scores[:, idx]) for idx in range(component_scores.size(1))])
                order = torch.argsort(signals, descending=True).detach().cpu().tolist()
                non_stage_indices = [non_stage_indices[idx] for idx in order[:max_components]]

            for idx in non_stage_indices:
                name = f"component:{stage2.component_names[idx]}"
                add_token(
                    name,
                    stage2.component_scores[:, idx],
                    2,
                    stage2.confidence,
                    stage2.component_eligible[idx],
                    stage2.component_weights[idx],
                )

        score_tensor = torch.stack(score_rows, dim=1)
        scalar_tensor = torch.stack(scalar_rows, dim=0).to(score_tensor)
        features = []
        for idx in range(score_tensor.size(1)):
            score = score_tensor[:, idx]
            scalars = scalar_tensor[idx].reshape(1, -1).expand(score_tensor.size(0), -1)
            delta_stage1 = score - stage1_score if self.direct_anchor_enabled else torch.zeros_like(score)
            delta_stage2 = score - stage2_score if self.graph_evidence_enabled else torch.zeros_like(score)
            if not self.use_token_delta_features:
                delta_stage1 = torch.zeros_like(score)
                delta_stage2 = torch.zeros_like(score)
            token_feature = torch.cat(
                [
                    score.reshape(-1, 1),
                    delta_stage1.reshape(-1, 1),
                    delta_stage1.abs().reshape(-1, 1),
                    delta_stage2.reshape(-1, 1),
                    confidence_rows[idx].reshape(-1, 1).to(score),
                    scalars,
                ],
                dim=1,
            )
            features.append(token_feature)

        return RNRv31AttentionInputs(
            token_names=token_names,
            token_scores=score_tensor.detach(),
            token_features=torch.stack(features, dim=1).detach(),
            source_ids=torch.tensor(source_ids, dtype=torch.long, device=device),
            orientation_ids=torch.tensor(orientation_ids, dtype=torch.long, device=device),
        )

    def forward_inputs(self, inputs: RNRv31AttentionInputs) -> RNRv31AttentionOutput:
        return self.mixer(inputs.token_features, inputs.source_ids, inputs.orientation_ids)

    def compute(
        self,
        data: Any,
        stage1: CrossFittedResult,
        stage2: Optional[RNRv3Result] = None,
    ) -> RNRv31Result:
        stage2_result = stage2 if stage2 is not None else self.v3_scorer.compute(data, stage1)
        inputs = self.build_inputs(stage1, stage2_result)
        output = self.forward_inputs(inputs)
        projected = self.projector.project(stage1.score, output.score_raw, output.confidence)
        final_score = projected["score"] if self.projector.enabled else rank_norm(output.score_raw.detach())
        diagnostics = self._diagnostics(stage1, stage2_result, inputs, output, final_score, projected)
        return RNRv31Result(
            score=final_score.detach(),
            candidate_score=output.score.detach(),
            score_raw=output.score_raw.detach(),
            base_score_raw=output.base_score_raw.detach(),
            rerank_delta=output.rerank_delta.detach(),
            confidence=output.confidence.detach(),
            attention_weights=output.attention_weights.detach(),
            token_names=inputs.token_names,
            token_scores=inputs.token_scores.detach(),
            token_features=inputs.token_features.detach(),
            source_ids=inputs.source_ids.detach(),
            orientation_ids=inputs.orientation_ids.detach(),
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
        if not self.graph_evidence_enabled or stage2_target is None:
            return stage_score

        v3_score = rank_norm(stage2_target.candidate_score.detach()).reshape(-1)
        if mode == "heldout_stage1":
            return stage_score if self.direct_anchor_enabled else v3_score
        if mode == "heldout_v3":
            return v3_score

        component_scores = stage2_target.component_scores
        non_stage = [idx for idx, name in enumerate(stage2_target.component_names) if name != "stage1"]
        parts = [v3_score]
        if self.direct_anchor_enabled:
            parts.insert(0, stage_score)
        if non_stage:
            ranked = torch.stack([rank_norm(component_scores[:, idx].detach()) for idx in non_stage], dim=1)
            reducer = str(_section_get(self.cfg, "attention", "target_component_reducer", "median"))
            if reducer == "mean":
                parts.append(ranked.mean(dim=1))
            else:
                parts.append(ranked.median(dim=1).values)
        return rank_norm(torch.stack(parts, dim=1).median(dim=1).values)

    def _diagnostics(
        self,
        stage1: CrossFittedResult,
        stage2: RNRv3Result,
        inputs: RNRv31AttentionInputs,
        output: RNRv31AttentionOutput,
        final_score: torch.Tensor,
        projected: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        weights = output.attention_weights
        source_weights = output.source_weights
        entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1)
        source_entropy = -(source_weights * source_weights.clamp_min(self.eps).log()).sum(dim=1)
        source_ids = inputs.source_ids.to(weights.device)
        stage1_mask = source_ids == 0
        candidate_mask = source_ids == 1
        component_mask = source_ids == 2

        def mean_weight(mask: torch.Tensor) -> torch.Tensor:
            if not bool(mask.any().detach().cpu().item()):
                return weights.new_tensor(0.0)
            return weights[:, mask].sum(dim=1).mean()

        diagnostics = {
            "rnrv31/token_count": weights.new_tensor(float(weights.size(1))),
            "rnrv31/component_token_count": weights.new_tensor(float(component_mask.float().sum().detach().cpu())),
            "rnrv31/attention_entropy_mean": entropy.mean(),
            "rnrv31/source_entropy_mean": source_entropy.mean(),
            "rnrv31/attention_max_mean": weights.max(dim=1).values.mean(),
            "rnrv31/attention_stage1_mean": source_weights[:, 0].mean(),
            "rnrv31/attention_candidate_mean": source_weights[:, 1].mean(),
            "rnrv31/attention_component_mean": source_weights[:, 2].mean(),
            "rnrv31/source_reliability_stage1": output.source_reliability[0],
            "rnrv31/source_reliability_candidate": output.source_reliability[1],
            "rnrv31/source_reliability_component": output.source_reliability[2],
            "rnrv31/token_attention_stage1_mean": mean_weight(stage1_mask),
            "rnrv31/token_attention_candidate_mean": mean_weight(candidate_mask),
            "rnrv31/token_attention_component_mean": mean_weight(component_mask),
            "rnrv31/rerank_delta_abs_mean": output.rerank_delta.abs().mean(),
            "rnrv31/rerank_delta_std": output.rerank_delta.float().std(unbiased=False),
            "rnrv31/single_source_credibility_mean": output.single_source_credibility.mean(),
            "rnrv31/single_source_credibility_top_mean": _top_fraction_mean(output.single_source_credibility, 0.05),
            "rnrv31/single_source_gap_mean": output.single_source_gap.mean(),
            "rnrv31/base_final_spearman": spearman_corr(output.base_score_raw, final_score),
            "rnrv31/confidence_mean": output.confidence.mean(),
            "rnrv31/stage1_final_spearman": spearman_corr(stage1.score, final_score),
            "rnrv31/v3_final_spearman": spearman_corr(stage2.score, final_score),
            "rnrv31/candidate_final_spearman": spearman_corr(stage2.candidate_score, final_score),
            "rnrv31/rank_safe_spearman_base": projected["spearman_base"],
            "rnrv31/rank_safe_max_rank_displacement": projected["max_rank_displacement"],
            "rnrv31/rank_safe_flip_ratio": projected["rank_flip_ratio"],
            "rnrv31/rank_safe_protected_pair_violations": projected["protected_pair_violations"],
        }
        tail_summary = _tail_anchor_summary(inputs, _cfg_get(self.cfg, "attention", None))
        if tail_summary is not None:
            diagnostics["rnrv31/tail_anchor_final_spearman"] = spearman_corr(tail_summary["anchor"], final_score)
            diagnostics["rnrv31/tail_anchor_signal"] = _signal(tail_summary["anchor"])
            diagnostics["rnrv31/single_source_gap_mean"] = tail_summary["single_source_gap"].mean()
        for idx, name in enumerate(inputs.token_names):
            diagnostics[f"rnrv31/token_weight/{name}"] = weights[:, idx].mean()
        return diagnostics


def attention_self_supervised_loss(
    output_a: RNRv31AttentionOutput,
    output_b: RNRv31AttentionOutput,
    target_score: torch.Tensor,
    target_confidence: torch.Tensor,
    cfg: Any,
    stage1_score: Optional[torch.Tensor] = None,
    stage2_score: Optional[torch.Tensor] = None,
    inputs: Optional[RNRv31AttentionInputs] = None,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
) -> Dict[str, torch.Tensor]:
    attn_cfg = _cfg_get(cfg, "attention", None)
    eps = float(_section_get(cfg, "crossfit", "eps", 1e-8))
    direct_anchor_enabled = not bool(_section_get(cfg, "ablation", "disable_direct_anchor", False))
    graph_evidence_enabled = not bool(_section_get(cfg, "ablation", "disable_graph_evidence", False))
    reliability_enabled = not bool(_section_get(cfg, "ablation", "disable_reliability", False))
    source_count = int(torch.unique(inputs.source_ids).numel()) if inputs is not None else 0
    cross_source_reliability_enabled = reliability_enabled and graph_evidence_enabled and source_count >= 2
    zero = output_a.score_raw.sum() * 0.0
    target = rank_norm(target_score.detach()).reshape(-1).to(output_a.score_raw)
    confidence = (
        target_confidence.detach().reshape(-1).to(output_a.score_raw).clamp(0.0, 1.0)
        if reliability_enabled
        else torch.ones_like(target)
    )
    mse_loss = (confidence * (output_a.score_raw - target).pow(2)).sum() / confidence.sum().clamp_min(eps)
    rank_loss = _pairwise_rank_loss(
        output_a.score_raw,
        target,
        confidence,
        max_pairs=int(_cfg_get(attn_cfg, "max_rank_pairs", 20_000)),
        margin=float(_cfg_get(attn_cfg, "rank_margin", 0.02)),
        temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
    )
    if reliability_enabled:
        stable_score = (output_a.score_raw - output_b.score_raw).pow(2).mean()
        stable_attention = _symmetric_kl(output_a.attention_weights, output_b.attention_weights, eps)
    else:
        stable_score = zero
        stable_attention = zero
    entropy = -(output_a.attention_weights * output_a.attention_weights.clamp_min(eps).log()).sum(dim=1).mean()
    source_entropy = -(output_a.source_weights * output_a.source_weights.clamp_min(eps).log()).sum(dim=1).mean()
    usage = output_a.attention_weights.mean(dim=0)
    usage_entropy = -(usage * usage.clamp_min(eps).log()).sum()
    top_tail = _top_tail_contrast_loss(
        output_a.score_raw,
        target,
        confidence,
        q_high=float(_cfg_get(attn_cfg, "top_tail_q_high", 0.90)),
        q_low=float(_cfg_get(attn_cfg, "top_tail_q_low", 0.50)),
        max_pairs=int(_cfg_get(attn_cfg, "max_top_tail_pairs", _cfg_get(attn_cfg, "max_rank_pairs", 20_000))),
        temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
    )
    # Tail corroboration requires both sources and their reliability evidence.
    if cross_source_reliability_enabled:
        extreme_tail_listwise, adaptive_tail_count, adaptive_tail_threshold = _extreme_tail_listwise_loss(
            output_a.score_raw,
            target,
            confidence,
            q=float(_cfg_get(attn_cfg, "extreme_tail_q", 0.02)),
            max_nodes=int(_cfg_get(attn_cfg, "max_extreme_tail_nodes", 4096)),
            temperature=float(_cfg_get(attn_cfg, "extreme_tail_temperature", 0.05)),
            target_temperature=float(_cfg_get(attn_cfg, "extreme_tail_target_temperature", 0.05)),
            attn_cfg=attn_cfg,
            inputs=inputs,
            target_inputs=target_inputs,
            credibility=output_a.single_source_credibility,
        )
    else:
        extreme_tail_listwise = zero
        adaptive_tail_count = zero
        adaptive_tail_threshold = zero
    auc_guard = _pairwise_rank_loss(
        output_a.score_raw,
        target,
        confidence,
        max_pairs=int(_cfg_get(attn_cfg, "max_auc_guard_pairs", _cfg_get(attn_cfg, "max_rank_pairs", 20_000))),
        margin=float(_cfg_get(attn_cfg, "auc_guard_margin", 0.0)),
        temperature=float(_cfg_get(attn_cfg, "auc_guard_temperature", _cfg_get(attn_cfg, "rank_temperature", 0.1))),
    )
    if not direct_anchor_enabled or stage1_score is None:
        stage1_protection = zero
    else:
        stage1_reference = rank_norm(stage1_score.detach()).reshape(-1).to(output_a.score_raw)
        stage1_protection = _pairwise_rank_loss(
            output_a.score_raw,
            stage1_reference,
            confidence,
            max_pairs=int(_cfg_get(attn_cfg, "max_stage1_pairs", _cfg_get(attn_cfg, "max_rank_pairs", 20_000))),
            margin=float(_cfg_get(attn_cfg, "stage1_protection_margin", 0.05)),
            temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
        )
    if not graph_evidence_enabled or stage2_score is None:
        stage2_protection = zero
    else:
        stage2_reference = rank_norm(stage2_score.detach()).reshape(-1).to(output_a.score_raw)
        stage2_protection = _pairwise_rank_loss(
            output_a.score_raw,
            stage2_reference,
            confidence,
            max_pairs=int(_cfg_get(attn_cfg, "max_stage2_pairs", _cfg_get(attn_cfg, "max_rank_pairs", 20_000))),
            margin=float(_cfg_get(attn_cfg, "stage2_protection_margin", 0.05)),
            temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
        )
    if cross_source_reliability_enabled:
        tail_anchor, single_spike = _tail_anchor_losses(
            output_a.score_raw,
            target,
            confidence,
            inputs,
            attn_cfg,
            eps,
        )
        source_alignment = _source_alignment_loss(
            output_a,
            inputs,
            target,
            attn_cfg,
            eps,
            target_inputs=target_inputs,
        )
        source_reliability = _source_reliability_loss(
            output_a,
            inputs,
            target,
            attn_cfg,
            eps,
            target_inputs=target_inputs,
        )
        single_source_credibility = _single_source_credibility_loss(
            output_a,
            inputs,
            target,
            confidence,
            attn_cfg,
            eps,
            target_inputs=target_inputs,
        )
    else:
        tail_anchor = zero
        single_spike = zero
        source_alignment = zero
        source_reliability = zero
        single_source_credibility = zero
    rerank_l2 = output_a.rerank_delta.pow(2).mean()

    loss = (
        float(_cfg_get(attn_cfg, "mse_weight", 0.25)) * mse_loss
        + float(_cfg_get(attn_cfg, "rank_weight", 1.0)) * rank_loss
        + float(_cfg_get(attn_cfg, "auc_guard_weight", 0.0)) * auc_guard
        + float(_cfg_get(attn_cfg, "top_tail_weight", 0.5)) * top_tail
        + float(_cfg_get(attn_cfg, "extreme_tail_listwise_weight", 0.0)) * extreme_tail_listwise
        + float(_cfg_get(attn_cfg, "stage1_protection_weight", 0.1)) * stage1_protection
        + float(_cfg_get(attn_cfg, "stage2_protection_weight", 0.0)) * stage2_protection
        + float(_cfg_get(attn_cfg, "tail_anchor_weight", 0.0)) * tail_anchor
        + float(_cfg_get(attn_cfg, "single_spike_weight", 0.0)) * single_spike
        + float(_cfg_get(attn_cfg, "source_alignment_weight", 0.0)) * source_alignment
        + float(_cfg_get(attn_cfg, "source_reliability_weight", 0.0)) * source_reliability
        + float(_cfg_get(attn_cfg, "single_source_credibility_weight", 0.0)) * single_source_credibility
        + float(_cfg_get(attn_cfg, "stability_weight", 0.1)) * (stable_score + stable_attention)
        + float(_cfg_get(attn_cfg, "sparsity_weight", 0.0)) * entropy
        + float(_cfg_get(attn_cfg, "source_sparsity_weight", 0.0)) * source_entropy
        + float(_cfg_get(attn_cfg, "rerank_l2_weight", 0.0)) * rerank_l2
        - float(_cfg_get(attn_cfg, "usage_entropy_weight", 0.0)) * usage_entropy
    )
    return {
        "loss": loss,
        "mse": mse_loss.detach(),
        "rank": rank_loss.detach(),
        "stable_score": stable_score.detach(),
        "stable_attention": stable_attention.detach(),
        "entropy": entropy.detach(),
        "source_entropy": source_entropy.detach(),
        "auc_guard": auc_guard.detach(),
        "top_tail": top_tail.detach(),
        "extreme_tail_listwise": extreme_tail_listwise.detach(),
        "adaptive_tail_count": adaptive_tail_count.detach(),
        "adaptive_tail_threshold": adaptive_tail_threshold.detach(),
        "stage1_protection": stage1_protection.detach(),
        "stage2_protection": stage2_protection.detach(),
        "tail_anchor": tail_anchor.detach(),
        "single_spike": single_spike.detach(),
        "source_alignment": source_alignment.detach(),
        "source_reliability": source_reliability.detach(),
        "single_source_credibility": single_source_credibility.detach(),
        "rerank_l2": rerank_l2.detach(),
        "usage_entropy": usage_entropy.detach(),
    }


def _pairwise_rank_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    max_pairs: int,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    n = int(score.numel())
    if n < 2 or max_pairs <= 0:
        return (score - target).pow(2).mean()
    pair_count = min(int(max_pairs), n * max(1, n - 1))
    idx_i = torch.randint(n, (pair_count,), device=score.device)
    idx_j = torch.randint(n, (pair_count,), device=score.device)
    diff_target = target[idx_i] - target[idx_j]
    keep = diff_target.abs() >= float(margin)
    if not bool(keep.any().detach().cpu().item()):
        return (score - target).pow(2).mean()
    idx_i = idx_i[keep]
    idx_j = idx_j[keep]
    diff_target = diff_target[keep]
    direction = diff_target.sign()
    diff_score = (score[idx_i] - score[idx_j]) / max(float(temperature), 1e-4)
    weights = (confidence[idx_i] * confidence[idx_j] * diff_target.abs()).clamp_min(1e-6)
    loss = F.softplus(-direction * diff_score) * weights
    return loss.sum() / weights.sum().clamp_min(1e-6)


def _top_tail_contrast_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    q_high: float,
    q_low: float,
    max_pairs: int,
    temperature: float,
) -> torch.Tensor:
    n = int(score.numel())
    if n < 2 or max_pairs <= 0:
        return (score - target).pow(2).mean()
    q_high = max(0.0, min(1.0, float(q_high)))
    q_low = max(0.0, min(q_high, float(q_low)))
    high_cut = torch.quantile(target.float(), q_high)
    low_cut = torch.quantile(target.float(), q_low)
    high = torch.nonzero(target >= high_cut, as_tuple=False).reshape(-1)
    low = torch.nonzero(target <= low_cut, as_tuple=False).reshape(-1)
    if high.numel() == 0 or low.numel() == 0:
        return (score - target).pow(2).mean()

    total_pairs = int(high.numel() * low.numel())
    pair_count = min(int(max_pairs), total_pairs)
    if total_pairs <= pair_count:
        idx_i = high.repeat_interleave(low.numel())
        idx_j = low.repeat(high.numel())
    else:
        idx_i = high[torch.randint(high.numel(), (pair_count,), device=score.device)]
        idx_j = low[torch.randint(low.numel(), (pair_count,), device=score.device)]

    target_gap = (target[idx_i] - target[idx_j]).clamp_min(0.0)
    weights = (confidence[idx_i] * confidence[idx_j] * target_gap).clamp_min(1e-6)
    diff_score = (score[idx_i] - score[idx_j]) / max(float(temperature), 1e-4)
    loss = F.softplus(-diff_score) * weights
    return loss.sum() / weights.sum().clamp_min(1e-6)


def _extreme_tail_listwise_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    q: float,
    max_nodes: int,
    temperature: float,
    target_temperature: float,
    attn_cfg: Any,
    inputs: Optional[RNRv31AttentionInputs] = None,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
    credibility: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool(_cfg_get(attn_cfg, "adaptive_extreme_tail", True)) and inputs is not None and target_inputs is not None:
        return _adaptive_extreme_tail_listwise_loss(
            score,
            target,
            confidence,
            inputs,
            target_inputs,
            attn_cfg,
            max_nodes=max_nodes,
            temperature=temperature,
            target_temperature=target_temperature,
            credibility=credibility,
        )

    n = int(score.numel())
    if n < 2 or int(max_nodes) <= 1:
        return (score - target).pow(2).mean(), score.new_tensor(0.0), score.new_tensor(0.0)
    q = max(0.0, min(1.0, float(q)))
    k = max(2, int(math.ceil(q * n)))
    k = min(k, n, max(2, int(max_nodes) // 2))
    top_target = torch.topk(target.float(), k=k).indices
    top_score = torch.topk(score.detach().float(), k=k).indices
    candidate = torch.unique(torch.cat([top_target, top_score], dim=0))
    if candidate.numel() < 2:
        return (score - target).pow(2).mean(), score.new_tensor(float(candidate.numel())), score.new_tensor(0.0)
    if candidate.numel() > int(max_nodes):
        keep_score = torch.maximum(score.detach()[candidate].float(), target.detach()[candidate].float())
        order = torch.topk(keep_score, k=int(max_nodes)).indices
        candidate = candidate[order]
    pred_log_prob = F.log_softmax(score[candidate] / max(float(temperature), 1e-4), dim=0)
    target_logits = target.detach()[candidate] / max(float(target_temperature), 1e-4)
    target_prob = torch.softmax(target_logits, dim=0)
    conf = confidence.detach()[candidate].clamp_min(1e-6)
    target_prob = target_prob * conf
    target_prob = target_prob / target_prob.sum().clamp_min(1e-6)
    threshold = target.detach()[top_target].min() if top_target.numel() > 0 else score.new_tensor(0.0)
    return -(target_prob * pred_log_prob).sum(), score.new_tensor(float(candidate.numel())), threshold.to(score)


def _adaptive_extreme_tail_listwise_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    inputs: RNRv31AttentionInputs,
    target_inputs: RNRv31AttentionInputs,
    attn_cfg: Any,
    max_nodes: int,
    temperature: float,
    target_temperature: float,
    credibility: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(score.numel())
    if n < 2 or int(max_nodes) <= 1:
        zero = score.new_tensor(0.0)
        return (score - target).pow(2).mean(), zero, zero

    current_source = _source_summary_scores(inputs, attn_cfg).to(score)
    heldout_source = _source_summary_scores(target_inputs, attn_cfg).to(score)
    stable_by_source = torch.minimum(current_source, heldout_source)
    stable_tail = stable_by_source.max(dim=1).values
    heldout_tail = heldout_source.max(dim=1).values
    target_rank = rank_norm(target.detach()).reshape(-1).to(score)
    credibility_term = credibility.detach().reshape(-1).to(score).clamp(0.0, 1.0) if credibility is not None else score.new_ones(n)

    target_mix = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "adaptive_tail_target_mix", 0.10))))
    credibility_mix = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "adaptive_tail_credibility_mix", 0.25))))
    stable_tail = rank_norm(stable_tail.detach()).to(score)
    heldout_tail = rank_norm(heldout_tail.detach()).to(score)
    pseudo_tail = (1.0 - target_mix) * stable_tail + target_mix * target_rank
    pseudo_tail = (1.0 - credibility_mix) * pseudo_tail + credibility_mix * credibility_term * heldout_tail
    pseudo_tail = rank_norm(pseudo_tail.detach()).to(score)

    center = pseudo_tail.median()
    mad = (pseudo_tail - center).abs().median().clamp_min(1e-6)
    threshold = center + float(_cfg_get(attn_cfg, "adaptive_tail_mad_scale", 1.8)) * mad
    min_quantile = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "adaptive_tail_min_quantile", 0.90))))
    threshold = torch.maximum(threshold, torch.quantile(pseudo_tail.float(), min_quantile).to(score))
    raw_count = int((pseudo_tail >= threshold).sum().detach().cpu().item())
    min_count = max(2, int(_cfg_get(attn_cfg, "adaptive_tail_min_nodes", 8)))
    max_count = max(min_count, min(int(max_nodes), n))
    base_count = max(min_count, min(max_count, raw_count))

    multipliers = [float(item) for item in list(_cfg_get(attn_cfg, "adaptive_tail_window_multipliers", [1.0, 2.0, 4.0]))]
    losses = []
    for multiplier in multipliers:
        window = max(2, min(max_count, int(math.ceil(base_count * max(float(multiplier), 1e-4)))))
        losses.append(
            _listwise_window_loss(
                score,
                pseudo_tail,
                confidence,
                window=window,
                max_nodes=max_nodes,
                temperature=temperature,
                target_temperature=target_temperature,
            )
        )
    if not losses:
        zero = score.new_tensor(0.0)
        return zero, score.new_tensor(float(base_count)), threshold.detach()
    return torch.stack(losses).mean(), score.new_tensor(float(base_count)), threshold.detach()


def _listwise_window_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    window: int,
    max_nodes: int,
    temperature: float,
    target_temperature: float,
) -> torch.Tensor:
    n = int(score.numel())
    window = max(2, min(int(window), n, int(max_nodes)))
    top_target = torch.topk(target.float(), k=window).indices
    top_score = torch.topk(score.detach().float(), k=window).indices
    candidate = torch.unique(torch.cat([top_target, top_score], dim=0))
    if candidate.numel() < 2:
        return score.sum() * 0.0
    if candidate.numel() > int(max_nodes):
        keep_score = torch.maximum(score.detach()[candidate].float(), target.detach()[candidate].float())
        order = torch.topk(keep_score, k=int(max_nodes)).indices
        candidate = candidate[order]
    pred_log_prob = F.log_softmax(score[candidate] / max(float(temperature), 1e-4), dim=0)
    target_logits = target.detach()[candidate] / max(float(target_temperature), 1e-4)
    target_prob = torch.softmax(target_logits, dim=0)
    weights = confidence.detach()[candidate].clamp_min(1e-6)
    target_prob = target_prob * weights
    target_prob = target_prob / target_prob.sum().clamp_min(1e-6)
    return -(target_prob * pred_log_prob).sum()


def _source_summary_scores(inputs: RNRv31AttentionInputs, attn_cfg: Any) -> torch.Tensor:
    token_scores = inputs.token_scores.float()
    source_ids = inputs.source_ids.to(token_scores.device)
    summaries = []
    for source_idx in (0, 1):
        mask = source_ids == source_idx
        if bool(mask.any().detach().cpu().item()):
            summaries.append(token_scores[:, mask].mean(dim=1))
        else:
            summaries.append(token_scores.new_zeros(token_scores.size(0)))

    component_mask = source_ids == 2
    if bool(component_mask.any().detach().cpu().item()):
        component_scores = token_scores[:, component_mask]
        reducer = str(_cfg_get(attn_cfg, "tail_anchor_component_reducer", "max"))
        if reducer == "mean":
            component_summary = component_scores.mean(dim=1)
        elif reducer == "top2_mean":
            k = min(2, component_scores.size(1))
            component_summary = torch.topk(component_scores, k=k, dim=1).values.mean(dim=1)
        else:
            component_summary = component_scores.max(dim=1).values
    else:
        component_summary = token_scores.new_zeros(token_scores.size(0))
    summaries.append(component_summary)
    return torch.stack(summaries, dim=1)


def _tail_anchor_summary(
    inputs: Optional[RNRv31AttentionInputs],
    attn_cfg: Any,
    target: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    if inputs is None:
        return None
    source_scores = _source_summary_scores(inputs, attn_cfg)
    top_values = torch.topk(source_scores, k=min(2, source_scores.size(1)), dim=1).values
    max_source = top_values[:, 0]
    second_source = top_values[:, -1] if top_values.size(1) > 1 else max_source
    mode = str(_cfg_get(attn_cfg, "tail_anchor_mode", "second_source"))
    if mode == "mean_source":
        anchor = source_scores.mean(dim=1)
    elif mode == "max_second_mean":
        anchor = 0.5 * (max_source + second_source)
    else:
        anchor = second_source
    if target is not None:
        target_mix = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "tail_anchor_target_mix", 0.0))))
        if target_mix > 0.0:
            anchor = (1.0 - target_mix) * anchor + target_mix * rank_norm(target.detach()).reshape(-1).to(anchor)
    return {
        "anchor": rank_norm(anchor.detach()),
        "max_source": max_source.detach(),
        "second_source": second_source.detach(),
        "single_source_gap": (max_source - second_source).detach(),
    }


def _tail_anchor_losses(
    score: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    inputs: Optional[RNRv31AttentionInputs],
    attn_cfg: Any,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summary = _tail_anchor_summary(inputs, attn_cfg, target=target)
    if summary is None:
        zero = score.sum() * 0.0
        return zero, zero

    anchor = summary["anchor"].to(score)
    max_source = summary["max_source"].to(score)
    second_source = summary["second_source"].to(score)
    q_high = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "tail_anchor_q_high", 0.95))))
    q_low = max(0.0, min(q_high, float(_cfg_get(attn_cfg, "tail_anchor_q_low", 0.60))))
    high_cut = torch.quantile(anchor.float(), q_high)
    low_cut = torch.quantile(torch.maximum(max_source.float(), target.float()), q_low)
    high = torch.nonzero(anchor >= high_cut, as_tuple=False).reshape(-1)
    low = torch.nonzero((max_source <= low_cut) & (target <= torch.quantile(target.float(), q_low)), as_tuple=False).reshape(-1)
    tail_anchor = _indexed_pair_loss(
        score,
        high,
        low,
        confidence,
        gap=(anchor - anchor.median()).clamp_min(0.0),
        max_pairs=int(_cfg_get(attn_cfg, "max_tail_anchor_pairs", _cfg_get(attn_cfg, "max_rank_pairs", 20_000))),
        temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
        eps=eps,
    )

    spike_high_q = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "single_spike_q_high", 0.95))))
    spike_low_q = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "single_spike_q_second_low", 0.70))))
    spike_cut = torch.quantile(max_source.float(), spike_high_q)
    second_low_cut = torch.quantile(second_source.float(), spike_low_q)
    corroborated = high
    single_spike = torch.nonzero((max_source >= spike_cut) & (second_source <= second_low_cut), as_tuple=False).reshape(-1)
    single_spike_loss = _indexed_pair_loss(
        score,
        corroborated,
        single_spike,
        confidence,
        gap=(anchor - second_source).abs(),
        max_pairs=int(_cfg_get(attn_cfg, "max_single_spike_pairs", _cfg_get(attn_cfg, "max_tail_anchor_pairs", 20_000))),
        temperature=float(_cfg_get(attn_cfg, "rank_temperature", 0.1)),
        eps=eps,
    )
    return tail_anchor, single_spike_loss


def _source_alignment_loss(
    output: RNRv31AttentionOutput,
    inputs: Optional[RNRv31AttentionInputs],
    target: torch.Tensor,
    attn_cfg: Any,
    eps: float,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
) -> torch.Tensor:
    if inputs is None:
        return output.score_raw.sum() * 0.0
    target_dist = _source_quality_target(inputs, target, attn_cfg, target_inputs=target_inputs).to(output.score_raw)
    predicted = output.source_weights.mean(dim=0).clamp_min(eps)
    return -(target_dist * predicted.log()).sum()


def _source_reliability_loss(
    output: RNRv31AttentionOutput,
    inputs: Optional[RNRv31AttentionInputs],
    target: torch.Tensor,
    attn_cfg: Any,
    eps: float,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
) -> torch.Tensor:
    if inputs is None:
        return output.score_raw.sum() * 0.0
    target_dist = _source_quality_target(inputs, target, attn_cfg, target_inputs=target_inputs).to(output.score_raw)
    predicted = output.source_reliability.clamp_min(eps)
    return -(target_dist * predicted.log()).sum()


def _source_quality_target(
    inputs: RNRv31AttentionInputs,
    target: torch.Tensor,
    attn_cfg: Any,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
) -> torch.Tensor:
    source_scores = _source_summary_scores(inputs, attn_cfg)
    source_ids = inputs.source_ids.to(source_scores.device)
    present = torch.stack([(source_ids == idx).any() for idx in range(source_scores.size(1))])
    top_tail_q = float(_cfg_get(attn_cfg, "source_alignment_top_tail_q", _cfg_get(attn_cfg, "top_tail_q", 0.05)))
    if target_inputs is not None:
        target_source_scores = _source_summary_scores(target_inputs, attn_cfg).to(source_scores)
    else:
        target_rank = rank_norm(target.detach()).reshape(-1).to(source_scores)
        target_source_scores = target_rank.reshape(-1, 1).expand_as(source_scores)

    qualities = []
    for idx in range(source_scores.size(1)):
        score = source_scores[:, idx]
        target_score = rank_norm(target_source_scores[:, idx].detach()).reshape(-1).to(score)
        corr = ((spearman_corr(score, target_score) + 1.0) * 0.5).clamp(0.0, 1.0)
        overlap = _top_tail_overlap(score, target_score, top_tail_q).to(score).clamp(0.0, 1.0)
        signal = _signal(score).clamp_min(0.0)
        qualities.append(0.45 * corr + 0.45 * overlap + 0.10 * signal.clamp_max(1.0))
    quality = torch.stack(qualities).detach().masked_fill(~present, -1e9)
    temperature = max(float(_cfg_get(attn_cfg, "source_alignment_temperature", 0.25)), 1e-4)
    return torch.softmax(quality / temperature, dim=0)


def _single_source_credibility_loss(
    output: RNRv31AttentionOutput,
    inputs: Optional[RNRv31AttentionInputs],
    target: torch.Tensor,
    confidence: torch.Tensor,
    attn_cfg: Any,
    eps: float,
    target_inputs: Optional[RNRv31AttentionInputs] = None,
) -> torch.Tensor:
    if inputs is None:
        return output.score_raw.sum() * 0.0
    source_scores = _source_summary_scores(inputs, attn_cfg).to(output.score_raw)
    top_values = torch.topk(source_scores, k=min(2, source_scores.size(1)), dim=1).values
    max_source = top_values[:, 0]
    second_source = top_values[:, -1] if top_values.size(1) > 1 else max_source
    gap = max_source - second_source

    if target_inputs is not None:
        target_source = _source_summary_scores(target_inputs, attn_cfg).to(output.score_raw)
        target_max = target_source.max(dim=1).values
    else:
        target_max = rank_norm(target.detach()).reshape(-1).to(output.score_raw)

    high_q = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "single_source_credibility_q_high", 0.95))))
    low_q = max(0.0, min(high_q, float(_cfg_get(attn_cfg, "single_source_credibility_q_low", 0.60))))
    gap_q = max(0.0, min(1.0, float(_cfg_get(attn_cfg, "single_source_credibility_gap_q", 0.75))))
    high_cut = torch.quantile(max_source.float(), high_q)
    target_high = torch.quantile(target_max.float(), high_q)
    target_low = torch.quantile(target_max.float(), low_q)
    gap_cut = torch.quantile(gap.float(), gap_q)

    current_high = max_source >= high_cut
    singleish = gap >= gap_cut
    positive = current_high & singleish & (target_max >= target_high)
    negative = current_high & singleish & (target_max <= target_low)
    keep = positive | negative
    if not bool(keep.any().detach().cpu().item()):
        return output.score_raw.sum() * 0.0
    pseudo = positive[keep].float()
    pred = output.single_source_credibility[keep].clamp(eps, 1.0 - eps)
    weights = (confidence.detach()[keep] * gap.detach()[keep].abs().clamp_min(eps)).to(pred)
    loss = F.binary_cross_entropy(pred, pseudo.to(pred), weight=weights, reduction="sum")
    return loss / weights.sum().clamp_min(eps)


def _indexed_pair_loss(
    score: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    confidence: torch.Tensor,
    gap: torch.Tensor,
    max_pairs: int,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    if high.numel() == 0 or low.numel() == 0 or int(max_pairs) <= 0:
        return score.sum() * 0.0
    total_pairs = int(high.numel() * low.numel())
    pair_count = min(int(max_pairs), total_pairs)
    if total_pairs <= pair_count:
        idx_i = high.repeat_interleave(low.numel())
        idx_j = low.repeat(high.numel())
    else:
        idx_i = high[torch.randint(high.numel(), (pair_count,), device=score.device)]
        idx_j = low[torch.randint(low.numel(), (pair_count,), device=score.device)]
    diff_score = (score[idx_i] - score[idx_j]) / max(float(temperature), 1e-4)
    weights = (confidence[idx_i] * confidence[idx_j] * (gap[idx_i] + eps)).clamp_min(eps)
    loss = F.softplus(-diff_score) * weights
    return loss.sum() / weights.sum().clamp_min(eps)


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return 0.5 * ((p * (p / q).log()).sum(dim=1) + (q * (q / p).log()).sum(dim=1)).mean()
