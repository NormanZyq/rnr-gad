from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch_geometric.utils import add_self_loops, coalesce

from src.models.components.rank_utils import rank_norm, spearman_corr
from src.models.components.rnr_gad import (
    CrossFittedResult,
    RankSafeProjector,
    _cauchy_combine_probabilities,
    _cfg_get,
    _degree,
    _empirical_upper_tail_probability,
    _robust_scale,
    _section_get,
)


@dataclass
class RNRv3Result:
    score: torch.Tensor
    candidate_score: torch.Tensor
    confidence: torch.Tensor
    component_names: List[str]
    component_probabilities: torch.Tensor
    component_scores: torch.Tensor
    component_weights: torch.Tensor
    component_eligible: torch.Tensor
    raw_component_names: List[str]
    raw_component_scores: torch.Tensor
    propagation_profile: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _as_float_tensor(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _column_robust_z(values: torch.Tensor, eps: float) -> torch.Tensor:
    cols = []
    for idx in range(values.size(1)):
        col = values[:, idx].float()
        center = col.median()
        scale = _robust_scale(col, eps)
        cols.append((col - center) / scale.clamp_min(eps))
    return torch.stack(cols, dim=1)


def _row_propagate(values: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index.long(), num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    out = values.new_zeros(values.shape)
    degree = values.new_zeros(num_nodes)
    out.index_add_(0, dst, values[src])
    degree.index_add_(0, dst, values.new_ones(src.numel()))
    return out / degree.clamp_min(1.0).reshape(-1, 1)


def _weighted_row_propagate(
    values: torch.Tensor,
    weights: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    eps: float,
) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index.long(), num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    node_weights = weights.reshape(-1, 1).to(values).clamp_min(eps)
    out = values.new_zeros(values.shape)
    denom = values.new_zeros(num_nodes, 1)
    out.index_add_(0, dst, values[src] * node_weights[src])
    denom.index_add_(0, dst, node_weights[src])
    return out / denom.clamp_min(eps)


def _node_dirichlet(values: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index.long(), num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    degree = _degree(edge_index, num_nodes, values.dtype).clamp_min(1.0)
    src, dst = edge_index
    left = values[src] / degree[src].sqrt().reshape(-1, 1)
    right = values[dst] / degree[dst].sqrt().reshape(-1, 1)
    edge_energy = (left - right).pow(2).mean(dim=1)
    out = values.new_zeros(num_nodes)
    count = values.new_zeros(num_nodes)
    out.index_add_(0, src, edge_energy)
    out.index_add_(0, dst, edge_energy)
    count.index_add_(0, src, values.new_ones(src.numel()))
    count.index_add_(0, dst, values.new_ones(src.numel()))
    return out / count.clamp_min(1.0)


def _top_tail_overlap(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    q: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(scores_a.numel())
    if n == 0:
        zero = scores_a.new_tensor(0.0)
        return zero, zero
    k = max(1, int(torch.ceil(scores_a.new_tensor(float(q * n))).item()))
    top_a = torch.topk(scores_a.reshape(-1), k).indices
    top_b = torch.topk(scores_b.reshape(-1), k).indices
    marker = torch.zeros(n, dtype=torch.bool, device=scores_a.device)
    marker[top_a] = True
    overlap = marker[top_b].float().sum()
    return overlap / float(k), overlap / (float(q) * float(q) * float(n) + eps)


class RNRv3Scorer:
    """Evidence-field propagation scorer for correction initialization."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.eps = float(_section_get(cfg, "crossfit", "eps", 1e-8))
        self.projector = RankSafeProjector(
            epsilon_max=float(_section_get(cfg, "rank_safety", "epsilon_max", 0.05)),
            enabled=bool(_section_get(cfg, "rank_safety", "enabled", True)),
            max_pairs=int(_section_get(cfg, "rank_safety", "max_pairs", 200_000)),
        )
        self.diagnostic_projector = RankSafeProjector(
            epsilon_max=1.0,
            enabled=False,
            max_pairs=int(_section_get(cfg, "rank_safety", "max_pairs", 200_000)),
        )

    def compute(self, data: Any, stage1: CrossFittedResult) -> RNRv3Result:
        raw_scores, profile = self._raw_propagation_scores(data, stage1)
        raw_names = list(raw_scores.keys())
        raw_tensor = torch.stack([raw_scores[name] for name in raw_names], dim=1)

        component_names, component_probabilities = self._component_probabilities(raw_scores, stage1)
        include_stage1 = bool(_section_get(self.cfg, "propagation_score", "include_stage1_prior", True))
        if include_stage1:
            component_names = ["stage1", *component_names]
            component_probabilities = [
                stage1.combined_probability.detach().clamp(self.eps, 1.0),
                *component_probabilities,
            ]
        if not component_probabilities:
            component_names = ["stage1"]
            component_probabilities = [stage1.combined_probability.detach().clamp(self.eps, 1.0)]

        component_tensor = torch.stack(component_probabilities, dim=1)
        component_scores = -torch.log(component_tensor.clamp_min(self.eps))
        component_weights, component_eligible = self._component_weights(
            component_names,
            component_scores,
            stage1.score,
        )
        combined_p = _cauchy_combine_probabilities(
            component_tensor,
            component_weights,
            self.eps,
            bool(_section_get(self.cfg, "aggregation", "ignore_large_p", True)),
        )
        candidate_score = rank_norm(-torch.log(combined_p.clamp_min(self.eps)))
        confidence = self._confidence(stage1.confidence, component_scores)

        if bool(_section_get(self.cfg, "propagation_score", "project_to_stage1", False)):
            projected = self.projector.project(stage1.score, candidate_score, confidence)
            final_score = projected["score"]
        else:
            projected = self.diagnostic_projector.project(stage1.score, candidate_score, confidence)
            final_score = candidate_score

        diagnostics = self._diagnostics(
            raw_names,
            raw_tensor,
            component_names,
            component_scores,
            component_weights,
            component_eligible,
            stage1,
            candidate_score,
            final_score,
            confidence,
            projected,
        )
        return RNRv3Result(
            score=final_score.detach(),
            candidate_score=candidate_score.detach(),
            confidence=confidence.detach(),
            component_names=component_names,
            component_probabilities=component_tensor.detach(),
            component_scores=component_scores.detach(),
            component_weights=component_weights.detach(),
            component_eligible=component_eligible.detach(),
            raw_component_names=raw_names,
            raw_component_scores=raw_tensor.detach(),
            propagation_profile=profile.detach(),
            diagnostics=diagnostics,
        )

    def _raw_propagation_scores(
        self,
        data: Any,
        stage1: CrossFittedResult,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        edge_index = data.edge_index.long()
        num_nodes = int(data.num_nodes)
        hops = max(1, int(_section_get(self.cfg, "propagation_score", "hops", 4)))

        evidence = _column_robust_z(stage1.view_scores.float(), self.eps)
        group_evidence = _column_robust_z(stage1.group_scores.float(), self.eps)
        stage_signal = rank_norm(stage1.score.detach()).reshape(-1, 1).to(evidence)
        profile = torch.cat([evidence, group_evidence, stage_signal], dim=1)

        sequence = [evidence]
        for _ in range(hops):
            sequence.append(_row_propagate(sequence[-1], edge_index, num_nodes))
        one_hop = sequence[1]
        anchor = sequence[-1]

        normal_neighbor = _weighted_row_propagate(
            evidence,
            stage1.combined_probability.detach(),
            edge_index,
            num_nodes,
            self.eps,
        )
        profile_neighbor = _weighted_row_propagate(
            profile,
            stage1.combined_probability.detach(),
            edge_index,
            num_nodes,
            self.eps,
        )
        stage_one_hop = _row_propagate(stage_signal, edge_index, num_nodes)

        propagated_parts = [stage_one_hop.reshape(-1)]
        propagated_parts.extend(item.mean(dim=1) for item in sequence[1:])

        raw_scores = {
            "evidence_highpass": (evidence - one_hop).pow(2).mean(dim=1).sqrt(),
            "evidence_abs_excess": (evidence - one_hop).abs().mean(dim=1),
            "evidence_positive_excess": (evidence - one_hop).clamp_min(0.0).mean(dim=1),
            "normal_neighbor_residual": (evidence - normal_neighbor).pow(2).mean(dim=1).sqrt(),
            "normal_neighbor_positive_residual": (evidence - normal_neighbor).clamp_min(0.0).mean(dim=1),
            "profile_normal_residual": (profile - profile_neighbor).pow(2).mean(dim=1).sqrt(),
            "trajectory_area": torch.stack(
                [(item - anchor).pow(2).mean(dim=1).sqrt() for item in sequence[:-1]],
                dim=1,
            ).mean(dim=1),
            "trajectory_velocity": torch.stack(
                [(sequence[idx] - sequence[idx - 1]).pow(2).mean(dim=1).sqrt() for idx in range(1, len(sequence))],
                dim=1,
            ).mean(dim=1),
            "evidence_dirichlet": _node_dirichlet(evidence, edge_index, num_nodes),
            "profile_dirichlet": _node_dirichlet(profile, edge_index, num_nodes),
            "stage1_harmonic_abs": (stage_signal - stage_one_hop).abs().reshape(-1),
            "stage1_harmonic_pos": (stage_signal - stage_one_hop).clamp_min(0.0).reshape(-1),
            "propagated_stage1": torch.stack(propagated_parts, dim=1).mean(dim=1),
        }
        return {
            name: torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
            for name, score in raw_scores.items()
        }, profile

    def _component_probabilities(
        self,
        raw_scores: Dict[str, torch.Tensor],
        stage1: CrossFittedResult,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        enabled = _section_get(self.cfg, "propagation_score", "enabled_components", None)
        enabled_names = list(enabled) if enabled is not None else list(raw_scores.keys())
        names = []
        probabilities = []
        for raw_name in enabled_names:
            if raw_name not in raw_scores:
                continue
            for orientation in self._orientations_for(raw_name):
                names.append(f"{raw_name}_{orientation}")
                probabilities.append(self._tail_probability(raw_scores[raw_name], stage1.degree_bins, orientation))
        return names, probabilities

    def _orientations_for(self, raw_name: str) -> List[str]:
        policy = str(_section_get(self.cfg, "propagation_score", "orientation_policy", "configured"))
        if policy == "all":
            return ["high", "low", "two"]
        orientation_cfg = _cfg_get(_cfg_get(self.cfg, "propagation_score", None), "component_orientations", None)
        value = _cfg_get(orientation_cfg, raw_name, None)
        if value is None:
            value = _section_get(self.cfg, "propagation_score", "default_orientations", ["low", "two"])
        orientations = [str(item) for item in list(value)]
        valid = [item for item in orientations if item in {"high", "low", "two"}]
        return valid or ["low", "two"]

    def _tail_probability(
        self,
        values: torch.Tensor,
        strata: torch.Tensor,
        orientation: str,
    ) -> torch.Tensor:
        high = _empirical_upper_tail_probability(values.reshape(-1).float(), strata.long(), self.eps)
        if orientation == "high":
            return high
        low = _empirical_upper_tail_probability((-values).reshape(-1).float(), strata.long(), self.eps)
        if orientation == "low":
            return low
        return (2.0 * torch.minimum(high, low)).clamp(self.eps, 1.0)

    def _component_weights(
        self,
        component_names: List[str],
        component_scores: torch.Tensor,
        stage1_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = _cfg_get(self.cfg, "propagation_score", None)
        min_signal = float(_cfg_get(cfg, "min_component_signal", 0.05))
        max_spearman = float(_cfg_get(cfg, "max_component_spearman", 0.995))
        max_tail_overlap = float(_cfg_get(cfg, "max_tail_overlap_fraction", 1.0))
        top_tail_q = float(_cfg_get(cfg, "top_tail_q", 0.05))
        component_weight = float(_cfg_get(cfg, "component_weight", 1.0))
        require_eligibility = bool(_cfg_get(cfg, "require_component_eligibility", True))
        stage1_prior_weight = float(_cfg_get(cfg, "stage1_prior_weight", 0.5))

        weights = component_scores.new_zeros(len(component_names))
        eligible = component_scores.new_zeros(len(component_names))
        for idx, name in enumerate(component_names):
            if name == "stage1":
                weights[idx] = max(stage1_prior_weight, self.eps)
                eligible[idx] = 1.0
                continue
            score = component_scores[:, idx]
            signal = torch.quantile(score.float(), 0.95) - score.median()
            corr = spearman_corr(score, stage1_score)
            tail_overlap, _ = _top_tail_overlap(score, stage1_score, top_tail_q, self.eps)
            ok = (signal >= min_signal) & (corr.abs() <= max_spearman) & (tail_overlap <= max_tail_overlap)
            eligible[idx] = ok.to(component_scores)
            if bool(ok.detach().cpu().item()) or not require_eligibility:
                weights[idx] = component_weight

        if float(weights.sum().detach().cpu()) <= self.eps:
            if "stage1" in component_names:
                weights[component_names.index("stage1")] = 1.0
            else:
                signal = torch.quantile(component_scores.float(), 0.95, dim=0) - component_scores.median(dim=0).values
                weights[int(torch.argmax(signal).detach().cpu())] = 1.0
        return weights / weights.sum().clamp_min(self.eps), eligible

    def _confidence(self, stage1_confidence: torch.Tensor, component_scores: torch.Tensor) -> torch.Tensor:
        if component_scores.size(1) <= 1:
            return stage1_confidence
        center = component_scores.median(dim=1, keepdim=True).values
        disagreement = (component_scores - center).abs().median(dim=1).values
        scale = disagreement.median()
        if float(scale.detach().cpu()) <= self.eps:
            scale = disagreement.mean().clamp_min(self.eps)
        return (stage1_confidence * torch.exp(-disagreement / scale.clamp_min(self.eps))).clamp(0.0, 1.0)

    def _diagnostics(
        self,
        raw_names: List[str],
        raw_tensor: torch.Tensor,
        component_names: List[str],
        component_scores: torch.Tensor,
        component_weights: torch.Tensor,
        component_eligible: torch.Tensor,
        stage1: CrossFittedResult,
        candidate_score: torch.Tensor,
        final_score: torch.Tensor,
        confidence: torch.Tensor,
        projected: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = final_score.device
        non_stage = [idx for idx, name in enumerate(component_names) if name != "stage1"]
        non_stage_eligible = (
            component_eligible[non_stage].max()
            if non_stage
            else component_scores.new_tensor(0.0)
        )
        diagnostics = {
            "rnrv3/raw_component_count": _as_float_tensor(float(len(raw_names)), device),
            "rnrv3/component_count": _as_float_tensor(float(len(component_names)), device),
            "rnrv3/eligible": torch.maximum(component_eligible.max(), non_stage_eligible),
            "rnrv3/non_stage_eligible": non_stage_eligible,
            "rnrv3/confidence_mean": confidence.mean(),
            "rnrv3/stage1_candidate_spearman": spearman_corr(stage1.score, candidate_score),
            "rnrv3/stage1_final_spearman": spearman_corr(stage1.score, final_score),
            "rnrv3/rank_safe_spearman_base": projected["spearman_base"],
            "rnrv3/rank_safe_max_rank_displacement": projected["max_rank_displacement"],
            "rnrv3/rank_safe_flip_ratio": projected["rank_flip_ratio"],
            "rnrv3/rank_safe_protected_pair_violations": projected["protected_pair_violations"],
        }

        for idx, name in enumerate(raw_names):
            raw = raw_tensor[:, idx]
            diagnostics[f"rnrv3/raw_signal/{name}"] = torch.quantile(raw.float(), 0.95) - raw.median()

        for idx, name in enumerate(component_names):
            score = component_scores[:, idx]
            tail_overlap, tail_random_ratio = _top_tail_overlap(
                score,
                stage1.score,
                float(_section_get(self.cfg, "propagation_score", "top_tail_q", 0.05)),
                self.eps,
            )
            diagnostics[f"rnrv3/component_weight/{name}"] = component_weights[idx]
            diagnostics[f"rnrv3/component_eligible/{name}"] = component_eligible[idx]
            diagnostics[f"rnrv3/component_signal/{name}"] = torch.quantile(score.float(), 0.95) - score.median()
            diagnostics[f"rnrv3/component_tail_overlap/{name}"] = tail_overlap
            diagnostics[f"rnrv3/component_tail_overlap_random_ratio/{name}"] = tail_random_ratio
            diagnostics[f"rnrv3/component_spearman_stage1/{name}"] = spearman_corr(score, stage1.score)
        return diagnostics
