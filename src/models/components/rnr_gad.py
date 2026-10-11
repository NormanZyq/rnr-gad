from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.decoders import AttrDecoder, DotEdgeDecoder
from src.models.components.encoders import build_encoder
from src.models.components.graph_utils import dense_knn_distance, safe_negative_sampling
from src.models.components.rank_utils import rank_norm, spearman_corr


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _section_get(cfg: Any, section: str, key: str, default: Any) -> Any:
    return _cfg_get(_cfg_get(cfg, section, None), key, default)


def _robust_scale(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(1.0)
    center = values.median()
    scale = 1.4826 * (values - center).abs().median()
    fallback = values.float().std(unbiased=False)
    scale = torch.where(scale > eps, scale, fallback)
    return scale.clamp_min(eps)


def _enabled_view_names(cfg: Any) -> List[str]:
    views_cfg = _cfg_get(cfg, "views", None)
    enabled = []
    if bool(_cfg_get(views_cfg, "attribute", True)):
        enabled.append("attribute")
    if bool(_cfg_get(views_cfg, "structure", True)):
        enabled.append("structure")
    if bool(_cfg_get(views_cfg, "context", True)):
        enabled.append("context")
    if bool(_cfg_get(views_cfg, "density", False)):
        enabled.append("density")
    return enabled or ["attribute"]


def _degree(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    degree = torch.zeros(num_nodes, device=edge_index.device, dtype=dtype)
    if edge_index.numel() == 0:
        return degree
    src, dst = edge_index
    ones = torch.ones(src.numel(), device=edge_index.device, dtype=dtype)
    degree.index_add_(0, src, ones)
    degree.index_add_(0, dst, ones)
    return degree


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute row-wise weighted medians for values shaped [N, K]."""
    if values.dim() == 1:
        return values
    if values.size(1) == 1:
        return values[:, 0]

    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if weights.numel() != values.size(1):
        raise ValueError("weighted_median received a weight vector with the wrong length.")
    if float(weights.clamp_min(0.0).sum().detach().cpu()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights.clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-8)

    sorted_values, order = torch.sort(values, dim=1)
    sorted_weights = weights[order]
    cumulative = sorted_weights.cumsum(dim=1)
    median_idx = (cumulative >= 0.5).to(torch.long).argmax(dim=1)
    return sorted_values.gather(1, median_idx.unsqueeze(1)).squeeze(1)


def context_prediction_error(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return z.new_zeros(num_nodes)
    src, dst = edge_index
    neighbor_sum = torch.zeros_like(z)
    neighbor_count = z.new_zeros(num_nodes)
    ones = z.new_ones(src.numel())

    neighbor_sum.index_add_(0, src, z[dst])
    neighbor_sum.index_add_(0, dst, z[src])
    neighbor_count.index_add_(0, src, ones)
    neighbor_count.index_add_(0, dst, ones)

    neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0).unsqueeze(-1)
    return (z - neighbor_mean).pow(2).mean(dim=-1)


def _neighbor_feature_statistics(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Dict[str, torch.Tensor]:
    if edge_index.numel() == 0:
        zeros = x.new_zeros(num_nodes)
        return {"neighbor_l2": zeros, "neighbor_cosine": zeros}

    src, dst = edge_index
    diff = (x[src] - x[dst]).pow(2).sum(dim=-1).sqrt()
    out = x.new_zeros(num_nodes)
    count = x.new_zeros(num_nodes)
    ones = x.new_ones(src.numel())
    out.index_add_(0, src, diff)
    out.index_add_(0, dst, diff)
    count.index_add_(0, src, ones)
    count.index_add_(0, dst, ones)
    neighbor_l2 = out / count.clamp_min(1.0)

    x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
    cosine = (x_norm[src] * x_norm[dst]).sum(dim=-1)
    cos_out = x.new_zeros(num_nodes)
    cos_out.index_add_(0, src, cosine)
    cos_out.index_add_(0, dst, cosine)
    neighbor_cosine = 1.0 - cos_out / count.clamp_min(1.0)
    return {"neighbor_l2": neighbor_l2, "neighbor_cosine": neighbor_cosine}


def _empirical_upper_tail_probability(
    values: torch.Tensor,
    strata: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values = values.reshape(-1)
    p = torch.empty_like(values, dtype=torch.float32)
    unique_strata = torch.unique(strata)
    for stratum in unique_strata:
        mask = strata == stratum
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        order = torch.argsort(values[idx], descending=True, stable=True)
        ranks = torch.arange(1, n + 1, device=values.device, dtype=torch.float32)
        p[idx[order]] = ranks / float(n + 1)
    return p.clamp_min(eps).clamp_max(1.0)


def _cauchy_combine_probabilities(
    probabilities: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    ignore_large_p: bool = False,
) -> torch.Tensor:
    p = probabilities.clamp(min=eps, max=1.0 - eps)
    if ignore_large_p:
        p = torch.minimum(p, p.new_full((), 0.5))
    weights = weights.to(device=p.device, dtype=p.dtype)
    weights = weights / weights.sum().clamp_min(eps)
    transformed = torch.tan(torch.pi * (0.5 - p)).clamp(min=-1e6, max=1e6)
    statistic = (weights.reshape(1, -1) * transformed).sum(dim=1)
    combined = 0.5 - torch.atan(statistic) / torch.pi
    return combined.clamp(min=eps, max=1.0)


@dataclass
class MaskedTaskBatch:
    x_input: torch.Tensor
    feature_mask: torch.Tensor
    message_edge_index: torch.Tensor
    pos_edge_index: torch.Tensor
    neg_edge_index: torch.Tensor


@dataclass
class CrossFittedResult:
    residuals: torch.Tensor
    standardized_residuals: torch.Tensor
    raw_view_scores: torch.Tensor
    view_scores: torch.Tensor
    tail_probabilities: torch.Tensor
    score: torch.Tensor
    combined_probability: torch.Tensor
    confidence: torch.Tensor
    view_weights: torch.Tensor
    view_scales: torch.Tensor
    view_names: List[str]
    view_groups: List[str]
    group_names: List[str]
    group_probabilities: torch.Tensor
    group_scores: torch.Tensor
    group_weights: torch.Tensor
    train_view_weights: torch.Tensor
    train_view_scales: torch.Tensor
    view_disagreement: torch.Tensor
    mask_instability: torch.Tensor
    degree_bins: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        dropout: float,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        if zero_init:
            last = self.net[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NormalityBackbone(nn.Module):
    """GCN normality model with node-wise self-supervised losses."""

    def __init__(self, in_dim: int, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        self.hidden_dim = int(_cfg_get(cfg, "hidden_dim", _cfg_get(cfg, "hid_dim", 64)))
        self.embedding_dim = int(_cfg_get(cfg, "embedding_dim", _cfg_get(cfg, "out_dim", 64)))
        dropout = float(_cfg_get(cfg, "dropout", 0.3))

        self.encoder = build_encoder(
            encoder_type=str(_cfg_get(cfg, "backbone", _cfg_get(cfg, "encoder_type", "gcn"))),
            in_dim=in_dim,
            hid_dim=self.hidden_dim,
            out_dim=self.embedding_dim,
            num_layers=int(_cfg_get(cfg, "num_layers", 2)),
            dropout=dropout,
        )
        self.attr_decoder = AttrDecoder(self.embedding_dim, self.hidden_dim, in_dim, dropout)
        self.edge_decoder = DotEdgeDecoder()
        self.adapter = ResidualAdapter(
            z_dim=self.embedding_dim,
            hidden_dim=int(_cfg_get(cfg, "adapter_dim", 32)),
            dropout=dropout,
            zero_init=bool(_cfg_get(cfg, "zero_init_adapter", True)),
        )
        self.view_names = _enabled_view_names(cfg)
        self._encoder_frozen = False

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self._encoder_frozen = True

    def train(self, mode: bool = True) -> "NormalityBackbone":
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        use_adapter: bool = False,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._encoder_frozen:
            with torch.no_grad():
                z0 = self.encoder(x, edge_index, edge_weight=edge_weight)
        else:
            z0 = self.encoder(x, edge_index, edge_weight=edge_weight)
        z = z0 + self.adapter(z0) if use_adapter else z0
        return z, z0

    def _sample_edge_holdout(
        self,
        edge_index: torch.Tensor,
        edge_holdout_rate: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_count = int(edge_index.size(1))
        if edge_count == 0:
            return edge_index, edge_index
        if edge_holdout_rate <= 0.0:
            return edge_index, edge_index

        holdout_count = max(1, int(round(edge_count * edge_holdout_rate)))
        if edge_count > 1:
            holdout_count = min(holdout_count, edge_count - 1)
        else:
            holdout_count = 1

        perm = torch.randperm(edge_count, device=edge_index.device)
        holdout_idx = perm[:holdout_count]
        keep_idx = perm[holdout_count:]
        return edge_index[:, keep_idx], edge_index[:, holdout_idx]

    def sample_task_batch(
        self,
        data: Any,
        feature_mask_rate: float,
        edge_holdout_rate: float,
        neg_ratio: float,
        eval_mode: bool = False,
    ) -> MaskedTaskBatch:
        x = data.x.float()
        edge_index = data.edge_index.long()
        num_nodes = int(data.num_nodes)

        if eval_mode:
            feature_mask = torch.ones_like(x, dtype=torch.bool)
        elif feature_mask_rate <= 0.0:
            feature_mask = torch.zeros_like(x, dtype=torch.bool)
        else:
            feature_mask = torch.rand_like(x, dtype=torch.float32) < feature_mask_rate
            if not feature_mask.any():
                feature_mask = torch.zeros_like(x, dtype=torch.bool)
                feature_mask.view(-1)[torch.randint(feature_mask.numel(), (1,), device=x.device)] = True

        x_input = x.clone()
        x_input[feature_mask] = 0.0

        message_edge_index, pos_edge_index = self._sample_edge_holdout(
            edge_index=edge_index,
            edge_holdout_rate=edge_holdout_rate,
        )
        neg_count = int(round(max(1, pos_edge_index.size(1)) * float(neg_ratio)))
        if pos_edge_index.numel() == 0:
            neg_count = 0
        neg_edge_index = safe_negative_sampling(edge_index, num_nodes, neg_count)

        return MaskedTaskBatch(
            x_input=x_input,
            feature_mask=feature_mask,
            message_edge_index=message_edge_index,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
        )

    def _message_edge_weight(
        self,
        message_edge_index: torch.Tensor,
        node_weights: Optional[torch.Tensor],
        beta: float,
    ) -> Optional[torch.Tensor]:
        if node_weights is None or beta <= 0.0 or message_edge_index.numel() == 0:
            return None
        src = message_edge_index[0]
        return node_weights.to(message_edge_index.device).float()[src].clamp_min(0.0).pow(beta)

    def nodewise_task_losses(
        self,
        data: Any,
        task: MaskedTaskBatch,
        use_adapter: bool = False,
        message_node_weights: Optional[torch.Tensor] = None,
        message_reweight_beta: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        x = data.x.float()
        edge_index = data.edge_index.long()
        num_nodes = int(data.num_nodes)
        edge_weight = self._message_edge_weight(
            task.message_edge_index,
            message_node_weights,
            message_reweight_beta,
        )
        z, z0 = self.encode(
            task.x_input,
            task.message_edge_index,
            use_adapter=use_adapter,
            edge_weight=edge_weight,
        )
        x_hat = self.attr_decoder(z)

        squared = (x_hat - x).pow(2)
        if task.feature_mask.any():
            mask_f = task.feature_mask.float()
            attr_loss = (squared * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
        else:
            attr_loss = squared.mean(dim=-1)

        pos_logits = self.edge_decoder(z, task.pos_edge_index)
        neg_logits = self.edge_decoder(z, task.neg_edge_index)
        structure_loss, edge_loss, edge_loss_index = self._structure_losses(
            task.pos_edge_index,
            task.neg_edge_index,
            pos_logits,
            neg_logits,
            num_nodes,
        )
        context_loss = context_prediction_error(z, edge_index, num_nodes)
        density_loss = dense_knn_distance(
            z,
            k=int(_section_get(self.cfg, "crossfit", "density_knn_k", 10)),
            max_nodes=int(_section_get(self.cfg, "crossfit", "max_dense_knn_nodes", 12_000)),
        )
        if density_loss.numel() != num_nodes:
            density_loss = z.new_zeros(num_nodes)

        return {
            "attribute": attr_loss,
            "structure": structure_loss,
            "context": context_loss,
            "density": density_loss,
            "edge_loss": edge_loss,
            "edge_loss_index": edge_loss_index,
            "z": z,
            "z0": z0,
            "x_hat": x_hat,
        }

    def _structure_losses(
        self,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = pos_edge_index.device
        dtype = pos_logits.dtype if pos_logits.numel() > 0 else torch.float32
        node_loss = torch.zeros(num_nodes, device=device, dtype=dtype)
        node_count = torch.zeros(num_nodes, device=device, dtype=dtype)
        edge_losses = []
        edge_indices = []

        def add(edge_index: torch.Tensor, logits: torch.Tensor, target: float) -> None:
            if logits.numel() == 0:
                return
            target_tensor = torch.full_like(logits, target)
            losses = F.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none")
            src, dst = edge_index
            node_loss.index_add_(0, src, losses)
            node_loss.index_add_(0, dst, losses)
            ones = torch.ones_like(losses)
            node_count.index_add_(0, src, ones)
            node_count.index_add_(0, dst, ones)
            edge_losses.append(losses)
            edge_indices.append(edge_index)

        add(pos_edge_index, pos_logits, 1.0)
        add(neg_edge_index, neg_logits, 0.0)

        if edge_losses:
            all_edge_loss = torch.cat(edge_losses, dim=0)
            all_edge_index = torch.cat(edge_indices, dim=1)
        else:
            all_edge_loss = pos_logits.new_empty(0)
            all_edge_index = pos_edge_index.new_empty((2, 0))
        return node_loss / node_count.clamp_min(1.0), all_edge_loss, all_edge_index


class CrossFittedScorer:
    """Build cross-fitted multi-view nonconformity scores without labels."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.learned_view_names = _enabled_view_names(cfg)
        self.eps = float(_section_get(cfg, "crossfit", "eps", 1e-8))

    def compute(
        self,
        backbone: NormalityBackbone,
        data: Any,
        num_masks: Optional[int] = None,
        use_adapter: bool = False,
        message_node_weights: Optional[torch.Tensor] = None,
        message_reweight_beta: float = 0.0,
    ) -> CrossFittedResult:
        mask_count = int(num_masks or _section_get(self.cfg, "crossfit", "num_eval_masks", 8))
        mask_count = max(1, mask_count)
        feature_mask_rate = float(_section_get(self.cfg, "crossfit", "feature_mask_rate", 0.3))
        edge_holdout_rate = float(_section_get(self.cfg, "crossfit", "edge_holdout_rate", 0.1))
        neg_ratio = float(_section_get(self.cfg, "crossfit", "neg_ratio", 1.0))

        was_training = backbone.training
        backbone.eval()
        per_mask = []
        with torch.no_grad():
            for _ in range(mask_count):
                task = backbone.sample_task_batch(
                    data,
                    feature_mask_rate=feature_mask_rate,
                    edge_holdout_rate=edge_holdout_rate,
                    neg_ratio=neg_ratio,
                    eval_mode=True,
                )
                losses = backbone.nodewise_task_losses(
                    data,
                    task,
                    use_adapter=use_adapter,
                    message_node_weights=message_node_weights,
                    message_reweight_beta=message_reweight_beta,
                )
                per_mask.append(torch.stack([losses[name].detach() for name in self.learned_view_names], dim=1))
        if was_training:
            backbone.train()

        learned_residuals = torch.stack(per_mask, dim=2)
        return self.summarize_residuals(data, learned_residuals)

    def summarize_residuals(
        self,
        data: Any,
        learned_residuals: torch.Tensor,
    ) -> CrossFittedResult:
        """Aggregate sampled residuals into Stage-1 evidence.

        Kept separate from sampling so experiments can reuse the same
        mask realizations without changing aggregation or mask budgets.
        """
        if learned_residuals.ndim != 3:
            raise ValueError(
                "learned_residuals must have shape [nodes, learned_views, masks], "
                f"got {tuple(learned_residuals.shape)}"
            )
        if int(learned_residuals.size(0)) != int(data.num_nodes):
            raise ValueError(
                "learned_residuals node count does not match data.num_nodes: "
                f"{learned_residuals.size(0)} != {int(data.num_nodes)}"
            )
        if int(learned_residuals.size(1)) != len(self.learned_view_names):
            raise ValueError(
                "learned_residuals view count does not match enabled learned views: "
                f"{learned_residuals.size(1)} != {len(self.learned_view_names)}"
            )
        if int(learned_residuals.size(2)) < 1:
            raise ValueError("learned_residuals must contain at least one mask realization")

        degree_bins = self._degree_bins(data.edge_index.long(), int(data.num_nodes), learned_residuals.dtype)
        standardized = self._degree_condition_standardize(learned_residuals, degree_bins)
        learned_raw_scores = learned_residuals.median(dim=2).values
        learned_view_scores = standardized.median(dim=2).values
        learned_reliability = self._view_reliability(learned_residuals)
        learned_scales = self._view_scales(learned_residuals)

        static_scores, static_names, static_groups, static_strata = self._static_view_bank(data, degree_bins)
        learned_groups = [
            str(_section_get(self.cfg, "learned_view_groups", name, "learned"))
            for name in self.learned_view_names
        ]
        learned_strata = [
            self._strata_for_group(group, degree_bins)
            for group in learned_groups
        ]

        raw_parts = []
        score_parts = []
        view_names = []
        view_groups = []
        strata_parts = []
        if static_scores.numel() > 0:
            raw_parts.append(static_scores)
            score_parts.append(static_scores)
            view_names.extend(static_names)
            view_groups.extend(static_groups)
            strata_parts.extend(static_strata)
        raw_parts.append(learned_raw_scores)
        score_parts.append(learned_view_scores)
        view_names.extend(self.learned_view_names)
        view_groups.extend(learned_groups)
        strata_parts.extend(learned_strata)

        raw_view_scores = torch.cat(raw_parts, dim=1)
        calibration_scores = torch.cat(score_parts, dim=1)
        tail_probabilities = torch.stack(
            [
                _empirical_upper_tail_probability(calibration_scores[:, idx], strata_parts[idx], self.eps)
                for idx in range(calibration_scores.size(1))
            ],
            dim=1,
        )
        view_scores = -torch.log(tail_probabilities.clamp_min(self.eps))
        group_names, group_probabilities, group_scores, group_weights = self._aggregate_groups(
            tail_probabilities,
            view_groups,
        )
        (
            stage2_eligible,
            learned_signal,
            learned_static_spearman,
            active_learned_views,
        ) = self._stage2_eligibility_metrics(group_names, group_scores, learned_scales)
        group_weights = self._apply_group_weight_policy(group_names, group_weights, stage2_eligible)
        combined_probability = _cauchy_combine_probabilities(
            group_probabilities,
            group_weights,
            self.eps,
            bool(_section_get(self.cfg, "aggregation", "ignore_large_p", True)),
        )
        score = rank_norm(-torch.log(combined_probability.clamp_min(self.eps)))
        confidence, view_disagreement, mask_instability = self._confidence(view_scores.unsqueeze(-1), view_scores)
        view_weights = self._balanced_view_weights(view_groups).to(raw_view_scores)
        view_scales = self._all_view_scales(raw_view_scores, len(static_names), learned_scales)
        train_view_weights = self._train_view_weights(learned_reliability, learned_groups, learned_scales)

        diagnostics = {
            "view_disagreement_mean": view_disagreement.mean(),
            "mask_instability_mean": mask_instability.mean(),
            "confidence_mean": confidence.mean(),
            "score_mean": score.mean(),
            "combined_probability_min": combined_probability.min(),
            "stage2_eligible": stage2_eligible,
            "stage2_learned_signal": learned_signal,
            "stage2_learned_static_spearman": learned_static_spearman,
            "stage2_active_learned_views": active_learned_views,
        }
        for idx, name in enumerate(view_names):
            diagnostics[f"view_reliability/{name}"] = view_weights[idx]
        for idx, name in enumerate(group_names):
            diagnostics[f"group_weight/{name}"] = group_weights[idx]
            diagnostics[f"group_signal_q95/{name}"] = torch.quantile(group_scores[:, idx].float(), 0.95)

        return CrossFittedResult(
            residuals=learned_residuals,
            standardized_residuals=standardized,
            raw_view_scores=raw_view_scores,
            tail_probabilities=tail_probabilities,
            view_scores=view_scores,
            score=score,
            combined_probability=combined_probability,
            confidence=confidence,
            view_weights=view_weights,
            view_scales=view_scales,
            view_names=view_names,
            view_groups=view_groups,
            group_names=group_names,
            group_probabilities=group_probabilities,
            group_scores=group_scores,
            group_weights=group_weights,
            train_view_weights=train_view_weights,
            train_view_scales=learned_scales,
            view_disagreement=view_disagreement,
            mask_instability=mask_instability,
            degree_bins=degree_bins,
            diagnostics=diagnostics,
        )

    def aggregate_with_weights(
        self,
        result: CrossFittedResult,
        view_weights: torch.Tensor,
    ) -> torch.Tensor:
        return result.score

    def _static_view_bank(
        self,
        data: Any,
        degree_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str], List[str], List[torch.Tensor]]:
        x = data.x.float()
        edge_index = data.edge_index.long()
        num_nodes = int(data.num_nodes)
        device = x.device
        raw_views = []
        names = []
        groups = []
        strata = []

        def add(name: str, group: str, score: torch.Tensor) -> None:
            raw_views.append(score.reshape(-1).float())
            names.append(name)
            groups.append(group)
            strata.append(self._strata_for_group(group, degree_bins))

        topology_cfg = _cfg_get(_cfg_get(self.cfg, "mechanism_views", None), "topology", None)
        local_cfg = _cfg_get(_cfg_get(self.cfg, "mechanism_views", None), "local_consistency", None)
        global_cfg = _cfg_get(_cfg_get(self.cfg, "mechanism_views", None), "global_attribute", None)

        degree = _degree(edge_index, num_nodes, dtype=x.dtype)
        log_degree = torch.log1p(degree)
        if bool(_cfg_get(topology_cfg, "degree_low", True)):
            add("degree_low", "topology", -log_degree)
        if bool(_cfg_get(topology_cfg, "degree_high", True)):
            add("degree_high", "topology", log_degree)

        if bool(_cfg_get(topology_cfg, "egonet_density", False)):
            add("degree_normalized_density", "topology", self._degree_normalized_density(edge_index, degree))

        neighbor_stats = _neighbor_feature_statistics(x, edge_index, num_nodes)
        if bool(_cfg_get(local_cfg, "neighbor_l2", True)):
            add("feature_neighbor_l2", "local_consistency", neighbor_stats["neighbor_l2"])
        if bool(_cfg_get(local_cfg, "neighbor_cosine", True)):
            add("feature_neighbor_cosine", "local_consistency", neighbor_stats["neighbor_cosine"])

        if bool(_cfg_get(global_cfg, "feature_mean_distance", True)):
            center = x.median(dim=0, keepdim=True).values
            add("feature_median_distance", "global_attribute", (x - center).pow(2).sum(dim=1).sqrt())
        if bool(_cfg_get(global_cfg, "feature_knn_distance", True)):
            knn = dense_knn_distance(
                x,
                k=int(_cfg_get(global_cfg, "knn_k", _section_get(self.cfg, "crossfit", "density_knn_k", 10))),
                max_nodes=int(
                    _cfg_get(
                        global_cfg,
                        "max_dense_knn_nodes",
                        _section_get(self.cfg, "crossfit", "max_dense_knn_nodes", 12_000),
                    )
                ),
            )
            if knn.numel() == num_nodes:
                add("feature_knn_distance", "global_attribute", knn)

        if not raw_views:
            return x.new_empty((num_nodes, 0)), names, groups, strata
        return torch.stack(raw_views, dim=1).to(device), names, groups, strata

    def _degree_normalized_density(
        self,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return degree.new_zeros(degree.numel())
        src, dst = edge_index
        neighbor_degree = degree.new_zeros(degree.numel())
        count = degree.new_zeros(degree.numel())
        ones = degree.new_ones(src.numel())
        neighbor_degree.index_add_(0, src, degree[dst])
        neighbor_degree.index_add_(0, dst, degree[src])
        count.index_add_(0, src, ones)
        count.index_add_(0, dst, ones)
        expected = neighbor_degree / count.clamp_min(1.0)
        return (degree - expected).abs()

    def _strata_for_group(self, group: str, degree_bins: torch.Tensor) -> torch.Tensor:
        calibration_cfg = _cfg_get(self.cfg, "calibration", None)
        default = str(_cfg_get(calibration_cfg, "default_strata", "degree_bins"))
        group_key = f"{group}_strata"
        mode = str(_cfg_get(calibration_cfg, group_key, "global" if group == "topology" else default))
        if mode == "degree_bins":
            return degree_bins
        return torch.zeros_like(degree_bins)

    def _aggregate_groups(
        self,
        tail_probabilities: torch.Tensor,
        view_groups: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        group_names = list(dict.fromkeys(view_groups))
        group_probabilities = []
        for group in group_names:
            indices = [idx for idx, item in enumerate(view_groups) if item == group]
            group_p = tail_probabilities[:, indices].min(dim=1).values
            if str(_section_get(self.cfg, "aggregation", "within_group", "bonferroni_min")) == "bonferroni_min":
                group_p = (group_p * float(len(indices))).clamp_max(1.0)
            group_probabilities.append(group_p)
        group_prob_tensor = torch.stack(group_probabilities, dim=1)
        group_scores = -torch.log(group_prob_tensor.clamp_min(self.eps))
        group_weights = self._group_weights(group_names, group_prob_tensor)
        return group_names, group_prob_tensor, group_scores, group_weights

    def _group_weights(self, group_names: List[str], group_probabilities: torch.Tensor) -> torch.Tensor:
        count = max(1, len(group_names))
        weights = group_probabilities.new_full((count,), 1.0 / float(count))
        aggregation_cfg = _cfg_get(self.cfg, "aggregation", None)
        min_weight = float(_cfg_get(aggregation_cfg, "min_group_weight", 0.0))
        max_weight = float(_cfg_get(aggregation_cfg, "max_group_weight", 1.0))
        if count * min_weight <= 1.0 and max_weight >= min_weight:
            weights = weights.clamp(min=min_weight, max=max_weight)
            weights = weights / weights.sum().clamp_min(self.eps)
        return weights

    def _apply_group_weight_policy(
        self,
        group_names: List[str],
        group_weights: torch.Tensor,
        stage2_eligible: torch.Tensor,
    ) -> torch.Tensor:
        aggregation_cfg = _cfg_get(self.cfg, "aggregation", None)
        policy = str(_cfg_get(aggregation_cfg, "learned_group_policy", "eligible_only"))
        if policy != "eligible_only" or "learned" not in group_names:
            return group_weights
        if bool((stage2_eligible > 0.5).detach().cpu().item()):
            return group_weights

        weights = group_weights.clone()
        weights[group_names.index("learned")] = 0.0
        if float(weights.sum().detach().cpu()) <= self.eps:
            return group_weights
        return weights / weights.sum().clamp_min(self.eps)

    def _balanced_view_weights(self, view_groups: List[str]) -> torch.Tensor:
        group_names = list(dict.fromkeys(view_groups))
        weights = []
        for group in view_groups:
            group_size = max(1, sum(1 for item in view_groups if item == group))
            weights.append(1.0 / float(len(group_names)) / float(group_size))
        return torch.tensor(weights, dtype=torch.float32)

    def _all_view_scales(
        self,
        raw_view_scores: torch.Tensor,
        static_count: int,
        learned_scales: torch.Tensor,
    ) -> torch.Tensor:
        scales = []
        for idx in range(static_count):
            scales.append(_robust_scale(raw_view_scores[:, idx], self.eps))
        scales.extend([scale for scale in learned_scales])
        return torch.stack(scales).to(raw_view_scores)

    def _train_view_weights(
        self,
        learned_reliability: torch.Tensor,
        learned_groups: List[str],
        learned_scales: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(_section_get(self.cfg, "refinement", "learned_task_weighting", "uniform"))
        min_scale = float(_section_get(self.cfg, "refinement", "min_train_view_scale", 1e-4))
        active = learned_scales.to(learned_reliability).reshape(-1) >= min_scale
        if mode == "reliability" and float(learned_reliability.sum().detach().cpu()) > self.eps:
            weights = learned_reliability
        else:
            weights = learned_reliability.new_ones(len(learned_groups))
        if active.numel() == weights.numel() and active.any():
            weights = weights * active.to(weights)
        return weights / weights.sum().clamp_min(self.eps)

    def _stage2_eligibility_metrics(
        self,
        group_names: List[str],
        group_scores: torch.Tensor,
        learned_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selection_cfg = _cfg_get(self.cfg, "selection", None)
        if not bool(_cfg_get(selection_cfg, "require_stage2_eligibility", True)):
            one = group_scores.new_tensor(1.0)
            return one, one, one, group_scores.new_tensor(float(learned_scales.numel()))
        if "learned" not in group_names:
            zero = group_scores.new_tensor(0.0)
            return zero, zero, zero, zero

        learned_idx = group_names.index("learned")
        learned_score = group_scores[:, learned_idx]
        signal = torch.quantile(learned_score.float(), 0.95) - learned_score.median()
        min_scale = float(_section_get(self.cfg, "refinement", "min_train_view_scale", 1e-4))
        active_learned_views = (
            (learned_scales.to(group_scores).reshape(-1) >= min_scale)
            .sum()
            .to(group_scores)
        )

        static_indices = [idx for idx, name in enumerate(group_names) if name != "learned"]
        if static_indices:
            static_reference = group_scores[:, static_indices].max(dim=1).values
            learned_static_spearman = spearman_corr(learned_score, static_reference).to(group_scores)
        else:
            learned_static_spearman = group_scores.new_tensor(0.0)

        min_signal = float(_cfg_get(selection_cfg, "min_learned_group_signal", 1.0))
        min_spearman = float(_cfg_get(selection_cfg, "min_learned_static_spearman", 0.05))
        min_active = int(_cfg_get(selection_cfg, "min_active_learned_views", 1))
        eligible = (
            (signal >= min_signal)
            & (learned_static_spearman >= min_spearman)
            & (active_learned_views >= float(min_active))
        )
        return (
            eligible.to(group_scores.dtype),
            signal.to(group_scores),
            learned_static_spearman,
            active_learned_views,
        )

    def _degree_bins(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bins = int(_section_get(self.cfg, "crossfit", "degree_bins", 5))
        bins = max(1, bins)
        if bins == 1:
            return torch.zeros(num_nodes, device=edge_index.device, dtype=torch.long)
        deg = _degree(edge_index, num_nodes, dtype=dtype)
        percentile = rank_norm(deg)
        return torch.clamp((percentile * bins).long(), min=0, max=bins - 1)

    def _degree_condition_standardize(
        self,
        residuals: torch.Tensor,
        degree_bins: torch.Tensor,
    ) -> torch.Tensor:
        standardized = torch.empty_like(residuals)
        num_bins = int(degree_bins.max().item()) + 1 if degree_bins.numel() > 0 else 1
        for view_idx in range(residuals.size(1)):
            for mask_idx in range(residuals.size(2)):
                values = residuals[:, view_idx, mask_idx]
                global_center = values.median()
                global_scale = _robust_scale(values, self.eps)
                for bin_idx in range(num_bins):
                    bin_mask = degree_bins == bin_idx
                    if int(bin_mask.sum().item()) >= 3:
                        bin_values = values[bin_mask]
                        center = bin_values.median()
                        scale = _robust_scale(bin_values, self.eps)
                    else:
                        center = global_center
                        scale = global_scale
                    standardized[bin_mask, view_idx, mask_idx] = (values[bin_mask] - center) / scale
        return standardized

    def _view_reliability(self, residuals: torch.Tensor) -> torch.Tensor:
        mask_count = residuals.size(2)
        within = residuals.var(dim=2, unbiased=False).mean(dim=0)
        mean_by_node = residuals.mean(dim=2)
        between = (mean_by_node.var(dim=0, unbiased=False) - within / float(mask_count)).clamp_min(0.0)
        reliability = between / (between + within + self.eps)
        if float(reliability.sum().detach().cpu()) <= self.eps:
            return torch.full_like(reliability, 1.0 / max(1, reliability.numel()))
        return reliability / reliability.sum().clamp_min(self.eps)

    def _view_scales(self, residuals: torch.Tensor) -> torch.Tensor:
        scales = []
        for view_idx in range(residuals.size(1)):
            values = residuals[:, view_idx, :].reshape(-1)
            center = values.median().abs()
            scale = center + _robust_scale(values, self.eps)
            scales.append(scale.clamp_min(self.eps))
        return torch.stack(scales)

    def _confidence(
        self,
        standardized: torch.Tensor,
        view_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if view_scores.size(1) <= 1:
            view_disagreement = view_scores.new_zeros(view_scores.size(0))
        else:
            center = view_scores.median(dim=1, keepdim=True).values
            view_disagreement = (view_scores - center).abs().median(dim=1).values
        if standardized.size(2) <= 1:
            mask_instability = view_scores.new_zeros(view_scores.size(0))
        else:
            mask_instability = standardized.var(dim=2, unbiased=False).mean(dim=1)

        lambda_u = float(_section_get(self.cfg, "rank_safety", "mask_instability_weight", 1.0))
        uncertainty = view_disagreement + lambda_u * mask_instability
        if float(uncertainty.max().detach().cpu()) <= self.eps:
            confidence = torch.ones_like(uncertainty)
        else:
            scale = uncertainty.median()
            if float(scale.detach().cpu()) <= self.eps:
                scale = uncertainty.mean().clamp_min(self.eps)
            confidence = torch.exp(-uncertainty / scale.clamp_min(self.eps)).clamp(0.0, 1.0)
        return confidence, view_disagreement, mask_instability


class RobustNormalityRefiner:
    """Bounded-influence MM refinement for the adapter and decoders."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.view_names = _enabled_view_names(cfg)
        self.eps = float(_section_get(cfg, "crossfit", "eps", 1e-8))
        self.decoder_reference: Optional[Dict[str, torch.Tensor]] = None

    def snapshot_reference(self, backbone: NormalityBackbone) -> None:
        self.decoder_reference = {
            key: value.detach().clone()
            for key, value in backbone.attr_decoder.state_dict().items()
        }

    def trainable_state(self, backbone: NormalityBackbone) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "adapter": copy.deepcopy(backbone.adapter.state_dict()),
            "attr_decoder": copy.deepcopy(backbone.attr_decoder.state_dict()),
        }

    def load_trainable_state(
        self,
        backbone: NormalityBackbone,
        state: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        backbone.adapter.load_state_dict(state["adapter"])
        backbone.attr_decoder.load_state_dict(state["attr_decoder"])

    def node_loss_from_result(
        self,
        result: CrossFittedResult,
        view_weights: Optional[torch.Tensor] = None,
        view_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if result.combined_probability.numel() == result.score.numel():
            return -torch.log(result.combined_probability.to(result.score).clamp_min(self.eps))
        weights = (view_weights if view_weights is not None else result.view_weights).to(result.raw_view_scores)
        scales = (view_scales if view_scales is not None else result.view_scales).to(result.raw_view_scores)
        scaled = result.raw_view_scores / scales.clamp_min(self.eps)
        return (scaled * weights.reshape(1, -1)).sum(dim=1)

    def node_weights_from_result(
        self,
        result: CrossFittedResult,
        view_weights: Optional[torch.Tensor] = None,
        view_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_loss = self.node_loss_from_result(result, view_weights, view_scales)
        tau = self.robust_tau(node_loss)
        min_weight = float(
            _section_get(
                self.cfg,
                "refinement",
                "normal_weight_floor",
                _section_get(self.cfg, "refinement", "min_normal_weight", 1e-4),
            )
        )
        if result.combined_probability.numel() == result.score.numel():
            alpha = float(_section_get(self.cfg, "refinement", "normal_weight_alpha", 0.1))
            weights = (result.combined_probability.to(node_loss) / max(alpha, self.eps)).clamp(
                min=min_weight,
                max=1.0,
            )
        else:
            weights = torch.exp(-node_loss / tau.clamp_min(self.eps)).clamp(min=min_weight, max=1.0)
        return weights.detach(), node_loss.detach(), tau.detach()

    def robust_tau(self, node_loss: torch.Tensor) -> torch.Tensor:
        method = str(_section_get(self.cfg, "refinement", "robust_scale_method", "median_mad"))
        factor = float(_section_get(self.cfg, "refinement", "robust_scale_factor", 2.5))
        if method == "median_mad":
            center = node_loss.median()
            scale = _robust_scale(node_loss, self.eps)
            return (center + factor * scale).clamp_min(self.eps)
        return node_loss.mean().clamp_min(self.eps)

    def training_objective(
        self,
        backbone: NormalityBackbone,
        data: Any,
        node_weights: torch.Tensor,
        view_weights: torch.Tensor,
        view_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_mask_rate = float(_section_get(self.cfg, "crossfit", "feature_mask_rate", 0.3))
        edge_holdout_rate = float(_section_get(self.cfg, "crossfit", "edge_holdout_rate", 0.1))
        neg_ratio = float(_section_get(self.cfg, "crossfit", "neg_ratio", 1.0))
        beta = float(_section_get(self.cfg, "refinement", "message_reweight_beta", 0.5))
        task = backbone.sample_task_batch(
            data,
            feature_mask_rate=feature_mask_rate,
            edge_holdout_rate=edge_holdout_rate,
            neg_ratio=neg_ratio,
            eval_mode=False,
        )
        losses = backbone.nodewise_task_losses(
            data,
            task,
            use_adapter=True,
            message_node_weights=node_weights,
            message_reweight_beta=beta,
        )

        node_weights = node_weights.to(losses["attribute"]).detach()
        view_weights = view_weights.to(losses["attribute"])
        view_scales = view_scales.to(losses["attribute"])
        denom = node_weights.sum().clamp_min(self.eps)
        total = losses["attribute"].sum() * 0.0
        diagnostics = {}

        for view_idx, name in enumerate(self.view_names):
            scale = view_scales[view_idx].clamp_min(self.eps)
            if name == "structure" and losses["edge_loss"].numel() > 0:
                edge_index = losses["edge_loss_index"]
                src, dst = edge_index
                edge_weights = torch.sqrt(node_weights[src] * node_weights[dst]).detach()
                view_loss = (
                    (edge_weights * losses["edge_loss"] / scale).sum()
                    / edge_weights.sum().clamp_min(self.eps)
                )
            else:
                view_loss = (node_weights * losses[name] / scale).sum() / denom
            total = total + view_weights[view_idx] * view_loss
            diagnostics[f"{name}_loss"] = view_loss.detach()

        proximal = self.proximal_loss(backbone)
        proximal_weight = float(_section_get(self.cfg, "refinement", "proximal_weight", 1.0))
        total = total + proximal_weight * proximal
        diagnostics.update(
            {
                "proximal_loss": proximal.detach(),
                "parameter_drift": self.parameter_drift(backbone).detach(),
                "effective_normal_weight": node_weights.mean().detach(),
                "weight_min": node_weights.min().detach(),
                "weight_p10": torch.quantile(node_weights.float(), 0.10).detach(),
            }
        )
        return total, diagnostics

    def robust_objective(
        self,
        result: CrossFittedResult,
        view_weights: torch.Tensor,
        view_scales: torch.Tensor,
        backbone: Optional[NormalityBackbone] = None,
    ) -> torch.Tensor:
        node_loss = self.node_loss_from_result(result, view_weights, view_scales)
        tau = self.robust_tau(node_loss)
        rho = tau * (1.0 - torch.exp(-node_loss / tau.clamp_min(self.eps)))
        objective = rho.mean()
        if backbone is not None:
            objective = objective + float(
                _section_get(self.cfg, "refinement", "proximal_weight", 1.0)
            ) * self.proximal_loss(backbone)
        return objective.detach()

    def proximal_loss(self, backbone: NormalityBackbone) -> torch.Tensor:
        loss = next(backbone.parameters()).new_tensor(0.0)
        for param in backbone.adapter.parameters():
            loss = loss + param.pow(2).sum()
        if self.decoder_reference is not None:
            ref = {
                key: value.to(next(backbone.attr_decoder.parameters()).device)
                for key, value in self.decoder_reference.items()
            }
            for key, param in backbone.attr_decoder.named_parameters():
                loss = loss + (param - ref[key]).pow(2).sum()
        return loss

    def parameter_drift(self, backbone: NormalityBackbone) -> torch.Tensor:
        return self.proximal_loss(backbone).clamp_min(0.0).sqrt()


class RankSafeProjector:
    def __init__(self, epsilon_max: float = 0.05, enabled: bool = True, max_pairs: int = 200_000) -> None:
        self.epsilon_max = float(epsilon_max)
        self.enabled = bool(enabled)
        self.max_pairs = int(max_pairs)

    def project(
        self,
        base_score: torch.Tensor,
        candidate_score: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_rank = rank_norm(base_score.detach()).reshape(-1)
        candidate_rank = rank_norm(candidate_score.detach()).reshape(-1)
        confidence = confidence.detach().reshape(-1).to(base_rank).clamp(0.0, 1.0)

        if not self.enabled:
            safe_score = candidate_rank
            epsilon = torch.ones_like(base_rank)
        else:
            epsilon = self.epsilon_max * (1.0 - confidence)
            safe_score = base_rank + (candidate_rank - base_rank).clamp(-epsilon, epsilon)
            safe_score = safe_score.clamp(0.0, 1.0)

        displacement = (safe_score - base_rank).abs()
        violations = self.protected_pair_violations(base_rank, safe_score, epsilon)
        flip_ratio = self.rank_flip_ratio(base_rank, safe_score)
        return {
            "score": safe_score,
            "rank_displacement": displacement,
            "max_rank_displacement": displacement.max() if displacement.numel() else displacement.new_tensor(0.0),
            "protected_pair_violations": violations,
            "rank_flip_ratio": flip_ratio,
            "spearman_base": spearman_corr(base_rank, safe_score),
        }

    def _pair_indices(self, num_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = num_nodes * (num_nodes - 1) // 2
        if total <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, torch.tensor(0, device=device)
        if total <= self.max_pairs:
            idx_i, idx_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
            return idx_i, idx_j, torch.tensor(total, device=device)
        idx_i = torch.randint(num_nodes, (self.max_pairs,), device=device)
        idx_j = torch.randint(num_nodes, (self.max_pairs,), device=device)
        keep = idx_i != idx_j
        return idx_i[keep], idx_j[keep], torch.tensor(total, device=device)

    def protected_pair_violations(
        self,
        base_rank: torch.Tensor,
        safe_score: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        idx_i, idx_j, _ = self._pair_indices(base_rank.numel(), base_rank.device)
        if idx_i.numel() == 0:
            return base_rank.new_tensor(0.0)
        base_diff = base_rank[idx_i] - base_rank[idx_j]
        safe_diff = safe_score[idx_i] - safe_score[idx_j]
        protected = base_diff.abs() > (epsilon[idx_i] + epsilon[idx_j] + 1e-8)
        violations = protected & (base_diff * safe_diff <= 0.0)
        return violations.float().sum()

    def rank_flip_ratio(self, base_rank: torch.Tensor, safe_score: torch.Tensor) -> torch.Tensor:
        idx_i, idx_j, _ = self._pair_indices(base_rank.numel(), base_rank.device)
        if idx_i.numel() == 0:
            return base_rank.new_tensor(0.0)
        base_diff = base_rank[idx_i] - base_rank[idx_j]
        safe_diff = safe_score[idx_i] - safe_score[idx_j]
        valid = base_diff != 0.0
        if not valid.any():
            return base_rank.new_tensor(0.0)
        return ((base_diff[valid] * safe_diff[valid]) < 0.0).float().mean()
