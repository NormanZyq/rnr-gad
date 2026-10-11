from typing import Optional, Tuple

import torch


def rank_norm(values: torch.Tensor) -> torch.Tensor:
    """Map a 1D score tensor to [0, 1] by rank order."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return flat
    if flat.numel() == 1:
        return torch.zeros_like(flat, dtype=torch.float32)

    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, flat.numel(), device=flat.device)
    return ranks.reshape_as(values).float()


def minmax_norm(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = values.reshape(-1)
    v_min = flat.min()
    v_max = flat.max()
    return ((values - v_min) / (v_max - v_min).clamp_min(eps)).float()


def spearman_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() < 2 or y.numel() < 2:
        return x.new_tensor(0.0, dtype=torch.float32)

    rx = rank_norm(x).reshape(-1)
    ry = rank_norm(y).reshape(-1)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    corr = (rx * ry).sum() / denom.clamp_min(eps)
    return corr.clamp(-1.0, 1.0)


def sample_pseudo_pairs(
    evidence: torch.Tensor,
    confidence: torch.Tensor,
    score_pre: torch.Tensor,
    q_high: float = 0.85,
    q_low: float = 0.15,
    confidence_tau: float = 0.6,
    evidence_margin: float = 0.2,
    max_pairs: int = 20_000,
    prefer_conflict_pairs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build high-evidence vs low-evidence pseudo ranking pairs without labels."""
    evidence = evidence.detach().reshape(-1)
    confidence = confidence.detach().reshape(-1)
    score_pre = score_pre.detach().reshape(-1)
    device = evidence.device

    empty_idx = torch.empty(0, dtype=torch.long, device=device)
    empty_weight = torch.empty(0, dtype=evidence.dtype, device=device)
    if evidence.numel() < 2:
        return empty_idx, empty_idx, empty_weight

    high_cut = torch.quantile(evidence.float(), q_high)
    low_cut = torch.quantile(evidence.float(), q_low)
    high = torch.nonzero((evidence >= high_cut) & (confidence > confidence_tau), as_tuple=False)
    low = torch.nonzero((evidence <= low_cut) & (confidence > confidence_tau), as_tuple=False)
    high = high.reshape(-1)
    low = low.reshape(-1)

    if high.numel() == 0 or low.numel() == 0:
        return empty_idx, empty_idx, empty_weight

    total_pairs = int(high.numel() * low.numel())
    max_pairs = min(max_pairs, total_pairs)
    if max_pairs <= 0:
        return empty_idx, empty_idx, empty_weight

    if total_pairs <= max_pairs:
        idx_i = high.repeat_interleave(low.numel())
        idx_j = low.repeat(high.numel())
    else:
        high_pos = torch.randint(high.numel(), (max_pairs,), device=device)
        low_pos = torch.randint(low.numel(), (max_pairs,), device=device)
        idx_i = high[high_pos]
        idx_j = low[low_pos]

    margin_mask = evidence[idx_i] > evidence[idx_j] + evidence_margin
    idx_i = idx_i[margin_mask]
    idx_j = idx_j[margin_mask]
    if idx_i.numel() == 0:
        return empty_idx, empty_idx, empty_weight

    if prefer_conflict_pairs:
        conflict_mask = score_pre[idx_i] <= score_pre[idx_j]
        if conflict_mask.any():
            idx_i = idx_i[conflict_mask]
            idx_j = idx_j[conflict_mask]

    weights = confidence[idx_i] * confidence[idx_j] * (evidence[idx_i] - evidence[idx_j]).abs()
    return idx_i.long(), idx_j.long(), weights.clamp_min(1e-6)


def sample_multiview_consensus_pairs(
    view_tensor: torch.Tensor,
    score_pre: Optional[torch.Tensor] = None,
    evidence: Optional[torch.Tensor] = None,
    q_high: float = 0.8,
    q_low: float = 0.2,
    min_consensus_views: int = 2,
    consensus_margin: float = 0.15,
    max_pairs: int = 20_000,
    include_score_pre: bool = True,
    include_evidence: bool = True,
    prefer_conflict_pairs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pseudo pairs from agreement across independent anomaly score views.

    Each column in `view_tensor` is expected to be a per-node score where larger
    means more anomalous. Optional `score_pre` and `evidence` are appended as
    additional views after rank/min-max normalization.
    """
    views = view_tensor.detach().float()
    if views.dim() == 1:
        views = views.unsqueeze(-1)

    extra_views = []
    if include_score_pre and score_pre is not None:
        extra_views.append(rank_norm(score_pre.detach()).reshape(-1, 1))
    if include_evidence and evidence is not None:
        extra_views.append(minmax_norm(evidence.detach()).reshape(-1, 1))
    if extra_views:
        views = torch.cat([views, *extra_views], dim=1)

    device = views.device
    empty_idx = torch.empty(0, dtype=torch.long, device=device)
    empty_weight = torch.empty(0, dtype=views.dtype, device=device)
    if views.numel() == 0 or views.size(0) < 2 or views.size(1) < 2:
        return empty_idx, empty_idx, empty_weight

    min_views = max(1, int(min_consensus_views))
    if views.size(1) < min_views:
        return empty_idx, empty_idx, empty_weight

    q_high = max(0.0, min(1.0, float(q_high)))
    q_low = max(0.0, min(q_high, float(q_low)))
    high_votes = (views >= q_high).sum(dim=1)
    low_votes = (views <= q_low).sum(dim=1)
    consensus_score = views.mean(dim=1)

    high = torch.nonzero(high_votes >= min_views, as_tuple=False).reshape(-1)
    low = torch.nonzero(low_votes >= min_views, as_tuple=False).reshape(-1)
    if high.numel() == 0 or low.numel() == 0:
        return empty_idx, empty_idx, empty_weight

    total_pairs = int(high.numel() * low.numel())
    max_pairs = min(int(max_pairs), total_pairs)
    if max_pairs <= 0:
        return empty_idx, empty_idx, empty_weight

    if total_pairs <= max_pairs:
        idx_i = high.repeat_interleave(low.numel())
        idx_j = low.repeat(high.numel())
    else:
        high_pos = torch.randint(high.numel(), (max_pairs,), device=device)
        low_pos = torch.randint(low.numel(), (max_pairs,), device=device)
        idx_i = high[high_pos]
        idx_j = low[low_pos]

    margin_mask = consensus_score[idx_i] > consensus_score[idx_j] + float(consensus_margin)
    idx_i = idx_i[margin_mask]
    idx_j = idx_j[margin_mask]
    if idx_i.numel() == 0:
        return empty_idx, empty_idx, empty_weight

    if prefer_conflict_pairs and score_pre is not None:
        pre = rank_norm(score_pre.detach()).reshape(-1)
        conflict_mask = pre[idx_i] <= pre[idx_j]
        if conflict_mask.any():
            idx_i = idx_i[conflict_mask]
            idx_j = idx_j[conflict_mask]

    high_strength = high_votes[idx_i].float() / float(views.size(1))
    low_strength = low_votes[idx_j].float() / float(views.size(1))
    gap = (consensus_score[idx_i] - consensus_score[idx_j]).abs()
    weights = high_strength * low_strength * gap
    return idx_i.long(), idx_j.long(), weights.clamp_min(1e-6)
