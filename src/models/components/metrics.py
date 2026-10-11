from typing import Dict

import torch
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from src.models.components.rank_utils import spearman_corr


def _nan_like(score: torch.Tensor) -> torch.Tensor:
    return score.new_tensor(float("nan"), dtype=torch.float32)


def _safe_binary_auroc(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.unique().numel() < 2:
        return _nan_like(score)
    return binary_auroc(score.float(), y.long())


def _safe_binary_average_precision(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.unique().numel() < 2:
        return _nan_like(score)
    return binary_average_precision(score.float(), y.long())


def precision_at_anomaly_k(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k = int(y.long().sum().item())
    if k <= 0:
        return _nan_like(score)
    k = min(k, score.numel())
    top_idx = torch.topk(score.float(), k=k).indices
    return y.float()[top_idx].mean()


def first_anomaly_rank(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(-1).long()
    anomaly_idx = torch.nonzero(y > 0, as_tuple=False).reshape(-1)
    if anomaly_idx.numel() == 0:
        return _nan_like(score)
    order = torch.argsort(score.reshape(-1).float(), descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.long, device=score.device)
    return ranks[anomaly_idx].min().to(score.dtype)


def precision_at_one(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(-1).long()
    if y.numel() == 0 or y.long().sum().item() <= 0:
        return _nan_like(score)
    idx = torch.argmax(score.reshape(-1).float())
    return y.float()[idx]


def true_corrected_pair_ratio(
    score_pre: torch.Tensor,
    score_final: torch.Tensor,
    y: torch.Tensor,
    max_pairs: int = 200_000,
) -> torch.Tensor:
    anomaly_idx = torch.nonzero(y > 0, as_tuple=False).reshape(-1)
    normal_idx = torch.nonzero(y == 0, as_tuple=False).reshape(-1)
    if anomaly_idx.numel() == 0 or normal_idx.numel() == 0:
        return _nan_like(score_final)

    total_pairs = int(anomaly_idx.numel() * normal_idx.numel())
    if total_pairs <= max_pairs:
        a = anomaly_idx.repeat_interleave(normal_idx.numel())
        n = normal_idx.repeat(anomaly_idx.numel())
    else:
        a = anomaly_idx[torch.randint(anomaly_idx.numel(), (max_pairs,), device=y.device)]
        n = normal_idx[torch.randint(normal_idx.numel(), (max_pairs,), device=y.device)]

    was_wrong = score_pre[a] < score_pre[n]
    denom = was_wrong.float().sum()
    if denom <= 0:
        return score_final.new_tensor(0.0, dtype=torch.float32)
    corrected = was_wrong & (score_final[a] > score_final[n])
    return corrected.float().sum() / denom


def compute_binary_eval_metrics(
    score_pre: torch.Tensor,
    score_final: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    y = (y.reshape(-1) > 0).long()
    score_pre = score_pre.reshape(-1)
    score_final = score_final.reshape(-1)

    score_pre_auroc = _safe_binary_auroc(score_pre, y)
    score_final_auroc = _safe_binary_auroc(score_final, y)
    score_pre_ap = _safe_binary_average_precision(score_pre, y)
    score_final_ap = _safe_binary_average_precision(score_final, y)

    return {
        "score_pre_auroc": score_pre_auroc,
        "score_pre_auprc": score_pre_ap,
        "score_pre_ap": score_pre_ap,
        "auroc": score_final_auroc,
        "auc_gain": score_final_auroc - score_pre_auroc,
        "auprc": score_final_ap,
        "ap": score_final_ap,
        "precision_at_k": precision_at_anomaly_k(score_final, y),
        "precision_at_1": precision_at_one(score_final, y),
        "first_anomaly_rank": first_anomaly_rank(score_final, y),
        "spearman_pre_final": spearman_corr(score_pre, score_final),
        "rank_plasticity": 1.0 - spearman_corr(score_pre, score_final),
        "true_corrected_pair_ratio": true_corrected_pair_ratio(score_pre, score_final, y),
    }
