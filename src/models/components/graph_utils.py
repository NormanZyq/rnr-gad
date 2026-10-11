from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, negative_sampling


def mask_feature_entries(
    x: torch.Tensor,
    mask_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask_rate <= 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)

    mask = torch.rand_like(x, dtype=torch.float32) < mask_rate
    if not mask.any():
        mask = torch.ones_like(x, dtype=torch.bool)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def augment_for_pretrain(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    feature_mask_rate: float,
    edge_drop_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_aug, feature_mask = mask_feature_entries(x, feature_mask_rate)
    if edge_drop_rate > 0.0 and edge_index.numel() > 0:
        edge_aug, _ = dropout_edge(edge_index, p=edge_drop_rate, training=True)
    else:
        edge_aug = edge_index
    return x_aug, edge_aug, feature_mask


def safe_negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
) -> torch.Tensor:
    if num_nodes < 2 or num_neg_samples <= 0:
        return edge_index.new_empty((2, 0))
    return negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )


def per_node_attr_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (x - x_hat).pow(2).mean(dim=-1)


def edge_reconstruction_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> torch.Tensor:
    losses = []
    if pos_logits.numel() > 0:
        losses.append(F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)))
    if neg_logits.numel() > 0:
        losses.append(F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))
    if not losses:
        return (pos_logits.sum() + neg_logits.sum()) * 0.0
    return torch.stack(losses).sum()


def per_node_struct_error(
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    device = pos_edge_index.device
    dtype = pos_logits.dtype if pos_logits.numel() > 0 else torch.float32
    err = torch.zeros(num_nodes, device=device, dtype=dtype)
    count = torch.zeros(num_nodes, device=device, dtype=dtype)

    def add_edges(edge_index: torch.Tensor, loss: torch.Tensor) -> None:
        if edge_index.numel() == 0:
            return
        src, dst = edge_index
        err.index_add_(0, src, loss)
        err.index_add_(0, dst, loss)
        ones = torch.ones_like(loss)
        count.index_add_(0, src, ones)
        count.index_add_(0, dst, ones)

    if pos_logits.numel() > 0:
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction="none"
        )
        add_edges(pos_edge_index, pos_loss)
    if neg_logits.numel() > 0:
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction="none"
        )
        add_edges(neg_edge_index, neg_loss)

    return err / count.clamp_min(1.0)


def neighbor_squared_distance(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return z.new_zeros(num_nodes)

    src, dst = edge_index
    dist = (z[src] - z[dst]).pow(2).sum(dim=-1)
    out = z.new_zeros(num_nodes)
    count = z.new_zeros(num_nodes)
    out.index_add_(0, src, dist)
    out.index_add_(0, dst, dist)
    ones = torch.ones_like(dist)
    count.index_add_(0, src, ones)
    count.index_add_(0, dst, ones)
    return out / count.clamp_min(1.0)


def dense_knn_distance(
    z: torch.Tensor,
    k: int,
    max_nodes: int,
) -> torch.Tensor:
    num_nodes = z.size(0)
    if num_nodes <= 1 or num_nodes > max_nodes or k <= 0:
        return z.new_empty(0)

    actual_k = min(k, num_nodes - 1)
    dist = torch.cdist(z, z, p=2).pow(2)
    dist.fill_diagonal_(float("inf"))
    return torch.topk(dist, k=actual_k, dim=1, largest=False).values.mean(dim=1)
