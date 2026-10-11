import torch
from torch import nn


class AttrDecoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DotEdgeDecoder(nn.Module):
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return z.new_empty(0)
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)
