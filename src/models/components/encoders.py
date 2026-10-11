from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GCNEncoder requires at least one layer.")

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hid_dim] * (num_layers - 1))
            dims.append(out_dim)

        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if layer_idx < len(self.convs) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def build_encoder(
    encoder_type: str,
    in_dim: int,
    hid_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if encoder_type.lower() != "gcn":
        raise ValueError(f"Unsupported encoder_type '{encoder_type}'. MVP currently supports 'gcn'.")
    return GCNEncoder(in_dim, hid_dim, out_dim, num_layers, dropout)
