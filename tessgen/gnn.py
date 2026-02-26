from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: Sequence[int], *, activation: nn.Module | None = None, dropout: float = 0.0) -> nn.Sequential:
    if activation is None:
        activation = nn.SiLU()
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        in_d = int(sizes[i])
        out_d = int(sizes[i + 1])
        layers.append(nn.Linear(in_d, out_d))
        if i < len(sizes) - 2:
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, *, r_min: float = 0.0, r_max: float = 2.0):
        super().__init__()
        self.n_rbf = int(n_rbf)
        centers = torch.linspace(r_min, r_max, steps=self.n_rbf)
        width = (centers[1] - centers[0]) if self.n_rbf > 1 else torch.tensor(1.0)
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", 1.0 / (width**2 + 1e-8))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (E,)
        diff = r[:, None] - self.centers[None, :]
        return torch.exp(-self.gamma * diff**2)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src: (E, D) or (E,)
    index: (E,) int64
    """
    if src.ndim == 1:
        out = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


@dataclass
class GraphBatch:
    h: torch.Tensor  # (N, Dh)
    x: torch.Tensor  # (N, 2)
    edge_index: torch.Tensor  # (2, E) src,dst
    batch: torch.Tensor  # (N,) graph id


class InvariantMPNNLayer(nn.Module):
    """
    A simple E(2)-invariant message passing layer:
    messages depend on node features and pairwise distances only.
    """

    def __init__(self, d_h: int, *, n_rbf: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf, r_min=0.0, r_max=math.sqrt(2.0))
        self.phi_m = mlp([2 * d_h + n_rbf, d_h, d_h], dropout=dropout)
        self.phi_h = mlp([2 * d_h, d_h, d_h], dropout=dropout)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        dx = x[src] - x[dst]
        r = torch.sqrt((dx**2).sum(dim=-1) + 1e-8)
        rbf = self.rbf(r)
        m_in = torch.cat([h[src], h[dst], rbf], dim=-1)
        m = self.phi_m(m_in)
        agg = scatter_sum(m, dst, dim_size=h.shape[0])
        h_up = self.phi_h(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_up)
        return h


def global_mean_max_pool(h: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """
    Returns (B, 2*Dh): concat(mean, max) pooling.
    """
    dh = h.shape[1]
    sum_h = scatter_sum(h, batch, dim_size=n_graphs)
    count = scatter_sum(torch.ones((h.shape[0],), device=h.device, dtype=h.dtype), batch, dim_size=n_graphs).clamp_min(1.0)
    mean_h = sum_h / count[:, None]

    max_h = torch.full((n_graphs, dh), -1e9, device=h.device, dtype=h.dtype)
    for i in range(n_graphs):
        mask = batch == i
        if mask.any():
            max_h[i] = h[mask].max(dim=0).values
        else:
            max_h[i] = 0.0

    return torch.cat([mean_h, max_h], dim=-1)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or scalar, assumed in [0, T)
        returns: (B, dim)
        """
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / float(half)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=-1)
        return emb
