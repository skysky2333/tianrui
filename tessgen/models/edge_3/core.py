from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..edge.core import label_candidate_pairs
from ...gnn import InvariantMPNNLayer, mlp


@dataclass(frozen=True)
class Edge3ModelConfig:
    d_h: int = 128
    n_layers: int = 3
    n_rbf: int = 16
    d_search: int = 16
    dropout: float = 0.0


class Edge3Model(nn.Module):
    """
    IDGL-lite edge model.

    - Candidate edges are scored over a fixed (u,v) set.
    - Message passing uses a (learned) neighborhood graph built from kNN in a learned embedding space.
    """

    def __init__(self, *, cfg: Edge3ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.node_in = mlp([2, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.search_proj = mlp([cfg.d_h, cfg.d_h, cfg.d_search], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [InvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.edge_head = mlp([2 * cfg.d_h + 1, cfg.d_h, 1], dropout=cfg.dropout)

    def forward(
        self,
        *,
        coords01: torch.Tensor,  # (N,2)
        msg_edge_index: torch.Tensor,  # (2, Emsg) directed
        cand_pairs_uv: torch.Tensor,  # (M,2) u<v
        h0: torch.Tensor | None = None,  # (N, Dh)
    ) -> torch.Tensor:
        if h0 is None:
            h = self.node_in(coords01)
        else:
            h = h0
        for layer in self.layers:
            h = layer(h, coords01, msg_edge_index)

        u = cand_pairs_uv[:, 0]
        v = cand_pairs_uv[:, 1]
        dx = coords01[u] - coords01[v]
        r = torch.sqrt((dx**2).sum(dim=-1) + 1e-8)[:, None]  # (M,1)
        edge_in = torch.cat([h[u], h[v], r], dim=-1)
        logits = self.edge_head(edge_in).squeeze(-1)  # (M,)
        return logits


__all__ = ["Edge3Model", "Edge3ModelConfig", "label_candidate_pairs"]

