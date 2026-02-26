from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ...gnn import InvariantMPNNLayer, mlp


@dataclass(frozen=True)
class EdgeModelConfig:
    d_h: int = 128
    n_layers: int = 3
    n_rbf: int = 16
    dropout: float = 0.0


class EdgeModel(nn.Module):
    """
    Predict edge existence over a fixed candidate set (u,v) pairs.

    - Node embeddings are produced by message passing over a (usually kNN) neighborhood graph.
    - Edge logits are computed from (h_u, h_v, ||x_u-x_v||).
    """

    def __init__(self, *, cfg: EdgeModelConfig):
        super().__init__()
        self.cfg = cfg
        self.node_in = mlp([2, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
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
    ) -> torch.Tensor:
        h = self.node_in(coords01)
        for layer in self.layers:
            h = layer(h, coords01, msg_edge_index)

        u = cand_pairs_uv[:, 0]
        v = cand_pairs_uv[:, 1]
        dx = coords01[u] - coords01[v]
        r = torch.sqrt((dx**2).sum(dim=-1) + 1e-8)[:, None]  # (M,1)
        edge_in = torch.cat([h[u], h[v], r], dim=-1)
        logits = self.edge_head(edge_in).squeeze(-1)  # (M,)
        return logits


def label_candidate_pairs(cand_pairs_uv: np.ndarray, true_edges_uv: np.ndarray) -> np.ndarray:
    """
    cand_pairs_uv: (M,2) u<v
    true_edges_uv: (E,2) u<v
    returns: y in {0,1} of shape (M,)
    """
    if cand_pairs_uv.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if true_edges_uv.size == 0:
        return np.zeros((cand_pairs_uv.shape[0],), dtype=np.float32)

    max_node = int(max(int(cand_pairs_uv.max()), int(true_edges_uv.max())) + 1)
    cand_code = cand_pairs_uv[:, 0].astype(np.int64, copy=False) * max_node + cand_pairs_uv[:, 1].astype(np.int64, copy=False)
    true_code = true_edges_uv[:, 0].astype(np.int64, copy=False) * max_node + true_edges_uv[:, 1].astype(np.int64, copy=False)
    y = np.isin(cand_code, true_code).astype(np.float32)
    return y
