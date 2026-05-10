from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...gnn import InvariantMPNNLayer, global_mean_max_pool, mlp
from ...utils import Batch


@dataclass(frozen=True)
class SurrogateConfig:
    d_h: int = 128
    n_layers: int = 4
    n_rbf: int = 16
    dropout: float = 0.0
    use_rd: bool = True


class SurrogateModel(nn.Module):
    def __init__(self, *, y_dim: int, cfg: SurrogateConfig):
        super().__init__()
        self.cfg = cfg
        self.y_dim = int(y_dim)
        self.use_rd = bool(cfg.use_rd)

        self.node_in = mlp([2, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [InvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.rd_embed = mlp([1, cfg.d_h, cfg.d_h], dropout=cfg.dropout) if self.use_rd else None

        # Extra graph scalars: log(1+N), log(1+E)
        self.graph_scalar = mlp([2, cfg.d_h, cfg.d_h], dropout=cfg.dropout)

        head_in_dim = cfg.d_h * 2 + cfg.d_h
        if self.use_rd:
            head_in_dim += cfg.d_h
        self.head = mlp([head_in_dim, cfg.d_h, self.y_dim], dropout=cfg.dropout)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = batch.x  # (N,2)
        edge_index = batch.edge_index  # (2,E)
        b = batch.batch  # (N,)
        n_graphs = int(batch.rd.shape[0])

        h = self.node_in(x)
        for layer in self.layers:
            h = layer(h, x, edge_index)

        pooled = global_mean_max_pool(h, b, n_graphs=n_graphs)
        scalars = torch.stack(
            [
                (1.0 + batch.n_nodes.float()).log(),
                (1.0 + batch.n_edges.float()).log(),
            ],
            dim=-1,
        )
        s_h = self.graph_scalar(scalars)
        head_inputs = [pooled, s_h]
        if self.use_rd:
            if self.rd_embed is None:
                raise RuntimeError("SurrogateModel is configured to use RD but rd_embed is missing.")
            head_inputs.insert(1, self.rd_embed(batch.rd))
        out = self.head(torch.cat(head_inputs, dim=-1))
        return out
