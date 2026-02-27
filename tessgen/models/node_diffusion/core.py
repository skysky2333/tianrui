from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ...gnn import GaussianRBF, InvariantMPNNLayer, SinusoidalEmbedding, mlp, scatter_sum


@dataclass(frozen=True)
class DiffusionConfig:
    n_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionSchedule(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, steps=cfg.n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def n_steps(self) -> int:
        return int(self.betas.shape[0])

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        x0: (N,2)
        t: (1,) int64
        eps: (N,2)
        """
        ab = self.alpha_bars[t].view(1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


@dataclass(frozen=True)
class NodeDenoiserConfig:
    cond_dim: int
    d_h: int = 128
    n_layers: int = 4
    n_rbf: int = 16
    dropout: float = 0.0


class NodeDenoiser(nn.Module):
    """
    Denoiser for 2D point set diffusion.

    Predicts epsilon (noise) with an equivariant edge-weighted vector aggregation:
      eps_i = Σ_j (x_i - x_j) * w_ij
    where w_ij is predicted from invariant node features and distances.
    """

    def __init__(self, cfg: NodeDenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalEmbedding(dim=cfg.d_h)
        self.time_mlp = mlp([cfg.d_h, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.cond_mlp = mlp([cfg.cond_dim, cfg.d_h, cfg.d_h], dropout=cfg.dropout)

        self.node_in = mlp([2 + cfg.d_h, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [InvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.rbf = GaussianRBF(n_rbf=cfg.n_rbf, r_min=0.0, r_max=float(np.sqrt(2.0)))
        self.w_head = mlp([2 * cfg.d_h + cfg.n_rbf, cfg.d_h, 1], dropout=cfg.dropout)

    def forward(
        self,
        *,
        x_t: torch.Tensor,  # (N,2)
        t: torch.Tensor,  # (1,) int64
        cond: torch.Tensor,  # (cond_dim,)
        edge_index: torch.Tensor,  # (2, E) directed, src->dst
    ) -> torch.Tensor:
        N = int(x_t.shape[0])
        t_emb = self.time_mlp(self.time_emb(t.view(1)).squeeze(0))  # (d_h,)
        c_emb = self.cond_mlp(cond.view(1, -1)).squeeze(0)  # (d_h,)
        g = t_emb + c_emb  # (d_h,)
        h = self.node_in(torch.cat([x_t, g.view(1, -1).expand(N, -1)], dim=-1))
        for layer in self.layers:
            h = layer(h, x_t, edge_index)

        src = edge_index[0]
        dst = edge_index[1]
        dx = x_t[src] - x_t[dst]
        r = torch.sqrt((dx**2).sum(dim=-1) + 1e-8)
        rbf = self.rbf(r)
        w_in = torch.cat([h[src], h[dst], rbf], dim=-1)
        w = self.w_head(w_in)  # (E,1)

        # vector message: (x_dst - x_src) * w
        vec = (x_t[dst] - x_t[src]) * w
        eps = scatter_sum(vec, dst, dim_size=N)  # (N,2)
        return eps
