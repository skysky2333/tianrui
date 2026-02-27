from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ...gnn import mlp


@dataclass(frozen=True)
class NPriorConfig:
    d_h: int = 128
    n_layers: int = 3
    dropout: float = 0.0
    sigma_min: float = 0.7


class NPriorModel(nn.Module):
    """
    Learn a broad prior over log(N) conditioned on (RD + metrics).

    Outputs (mu, log_sigma) such that:
      logN ~ Normal(mu, sigma^2)
    """

    def __init__(self, *, x_dim: int, cfg: NPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.x_dim = int(x_dim)
        if self.x_dim <= 0:
            raise ValueError("x_dim must be > 0")
        if int(cfg.n_layers) <= 0:
            raise ValueError("n_layers must be > 0")
        if float(cfg.sigma_min) <= 0.0:
            raise ValueError("sigma_min must be > 0")

        sizes = [self.x_dim] + [int(cfg.d_h)] * int(cfg.n_layers) + [2]
        self.net = mlp(sizes, dropout=float(cfg.dropout))
        self._log_sigma_min = float(math.log(float(cfg.sigma_min)))

    def forward(self, x_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x_z)
        mu = out[..., 0]
        log_sigma = out[..., 1].clamp(min=self._log_sigma_min, max=3.0)
        return mu, log_sigma

