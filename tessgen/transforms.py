from __future__ import annotations

import torch


def apply_log_cols_torch(x: torch.Tensor, cols: list[str], log_cols: set[str], *, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (B, D) aligned with cols
    """
    x = x.clone()
    for i, c in enumerate(cols):
        if c in log_cols:
            x[:, i] = torch.log(x[:, i].clamp_min(eps))
    return x


def invert_log_cols_torch(x: torch.Tensor, cols: list[str], log_cols: set[str]) -> torch.Tensor:
    x = x.clone()
    for i, c in enumerate(cols):
        if c in log_cols:
            x[:, i] = torch.exp(x[:, i])
    return x

