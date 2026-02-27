from __future__ import annotations

from collections.abc import Sequence

import torch

from .ckpt import NPriorBundle
from .transforms import apply_log_cols_torch


def clamp_and_unique(ns: Sequence[int], *, min_n: int, max_n: int) -> list[int]:
    out = []
    seen = set()
    for n in ns:
        n_i = int(n)
        n_i = int(min(max(n_i, int(min_n)), int(max_n)))
        if n_i in seen:
            continue
        seen.add(n_i)
        out.append(n_i)
    return out


@torch.no_grad()
def sample_n_candidates_from_prior(
    prior: NPriorBundle,
    *,
    rd: float,
    cond_vals_raw: torch.Tensor,  # (1, Dc) raw values in prior.cond_cols order
    n_samples: int,
    min_n: int,
    max_n: int,
    device: torch.device,
) -> list[int]:
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if cond_vals_raw.ndim != 2 or cond_vals_raw.shape[0] != 1 or cond_vals_raw.shape[1] != len(prior.cond_cols):
        raise ValueError("cond_vals_raw must be shape (1, len(prior.cond_cols))")

    rd_t = torch.tensor([[float(rd)]], device=device, dtype=torch.float32)
    cond_raw = cond_vals_raw.to(device=device, dtype=torch.float32)
    cond_t = apply_log_cols_torch(cond_raw, prior.cond_cols, prior.log_cols)
    x = torch.cat([rd_t, cond_t], dim=-1)
    x_z = prior.scaler.transform_torch(x)
    mu, log_sigma = prior.model(x_z)
    sigma = torch.exp(log_sigma).view(1)

    eps = torch.randn((n_samples,), device=device, dtype=torch.float32)
    logn = mu.view(1) + sigma * eps
    n = torch.round(torch.exp(logn)).clamp(min=float(min_n), max=float(max_n)).to(torch.int64)
    return clamp_and_unique([int(v) for v in n.detach().cpu().tolist()], min_n=int(min_n), max_n=int(max_n))

