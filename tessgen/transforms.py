from __future__ import annotations

import math

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


def _logit_scale(*, eps: float) -> float:
    eps_f = float(eps)
    if not (0.0 < eps_f < 0.5):
        raise ValueError(f"coord_eps must be in (0,0.5); got {eps_f}")
    # For x in [eps, 1-eps], logit(x) spans [-L, +L] where L = log((1-eps)/eps).
    # We scale by 2L so that the clamped endpoints map to [-0.5, +0.5], matching the
    # typical coordinate scale of the original unit box.
    L = math.log((1.0 - eps_f) / eps_f)
    return float(2.0 * L)


def coord01_to_diffusion_space_torch(
    x01: torch.Tensor,
    *,
    coord_space: str,
    coord_eps: float = 1e-4,
) -> torch.Tensor:
    """
    Map coordinates in [0,1] into the diffusion space.

    Supported coord_space:
      - "unit": identity (diffuse directly in [0,1] space)
      - "logit": scaled logit transform to an (approximately) unbounded space

    Notes
    -----
    For "logit", we clamp to [eps, 1-eps] before applying logit for numerical safety.
    """
    mode = str(coord_space)
    if mode == "unit":
        return x01
    if mode != "logit":
        raise ValueError(f"Unsupported coord_space={coord_space!r} (expected 'unit'|'logit')")

    scale = _logit_scale(eps=float(coord_eps))
    x = x01.clamp(min=float(coord_eps), max=1.0 - float(coord_eps))
    # logit(x) = log(x) - log(1-x), using log1p for stability near 0.
    u = (torch.log(x) - torch.log1p(-x)) / float(scale)
    return u


def diffusion_space_to_coord01_torch(
    u: torch.Tensor,
    *,
    coord_space: str,
    coord_eps: float = 1e-4,
) -> torch.Tensor:
    """
    Inverse map from diffusion space back to coordinates in [0,1].
    """
    mode = str(coord_space)
    if mode == "unit":
        return u
    if mode != "logit":
        raise ValueError(f"Unsupported coord_space={coord_space!r} (expected 'unit'|'logit')")

    scale = _logit_scale(eps=float(coord_eps))
    x01 = torch.sigmoid(u * float(scale))
    # NOTE: we do not clamp to [eps, 1-eps] here; sigmoid already yields (0,1) unless
    # it saturates numerically. Consumers may clamp if they require strict bounds.
    return x01.clamp(0.0, 1.0)
