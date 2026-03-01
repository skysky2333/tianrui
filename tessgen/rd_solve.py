from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .ckpt import SurrogateBundle
from .transforms import invert_log_cols_torch
from .utils import Batch


@torch.no_grad()
def solve_rd_for_target(
    *,
    surrogate: SurrogateBundle,
    x: torch.Tensor,  # (N,2) on device
    edge_index: torch.Tensor,  # (2,E) on device
    n_nodes: int,
    n_edges: int,
    y_target_z: torch.Tensor,  # (1,Dy) on device
    rd_min: float,
    rd_max: float,
    grid_steps: int = 21,
    refine_iters: int = 24,
    device: torch.device,
) -> dict[str, Any]:
    """
    Solve a continuous RD value for a fixed graph to best match the target metrics under the surrogate.

    This treats RD as a *design variable* and optimizes:
        RD* = argmin_{RD in [rd_min, rd_max]} MSE( surrogate(graph, RD), y_target )

    Notes
    -----
    - Optimization is performed in *log(RD)* space for numerical stability.
    - Uses a coarse log-spaced grid + golden-section refinement within the best local bracket.
    - Returns predictions in raw metric space (after inverse-scaling + inverse-log transforms).
    """
    rd_min = float(rd_min)
    rd_max = float(rd_max)
    if not (rd_min > 0.0 and rd_max > rd_min):
        raise ValueError(f"Expected 0 < rd_min < rd_max, got rd_min={rd_min} rd_max={rd_max}")
    grid_steps = int(grid_steps)
    refine_iters = int(refine_iters)
    if grid_steps < 3:
        raise ValueError("grid_steps must be >= 3")
    if refine_iters < 0:
        raise ValueError("refine_iters must be >= 0")

    x = x.to(device=device, dtype=torch.float32)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    y_target_z = y_target_z.to(device=device, dtype=torch.float32)

    batch_vec = torch.zeros((int(n_nodes),), device=device, dtype=torch.long)
    n_nodes_t = torch.tensor([int(n_nodes)], device=device, dtype=torch.long)
    n_edges_t = torch.tensor([int(n_edges)], device=device, dtype=torch.long)

    def eval_rd(rd: float) -> tuple[float, torch.Tensor]:
        rd_f = float(rd)
        rd_t = torch.tensor([[rd_f]], device=device, dtype=torch.float32)
        batch = Batch(
            x=x,
            edge_index=edge_index,
            batch=batch_vec,
            rd=rd_t,
            y=y_target_z,
            n_nodes=n_nodes_t,
            n_edges=n_edges_t,
        )
        pred_z = surrogate.model(batch)  # (1,Dy)
        err = torch.mean((pred_z - y_target_z) ** 2).item()
        return float(err), pred_z

    # Coarse search on a log-spaced grid.
    rds_grid = np.geomspace(rd_min, rd_max, num=grid_steps, dtype=np.float64)
    errs_grid: list[float] = []
    best_err = float("inf")
    best_rd = float("nan")
    best_pred_z = None

    for rd in rds_grid.tolist():
        err, pred_z = eval_rd(float(rd))
        errs_grid.append(float(err))
        if err < best_err:
            best_err = float(err)
            best_rd = float(rd)
            best_pred_z = pred_z

    if best_pred_z is None:
        raise RuntimeError("Internal error: no RD candidates evaluated")

    best_i = int(np.argmin(np.asarray(errs_grid, dtype=np.float64)))
    hit_bound = bool(best_i == 0 or best_i == (len(rds_grid) - 1))

    # Golden-section refinement in log(RD) around the best grid point (if interior).
    if (not hit_bound) and refine_iters > 0:
        left = float(math.log(float(rds_grid[best_i - 1])))
        right = float(math.log(float(rds_grid[best_i + 1])))
        if right <= left:
            raise RuntimeError("Invalid refinement bracket")

        phi = 0.5 * (1.0 + math.sqrt(5.0))

        c = right - (right - left) / phi
        d = left + (right - left) / phi
        fc, pred_c = eval_rd(float(math.exp(c)))
        fd, pred_d = eval_rd(float(math.exp(d)))
        if fc < best_err:
            best_err, best_rd, best_pred_z = float(fc), float(math.exp(c)), pred_c
        if fd < best_err:
            best_err, best_rd, best_pred_z = float(fd), float(math.exp(d)), pred_d

        for _ in range(int(refine_iters)):
            if fc < fd:
                right = d
                d, fd, pred_d = c, fc, pred_c
                c = right - (right - left) / phi
                fc, pred_c = eval_rd(float(math.exp(c)))
                if fc < best_err:
                    best_err, best_rd, best_pred_z = float(fc), float(math.exp(c)), pred_c
            else:
                left = c
                c, fc, pred_c = d, fd, pred_d
                d = left + (right - left) / phi
                fd, pred_d = eval_rd(float(math.exp(d)))
                if fd < best_err:
                    best_err, best_rd, best_pred_z = float(fd), float(math.exp(d)), pred_d

        # Clamp for numerical safety.
        best_rd = float(min(max(best_rd, rd_min), rd_max))
        hit_bound = bool(best_rd <= rd_min * (1.0 + 1e-8) or best_rd >= rd_max * (1.0 - 1e-8))

    pred_z_best = best_pred_z
    pred_t = surrogate.scaler.inverse_transform_torch(pred_z_best)
    pred_vec = (
        invert_log_cols_torch(pred_t, surrogate.target_cols, surrogate.log_cols)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if pred_vec.ndim != 2 or pred_vec.shape[0] != 1:
        raise RuntimeError(f"Expected pred_vec shape (1,Dy), got {pred_vec.shape}")

    return {
        "rd_best": float(best_rd),
        "err_best": float(best_err),
        "hit_bound": bool(hit_bound),
        "pred_vec_best": [float(v) for v in pred_vec.squeeze(0).tolist()],
        "grid": {"rds": [float(x) for x in rds_grid.tolist()], "errs": [float(x) for x in errs_grid]},
    }
