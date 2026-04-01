from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .ckpt import EdgeBundle, NPriorBundle, NodeDiffusionBundle, SurrogateBundle
from .constants import COORD_MIN, COORD_RANGE
from .data import GraphStore, undirected_to_directed_edge_index
from .generation import ddpm_sample_coords, sample_edges_from_coords
from .metrics import mae, pearson_r, r2_score, rmse, spearman_r
from .n_select import clamp_and_unique, sample_n_candidates_from_prior
from .reporting import save_histogram_both, save_scatter_plot_both
from .rd_solve import solve_rd_for_target
from .transforms import apply_log_cols_torch, invert_log_cols_torch
from .utils import Batch, ensure_dir
from .viz import save_graph_figure


def _compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {
        "n": int(y_true.size),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    if y_true.size:
        out["frac_over"] = float(np.mean(y_pred > y_true))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = y_pred / y_true
        ratio = ratio[np.isfinite(ratio)]
        out["median_ratio"] = float(np.median(ratio)) if ratio.size else float("nan")

        pos = (y_true > 0.0) & (y_pred > 0.0)
        out["n_log"] = int(np.sum(pos))
        if int(out["n_log"]) > 0:
            log_err = np.log10(y_pred[pos]) - np.log10(y_true[pos])
            out["mean_log10_err"] = float(np.mean(log_err))
            out["rmse_log10"] = float(np.sqrt(float(np.mean(log_err**2))))
        else:
            out["mean_log10_err"] = float("nan")
            out["rmse_log10"] = float("nan")
    else:
        out["frac_over"] = float("nan")
        out["median_ratio"] = float("nan")
        out["n_log"] = 0
        out["mean_log10_err"] = float("nan")
        out["rmse_log10"] = float("nan")

    return out


@torch.no_grad()
def run_cycle_eval(
    *,
    df: pd.DataFrame,
    row_indices: list[int],
    tess_root: str,
    surrogate: SurrogateBundle,
    node_bundle: NodeDiffusionBundle,
    edge_bundle: EdgeBundle,
    n_prior: NPriorBundle | None = None,
    device: torch.device,
    rd_mode: str = "fixed",  # fixed|solve
    rd_min: float = 0.01,
    rd_max: float = 0.2,
    rd_grid_steps: int = 21,
    rd_refine_iters: int = 24,
    k_best: int,
    deg_cap: int,
    min_n: int,
    max_n: int,
    edge_thr: float = 0.5,
    n_mode: str = "true",  # true|fixed|candidates|prior
    n_fixed: int = 0,
    n_candidates: list[int] | None = None,
    n_prior_samples: int = 12,
    out_dir: str,
    save_row_figs: bool = True,
    save_graph_files: bool = False,
    progress_every: int = 1,  # <=0 disables progress
    progress_prefix: str = "cycle",
) -> dict:
    """
    End-to-end evaluation:
      metrics + RD (+ N selection) -> (node diffusion -> edge model) -> surrogate -> metrics

    Writes:
      - rows.jsonl (per-row details)
      - figures/ (summary plots, PNG+SVG)
      - graphs_true/, graphs_single/, graphs_best/ (per-row plots, PNG+SVG) if save_row_figs
      - graphs_tried/ (all best-of-k tries per row, PNG+SVG) if save_row_figs
    """
    out = Path(out_dir)
    ensure_dir(str(out))

    if not row_indices:
        raise ValueError("row_indices is empty")
    k_best = int(k_best)
    if k_best <= 0:
        raise ValueError("k_best must be >= 1")

    rd_mode_s = str(rd_mode)
    if rd_mode_s not in {"fixed", "solve"}:
        raise ValueError(f"Unsupported rd_mode={rd_mode_s!r} (expected 'fixed' or 'solve')")
    rd_min_f = float(rd_min)
    rd_max_f = float(rd_max)
    if rd_mode_s == "solve":
        if not (rd_min_f > 0.0 and rd_max_f > rd_min_f):
            raise ValueError(f"Expected 0 < rd_min < rd_max, got rd_min={rd_min_f} rd_max={rd_max_f}")
        if int(rd_grid_steps) < 3:
            raise ValueError("rd_grid_steps must be >= 3")
        if int(rd_refine_iters) < 0:
            raise ValueError("rd_refine_iters must be >= 0")

    need_cols = set(node_bundle.cond_cols) | set(surrogate.target_cols)
    need_rd_col = bool(node_bundle.use_rd) or (rd_mode_s == "fixed")
    if str(n_mode) == "prior":
        if n_prior is None:
            raise ValueError("n_prior must be provided when n_mode='prior'")
        need_cols |= set(n_prior.cond_cols)
        need_rd_col = need_rd_col or bool(n_prior.use_rd)
    if need_rd_col:
        need_cols.add("RD")
    missing = [c for c in sorted(need_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    rs_col = "RS" if "RS" in surrogate.target_cols else surrogate.target_cols[0]
    rs_idx = surrogate.target_cols.index(rs_col)

    rows_path = out / "rows.jsonl"
    figs_dir = out / "figures"
    graphs_true_dir = out / "graphs_true"
    graphs_single_dir = out / "graphs_single"
    graphs_best_dir = out / "graphs_best"
    graphs_tried_dir = out / "graphs_tried"

    y_true_all: list[np.ndarray] = []
    y_truegraph_all: list[np.ndarray] = []
    y_single_all: list[np.ndarray] = []
    y_best_all: list[np.ndarray] = []
    rd_all: list[float] = []

    store = GraphStore(tess_root=tess_root)

    disable_progress = int(progress_every) <= 0
    total_units = int(len(row_indices) * int(k_best))
    t0 = time.time()

    with tqdm(
        total=int(total_units),
        desc=str(progress_prefix),
        file=sys.stderr,
        disable=bool(disable_progress),
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        with open(rows_path, "w") as f_rows:
            for row_pos, row_idx in enumerate(row_indices, start=1):
                row_idx = int(row_idx)
                r = df.iloc[row_idx]
                graph_id = int((row_idx // 5) + 1)
                rd_label = float(r["RD"]) if "RD" in df.columns else float("nan")
                rd = float(rd_label)
                if bool(node_bundle.use_rd) and (not np.isfinite(rd)):
                    raise ValueError(f"Row idx={row_idx} has non-finite RD={rd!r}, but node diffusion requires RD.")
                rd_t_fixed = torch.tensor([[float(rd)]], device=device, dtype=torch.float32)
                rd_t_diff = rd_t_fixed if bool(node_bundle.use_rd) else None

                # True graph baseline (surrogate on ground-truth graph)
                pbar.set_postfix_str(f"row={row_pos}/{len(row_indices)} idx={row_idx} step=true_graph", refresh=True)
                t_base0 = time.perf_counter()
                g_true = store.get(graph_id)
                n_nodes_true = int(g_true.n_nodes)
                n_edges_true = int(g_true.n_edges)

                y_true_vec = np.array([float(r[c]) for c in surrogate.target_cols], dtype=np.float32)
                y_target = torch.from_numpy(y_true_vec).to(device=device, dtype=torch.float32).unsqueeze(0)
                y_target_t = apply_log_cols_torch(y_target, surrogate.target_cols, surrogate.log_cols)
                y_target_z = surrogate.scaler.transform_torch(y_target_t)

                rd_true_solved = None
                rd_true_hit_bound = None
                if rd_mode_s == "fixed":
                    batch_true = Batch(
                        x=g_true.coords01.to(device),
                        edge_index=g_true.edge_index.to(device),
                        batch=torch.zeros((n_nodes_true,), device=device, dtype=torch.long),
                        rd=rd_t_fixed,
                        y=y_target_z,
                        n_nodes=torch.tensor([n_nodes_true], device=device, dtype=torch.long),
                        n_edges=torch.tensor([n_edges_true], device=device, dtype=torch.long),
                    )
                    pred_z_true = surrogate.model(batch_true)
                    err_true = torch.mean((pred_z_true - y_target_z) ** 2).item()
                    pred_t_true = surrogate.scaler.inverse_transform_torch(pred_z_true)
                    pred_vec_true = (
                        invert_log_cols_torch(pred_t_true, surrogate.target_cols, surrogate.log_cols)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                else:
                    solved = solve_rd_for_target(
                        surrogate=surrogate,
                        x=g_true.coords01.to(device),
                        edge_index=g_true.edge_index.to(device),
                        n_nodes=int(n_nodes_true),
                        n_edges=int(n_edges_true),
                        y_target_z=y_target_z,
                        rd_min=rd_min_f,
                        rd_max=rd_max_f,
                        grid_steps=int(rd_grid_steps),
                        refine_iters=int(rd_refine_iters),
                        device=device,
                    )
                    err_true = float(solved["err_best"])
                    pred_vec_true = np.asarray(solved["pred_vec_best"], dtype=np.float32)
                    rd_true_solved = float(solved["rd_best"])
                    rd_true_hit_bound = bool(solved["hit_bound"])
                t_base = float(time.perf_counter() - t_base0)
                pbar.set_postfix_str(
                    f"row={row_pos}/{len(row_indices)} idx={row_idx} step=true_graph base={t_base:.3g}s err={err_true:.3g}",
                    refresh=True,
                )

                # Condition values (metrics) for node diffusion (raw -> log-transformed subset)
                cond_vals = torch.tensor([[float(r[c]) for c in node_bundle.cond_cols]], device=device, dtype=torch.float32)
                cond_vals = apply_log_cols_torch(cond_vals, node_bundle.cond_cols, node_bundle.log_cols)

                n_mode_s = str(n_mode)
                if n_mode_s == "true":
                    n_values = [int(n_nodes_true)]
                elif n_mode_s == "fixed":
                    if int(n_fixed) <= 0:
                        raise ValueError("n_fixed must be > 0 when n_mode='fixed'")
                    n_values = [int(n_fixed)]
                elif n_mode_s == "candidates":
                    if not n_candidates:
                        raise ValueError("n_candidates must be non-empty when n_mode='candidates'")
                    n_values = clamp_and_unique([int(x) for x in n_candidates], min_n=int(min_n), max_n=int(max_n))
                elif n_mode_s == "prior":
                    if n_prior is None:
                        raise ValueError("n_prior must be provided when n_mode='prior'")
                    cond_vals_prior_raw = torch.tensor(
                        [[float(r[c]) for c in n_prior.cond_cols]],
                        device=device,
                        dtype=torch.float32,
                    )
                    n_values = sample_n_candidates_from_prior(
                        n_prior,
                        rd=float(rd),
                        cond_vals_raw=cond_vals_prior_raw,
                        n_samples=int(n_prior_samples),
                        min_n=int(min_n),
                        max_n=int(max_n),
                        device=device,
                    )
                else:
                    raise ValueError(f"Unsupported n_mode={n_mode_s!r} (expected true|fixed|candidates|prior)")
                if not n_values:
                    raise RuntimeError("No N candidates produced")

                single = None
                best = None
                best_j = None
                tried: list[dict] = []
                for j in range(k_best):
                    n_nodes = int(n_values[j % len(n_values)])
                    logn_t = torch.tensor([[float(np.log(float(n_nodes)))]], device=device, dtype=torch.float32)
                    if bool(node_bundle.use_rd):
                        if rd_t_diff is None:
                            raise RuntimeError("Internal error: rd_t_diff is None but node_bundle.use_rd=True")
                        cond_full = torch.cat([rd_t_diff, logn_t, cond_vals], dim=-1)
                    else:
                        cond_full = torch.cat([logn_t, cond_vals], dim=-1)
                    cond_z = node_bundle.cond_scaler.transform_torch(cond_full).squeeze(0)

                    pbar.set_postfix_str(
                        f"row={row_pos}/{len(row_indices)} idx={row_idx} step=ddpm j={j+1}/{k_best} rd={rd:.4g} N={n_nodes}",
                        refresh=True,
                    )
                    t_ddpm0 = time.perf_counter()
                    coords01 = ddpm_sample_coords(
                        schedule=node_bundle.schedule,
                        denoiser=node_bundle.denoiser,
                        cond_z=cond_z,
                        n_nodes=n_nodes,
                        k_nn=node_bundle.k_nn,
                        coord_space=str(getattr(node_bundle, "coord_space", "unit")),
                        coord_eps=float(getattr(node_bundle, "coord_eps", 1e-4)),
                        device=device,
                    )
                    t_ddpm = float(time.perf_counter() - t_ddpm0)

                    pbar.set_postfix_str(
                        f"row={row_pos}/{len(row_indices)} idx={row_idx} step=edge j={j+1}/{k_best} rd={rd:.4g} N={n_nodes} ddpm={t_ddpm:.3g}s",
                        refresh=True,
                    )
                    t_edge0 = time.perf_counter()
                    edges_uv = sample_edges_from_coords(
                        edge_bundle=edge_bundle,
                        coords01=coords01,
                        deg_cap=int(deg_cap),
                        edge_thr=float(edge_thr),
                        ensure_connected=True,
                        device=device,
                    )
                    t_edge = float(time.perf_counter() - t_edge0)

                    edge_index = undirected_to_directed_edge_index(edges_uv).to(device)
                    rd_solved = None
                    rd_hit_bound = None
                    if rd_mode_s == "fixed":
                        batch = Batch(
                            x=coords01,
                            edge_index=edge_index,
                            batch=torch.zeros((n_nodes,), device=device, dtype=torch.long),
                            rd=rd_t_fixed,
                            y=y_target_z,
                            n_nodes=torch.tensor([n_nodes], device=device, dtype=torch.long),
                            n_edges=torch.tensor([int(edges_uv.shape[0])], device=device, dtype=torch.long),
                        )

                    pbar.set_postfix_str(
                        f"row={row_pos}/{len(row_indices)} idx={row_idx} step=surrogate j={j+1}/{k_best} rd={rd:.4g} N={n_nodes} ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s",
                        refresh=True,
                    )
                    t_sur0 = time.perf_counter()
                    if rd_mode_s == "fixed":
                        pred_z = surrogate.model(batch)
                        err = torch.mean((pred_z - y_target_z) ** 2).item()
                        pred_t = surrogate.scaler.inverse_transform_torch(pred_z)
                        pred_vec = (
                            invert_log_cols_torch(pred_t, surrogate.target_cols, surrogate.log_cols)
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    else:
                        solved = solve_rd_for_target(
                            surrogate=surrogate,
                            x=coords01,
                            edge_index=edge_index,
                            n_nodes=int(n_nodes),
                            n_edges=int(edges_uv.shape[0]),
                            y_target_z=y_target_z,
                            rd_min=rd_min_f,
                            rd_max=rd_max_f,
                            grid_steps=int(rd_grid_steps),
                            refine_iters=int(rd_refine_iters),
                            device=device,
                        )
                        err = float(solved["err_best"])
                        pred_vec = np.asarray(solved["pred_vec_best"], dtype=np.float32)
                        rd_solved = float(solved["rd_best"])
                        rd_hit_bound = bool(solved["hit_bound"])
                    t_sur = float(time.perf_counter() - t_sur0)

                    cand = {
                        "j": int(j),
                        "err_mse_z": float(err),
                        "rd_solved": float(rd_solved) if rd_solved is not None else None,
                        "rd_hit_bound": bool(rd_hit_bound) if rd_hit_bound is not None else None,
                        "n_nodes": int(n_nodes),
                        "n_edges": int(edges_uv.shape[0]),
                        "coords01": coords01.detach().cpu().numpy(),
                        "edges_uv": edges_uv,
                        "pred_vec": pred_vec,
                    }
                    tried.append(cand)
                    if j == 0:
                        single = cand
                        best = cand
                        best_j = int(j)
                    else:
                        if cand["err_mse_z"] < float(best["err_mse_z"]):  # type: ignore[index]
                            best = cand
                            best_j = int(j)

                    best_err = float(best["err_mse_z"]) if best is not None else float("nan")
                    pbar.set_postfix_str(
                        (
                            f"row={row_pos}/{len(row_indices)} idx={row_idx} step=done j={j+1}/{k_best} rd={rd:.4g} N={n_nodes} "
                            f"ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s sur={t_sur:.3g}s err={float(err):.3g} best={best_err:.3g}"
                        ),
                        refresh=True,
                    )
                    pbar.update(1)

                if single is None or best is None:
                    raise RuntimeError("No candidates were generated")
                if best_j is None:
                    raise RuntimeError("Internal error: best_j is None")

                true_pred = {c: float(pred_vec_true[i]) for i, c in enumerate(surrogate.target_cols)}
                single_pred = {c: float(single["pred_vec"][i]) for i, c in enumerate(surrogate.target_cols)}
                best_pred = {c: float(best["pred_vec"][i]) for i, c in enumerate(surrogate.target_cols)}

                y_true_all.append(y_true_vec)
                y_truegraph_all.append(pred_vec_true)
                y_single_all.append(single["pred_vec"])
                y_best_all.append(best["pred_vec"])
                rd_all.append(rd)

                rs_true = float(y_true_vec[rs_idx])
                rs_truegraph = float(pred_vec_true[rs_idx])
                rs_single = float(single["pred_vec"][rs_idx])
                rs_best = float(best["pred_vec"][rs_idx])

                row_record = {
                    "row_idx": int(row_idx),
                    "graph_id": int(graph_id),
                    "rd": float(rd),
                    "rd_mode": str(rd_mode_s),
                    "rd_cond": float(rd) if bool(node_bundle.use_rd) else None,
                    "targets": {c: float(y_true_vec[i]) for i, c in enumerate(surrogate.target_cols)},
                    "rs_col": rs_col,
                    "true_graph": {
                        "err_mse_z": float(err_true),
                        "n_nodes": int(n_nodes_true),
                        "n_edges": int(n_edges_true),
                        "rd_used": float(rd) if rd_mode_s == "fixed" else None,
                        "rd_solved": float(rd_true_solved) if rd_true_solved is not None else None,
                        "rd_hit_bound": bool(rd_true_hit_bound) if rd_true_hit_bound is not None else None,
                        "pred": true_pred,
                    },
                    "single": {
                        "j": 1,
                        "err_mse_z": float(single["err_mse_z"]),
                        "n_nodes": int(single["n_nodes"]),
                        "n_edges": int(single["n_edges"]),
                        "rd_used": float(rd) if rd_mode_s == "fixed" else None,
                        "rd_solved": single.get("rd_solved"),
                        "rd_hit_bound": single.get("rd_hit_bound"),
                        "pred": single_pred,
                    },
                    "best": {
                        "k": int(k_best),
                        "j": int(best_j) + 1,
                        "err_mse_z": float(best["err_mse_z"]),
                        "n_nodes": int(best["n_nodes"]),
                        "n_edges": int(best["n_edges"]),
                        "rd_used": float(rd) if rd_mode_s == "fixed" else None,
                        "rd_solved": best.get("rd_solved"),
                        "rd_hit_bound": best.get("rd_hit_bound"),
                        "pred": best_pred,
                    },
                    "tried": [
                        {
                            "j": int(cand2["j"]) + 1,
                            "is_best": bool(int(cand2["j"]) == int(best_j)),
                            "err_mse_z": float(cand2["err_mse_z"]),
                            "n_nodes": int(cand2["n_nodes"]),
                            "n_edges": int(cand2["n_edges"]),
                            "rd_used": float(rd) if rd_mode_s == "fixed" else None,
                            "rd_solved": cand2.get("rd_solved"),
                            "rd_hit_bound": cand2.get("rd_hit_bound"),
                        }
                        for cand2 in tried
                    ],
                }

                if save_row_figs:
                    pbar.set_postfix_str(f"row={row_pos}/{len(row_indices)} idx={row_idx} step=save_figs", refresh=True)
                    coords_true = g_true.coords01.detach().cpu().numpy() * COORD_RANGE + COORD_MIN
                    coords_single = single["coords01"] * COORD_RANGE + COORD_MIN
                    coords_best = best["coords01"] * COORD_RANGE + COORD_MIN
                    rd_true_s = f" rd_solved={float(rd_true_solved):.4g}" if rd_true_solved is not None else ""
                    rd_true_bound_s = " hit_bound=True" if bool(rd_true_hit_bound) else ""
                    rd_single_s = (
                        f" rd_solved={float(single.get('rd_solved')):.4g}" if single.get("rd_solved") is not None else ""
                    )
                    rd_single_bound_s = " hit_bound=True" if bool(single.get("rd_hit_bound")) else ""
                    rd_best_s = f" rd_solved={float(best.get('rd_solved')):.4g}" if best.get("rd_solved") is not None else ""
                    rd_best_bound_s = " hit_bound=True" if bool(best.get("rd_hit_bound")) else ""
                    true_png = graphs_true_dir / f"graph_row{row_idx}.png"
                    true_svg = graphs_true_dir / f"graph_row{row_idx}.svg"
                    single_png = graphs_single_dir / f"graph_row{row_idx}.png"
                    single_svg = graphs_single_dir / f"graph_row{row_idx}.svg"
                    best_png = graphs_best_dir / f"graph_row{row_idx}.png"
                    best_svg = graphs_best_dir / f"graph_row{row_idx}.svg"

                    save_graph_figure(
                        coords=coords_true,
                        edges_uv=g_true.edges_undirected.detach().cpu().numpy(),
                        out_png=str(true_png),
                        out_svg=str(true_svg),
                        title=(
                            f"true | row={row_idx} rd={rd:.4g} N={n_nodes_true} E={n_edges_true}\n"
                            f"{rs_col}_true={rs_true:.4g} {rs_col}_pred={rs_truegraph:.4g} err_mse_z={err_true:.3g}"
                            f"{rd_true_s}{rd_true_bound_s}"
                        ),
                    )
                    save_graph_figure(
                        coords=coords_single,
                        edges_uv=single["edges_uv"],
                        out_png=str(single_png),
                        out_svg=str(single_svg),
                        title=(
                            f"single | row={row_idx} rd={rd:.4g} err_mse_z={single['err_mse_z']:.3g}\n"
                            f"{rs_col}_true={rs_true:.4g} {rs_col}_pred={rs_single:.4g}{rd_single_s}{rd_single_bound_s}"
                        ),
                    )
                    save_graph_figure(
                        coords=coords_best,
                        edges_uv=best["edges_uv"],
                        out_png=str(best_png),
                        out_svg=str(best_svg),
                        title=(
                            f"best | row={row_idx} rd={rd:.4g} err_mse_z={best['err_mse_z']:.3g}\n"
                            f"{rs_col}_true={rs_true:.4g} {rs_col}_pred={rs_best:.4g}{rd_best_s}{rd_best_bound_s}"
                        ),
                    )

                    tried_figs: list[dict[str, object]] = []
                    for cand2 in tried:
                        j = int(cand2["j"]) + 1
                        coords_try = cand2["coords01"] * COORD_RANGE + COORD_MIN
                        row_try_dir = graphs_tried_dir / f"row{row_idx}"
                        try_png = row_try_dir / f"try_{j:02d}.png"
                        try_svg = row_try_dir / f"try_{j:02d}.svg"

                        rs_try = float(cand2["pred_vec"][rs_idx])
                        rd_solved_try = cand2.get("rd_solved")
                        rd_hit_bound_try = cand2.get("rd_hit_bound")
                        rd_solved_s = (
                            f" rd_solved={float(rd_solved_try):.4g}" if rd_solved_try is not None else ""
                        )
                        rd_bound_s = " hit_bound=True" if bool(rd_hit_bound_try) else ""
                        best_s = " best=True" if int(cand2["j"]) == int(best_j) else ""

                        save_graph_figure(
                            coords=coords_try,
                            edges_uv=cand2["edges_uv"],
                            out_png=str(try_png),
                            out_svg=str(try_svg),
                            title=(
                                f"try {j}/{k_best} | row={row_idx} N={int(cand2['n_nodes'])} E={int(cand2['n_edges'])} "
                                f"err_mse_z={float(cand2['err_mse_z']):.3g}\n"
                                f"rd_label={rd:.4g}{rd_solved_s}{rd_bound_s}{best_s}\n"
                                f"{rs_col}_true={rs_true:.4g} {rs_col}_pred={rs_try:.4g}"
                            ),
                        )
                        tried_figs.append(
                            {
                                "j": int(j),
                                "is_best": bool(int(cand2["j"]) == int(best_j)),
                                "png": str(try_png),
                                "svg": str(try_svg),
                                "rd_solved": rd_solved_try,
                                "rd_hit_bound": rd_hit_bound_try,
                            }
                        )
                    row_record["figures"] = {
                        "true_png": str(true_png),
                        "true_svg": str(true_svg),
                        "single_png": str(single_png),
                        "single_svg": str(single_svg),
                        "best_png": str(best_png),
                        "best_svg": str(best_svg),
                        "tried": tried_figs,
                    }

                if save_graph_files:
                    for mode, cand2 in [("single", single), ("best", best)]:
                        coords = cand2["coords01"] * COORD_RANGE + COORD_MIN
                        node_path = out / f"Node_{mode}_row{row_idx}.txt"
                        conn_path = out / f"Connection_{mode}_row{row_idx}.txt"
                        with open(node_path, "w") as fn:
                            for nid, (x, y) in enumerate(coords, start=1):
                                fn.write(f"{nid}, {x:.6e}, {y:.6e}\n")
                        with open(conn_path, "w") as fe:
                            for eid, (u, v) in enumerate(cand2["edges_uv"].tolist(), start=1):
                                fe.write(f"{eid}, {int(u)+1}, {int(v)+1}\n")

                f_rows.write(json.dumps(row_record) + "\n")

        pbar.set_postfix_str("step=summary_plots", refresh=True)
        y_true_np = np.stack(y_true_all, axis=0)
        y_truegraph_np = np.stack(y_truegraph_all, axis=0)
        y_single_np = np.stack(y_single_all, axis=0)
        y_best_np = np.stack(y_best_all, axis=0)
        rd_np = np.asarray(rd_all, dtype=np.float64)

        rs_true_all = y_true_np[:, rs_idx]
        rs_truegraph_all = y_truegraph_np[:, rs_idx]
        rs_single_all = y_single_np[:, rs_idx]
        rs_best_all = y_best_np[:, rs_idx]

        metrics = {
            "rs_col": rs_col,
            "true_graph": _compute_metrics(y_true=rs_true_all, y_pred=rs_truegraph_all),
            "single": _compute_metrics(y_true=rs_true_all, y_pred=rs_single_all),
            "best": _compute_metrics(y_true=rs_true_all, y_pred=rs_best_all),
            "per_rd": {},
        }
        rd_vals = np.unique(rd_np[np.isfinite(rd_np)])
        for rd_val in rd_vals.tolist():
            rd_f = float(rd_val)
            mask = rd_np == rd_f
            metrics["per_rd"][str(rd_f)] = {
                "n": int(mask.sum()),
                "true_graph": _compute_metrics(y_true=rs_true_all[mask], y_pred=rs_truegraph_all[mask]),
                "single": _compute_metrics(y_true=rs_true_all[mask], y_pred=rs_single_all[mask]),
                "best": _compute_metrics(y_true=rs_true_all[mask], y_pred=rs_best_all[mask]),
            }

        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_true_graph.png"),
            out_svg=str(figs_dir / "rs_scatter_true_graph.svg"),
            x=rs_true_all.astype(np.float64).tolist(),
            y=rs_truegraph_all.astype(np.float64).tolist(),
            title=f"{rs_col}: true vs predicted (true graph baseline)",
            xlabel=f"true {rs_col}",
            ylabel=f"pred {rs_col}",
        )
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_single.png"),
            out_svg=str(figs_dir / "rs_scatter_single.svg"),
            x=rs_true_all.astype(np.float64).tolist(),
            y=rs_single_all.astype(np.float64).tolist(),
            title=f"{rs_col}: true vs predicted (single)",
            xlabel=f"true {rs_col}",
            ylabel=f"pred {rs_col}",
        )
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_best.png"),
            out_svg=str(figs_dir / "rs_scatter_best.svg"),
            x=rs_true_all.astype(np.float64).tolist(),
            y=rs_best_all.astype(np.float64).tolist(),
            title=f"{rs_col}: true vs predicted (best-of-{k_best})",
            xlabel=f"true {rs_col}",
            ylabel=f"pred {rs_col}",
        )
        save_histogram_both(
            out_png=str(figs_dir / "rs_error_hist_true_graph.png"),
            out_svg=str(figs_dir / "rs_error_hist_true_graph.svg"),
            values=(rs_truegraph_all - rs_true_all).astype(np.float64).tolist(),
            title=f"{rs_col} error (pred-true) (true graph baseline)",
            xlabel=f"{rs_col}_pred - {rs_col}_true",
            bins=80,
        )
        save_histogram_both(
            out_png=str(figs_dir / "rs_error_hist_single.png"),
            out_svg=str(figs_dir / "rs_error_hist_single.svg"),
            values=(rs_single_all - rs_true_all).astype(np.float64).tolist(),
            title=f"{rs_col} error (pred-true) (single)",
            xlabel=f"{rs_col}_pred - {rs_col}_true",
            bins=80,
        )
        save_histogram_both(
            out_png=str(figs_dir / "rs_error_hist_best.png"),
            out_svg=str(figs_dir / "rs_error_hist_best.svg"),
            values=(rs_best_all - rs_true_all).astype(np.float64).tolist(),
            title=f"{rs_col} error (pred-true) (best-of-{k_best})",
            xlabel=f"{rs_col}_pred - {rs_col}_true",
            bins=80,
        )

        artifacts: dict[str, str] = {
            "rows_jsonl": str(rows_path),
            "figures_dir": str(figs_dir),
        }
        if save_row_figs:
            artifacts["graphs_true_dir"] = str(graphs_true_dir)
            artifacts["graphs_single_dir"] = str(graphs_single_dir)
            artifacts["graphs_best_dir"] = str(graphs_best_dir)
            artifacts["graphs_tried_dir"] = str(graphs_tried_dir)

        pbar.set_postfix_str("step=complete", refresh=True)

    return {
        "task": "cycle_eval",
        "rows": {"n": int(len(row_indices))},
        "k_best": int(k_best),
        "deg_cap": int(deg_cap),
        "edge_thr": float(edge_thr),
        "metrics": metrics,
        "artifacts": artifacts,
        "elapsed_sec": float(time.time() - t0),
    }
