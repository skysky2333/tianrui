from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..ckpt import load_edge_model, load_n_prior, load_node_diffusion, load_surrogate
from ..constants import COORD_MIN, COORD_RANGE
from ..data import undirected_to_directed_edge_index
from ..generation import ddpm_sample_coords, sample_edges_from_coords
from ..n_select import clamp_and_unique, sample_n_candidates_from_prior
from ..rd_solve import solve_rd_for_target
from ..transforms import apply_log_cols_torch, invert_log_cols_torch
from ..outdirs import finalize_out_dir, make_timestamped_run_dir
from ..utils import Batch, device_from_arg, set_seed
from ..viz import save_graph_figure


def _parse_kv_list(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --cond item (expected KEY=VALUE): {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate graphs conditioned on metrics; optionally solve RD during scoring")
    p.add_argument(
        "--rd",
        type=float,
        default=None,
        help="If provided, fix RD (used for scoring in rd_mode=fixed; used as the RD conditioning value if node diffusion expects RD).",
    )
    p.add_argument(
        "--rd_candidates",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.15, 0.2],
        help="Candidate RD values (used for rd_mode=fixed when --rd is omitted; also used to condition diffusion when diffusion expects RD and --rd is omitted).",
    )
    p.add_argument("--rd_mode", type=str, default="fixed", help="How to handle RD during scoring: fixed|solve")
    p.add_argument("--rd_min", type=float, default=0.01, help="RD solve lower bound (used when rd_mode=solve)")
    p.add_argument("--rd_max", type=float, default=0.2, help="RD solve upper bound (used when rd_mode=solve)")
    p.add_argument("--rd_grid_steps", type=int, default=21, help="Coarse grid steps for RD solve (used when rd_mode=solve)")
    p.add_argument("--rd_refine_iters", type=int, default=24, help="Golden-section refinement iterations (used when rd_mode=solve)")
    p.add_argument("--cond", type=str, nargs="*", default=[], help="Condition values as KEY=VALUE, e.g. RS=0.01")
    p.add_argument("--k", type=int, default=2, help="Samples per (RD, N) combination")
    p.add_argument("--top_m", type=int, default=3, help="How many best samples to save")
    p.add_argument("--deg_cap", type=int, default=12)
    p.add_argument("--edge_thr", type=float, default=0.5, help="Edge probability threshold used during edge sampling")
    p.add_argument("--min_n", type=int, default=64)
    p.add_argument("--max_n", type=int, default=5000)
    p.add_argument("--n_nodes", type=int, default=None, help="If set, fix the number of nodes N.")
    p.add_argument("--n_candidates", type=int, nargs="*", default=[], help="Candidate N values to search over.")
    p.add_argument("--n_prior_ckpt", type=str, default="", help="NPrior checkpoint used to sample candidate N values.")
    p.add_argument("--n_prior_samples", type=int, default=12, help="How many N candidates to sample from NPrior (per RD).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--surrogate_ckpt", type=str, required=True)
    p.add_argument("--node_ckpt", type=str, required=True)
    p.add_argument("--edge_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    device = device_from_arg(args.device)

    if not (0.0 <= float(args.edge_thr) <= 1.0):
        raise SystemExit(f"--edge_thr must be in [0,1]; got {args.edge_thr}")

    cond_dict = _parse_kv_list(list(args.cond))

    surrogate = load_surrogate(args.surrogate_ckpt, device=device)
    node_bundle = load_node_diffusion(args.node_ckpt, device=device)
    edge_bundle = load_edge_model(args.edge_ckpt, device=device)

    rd_mode_s = str(args.rd_mode)
    if rd_mode_s not in {"fixed", "solve"}:
        raise SystemExit(f"--rd_mode must be one of: fixed, solve (got {rd_mode_s!r})")
    rd_min_f = float(args.rd_min)
    rd_max_f = float(args.rd_max)
    if rd_mode_s == "solve" and not (rd_min_f > 0.0 and rd_max_f > rd_min_f):
        raise SystemExit(f"For --rd_mode solve: expected 0 < --rd_min < --rd_max; got rd_min={rd_min_f} rd_max={rd_max_f}")
    if rd_mode_s == "solve" and int(args.rd_grid_steps) < 3:
        raise SystemExit("--rd_grid_steps must be >= 3")
    if rd_mode_s == "solve" and int(args.rd_refine_iters) < 0:
        raise SystemExit("--rd_refine_iters must be >= 0")

    n_prior = None
    if args.n_nodes is None and not list(args.n_candidates):
        if not str(args.n_prior_ckpt):
            raise SystemExit("Provide --n_nodes, --n_candidates, or --n_prior_ckpt for automatic N selection.")
        n_prior = load_n_prior(args.n_prior_ckpt, device=device)

    # Validate required condition keys
    for c in node_bundle.cond_cols:
        if c not in cond_dict:
            raise SystemExit(f"Missing required condition {c}. Provide via --cond {c}=...")
    for c in surrogate.target_cols:
        if c not in cond_dict:
            raise SystemExit(f"Missing required target {c} for scoring. Provide via --cond {c}=...")
    if n_prior is not None:
        for c in n_prior.cond_cols:
            if c not in cond_dict:
                raise SystemExit(f"Missing required NPrior condition {c}. Provide via --cond {c}=...")

    # RD loop values:
    # - rd_mode=fixed: iterate RD values for scoring (and diffusion conditioning if needed)
    # - rd_mode=solve: only iterate RD values when diffusion itself needs RD conditioning
    if rd_mode_s == "fixed":
        if args.rd is not None:
            rd_loop_values: list[float | None] = [float(args.rd)]
        else:
            rd_loop_values = [float(x) for x in list(args.rd_candidates)]
            if not rd_loop_values:
                raise SystemExit("No RD provided: pass --rd, or provide at least one value in --rd_candidates.")
    else:
        if bool(node_bundle.use_rd):
            if args.rd is not None:
                rd_loop_values = [float(args.rd)]
            else:
                rd_loop_values = [float(x) for x in list(args.rd_candidates)]
                if not rd_loop_values:
                    raise SystemExit(
                        "Diffusion checkpoint requires RD as conditioning input. Pass --rd, or provide at least one value in --rd_candidates."
                    )
        else:
            rd_loop_values = [None]
    k_per_combo = int(args.k)
    if k_per_combo <= 0:
        raise SystemExit("--k must be >= 1")

    # Condition values (metrics) for node diffusion: cond_cols
    cond_vals = torch.tensor([[float(cond_dict[c]) for c in node_bundle.cond_cols]], device=device, dtype=torch.float32)
    cond_vals = apply_log_cols_torch(cond_vals, node_bundle.cond_cols, node_bundle.log_cols)

    # Raw condition values for NPrior (if used)
    cond_vals_prior_raw = None
    if n_prior is not None:
        cond_vals_prior_raw = torch.tensor(
            [[float(cond_dict[c]) for c in n_prior.cond_cols]],
            device=device,
            dtype=torch.float32,
        )

    # Target vector for surrogate scoring (standardized, in surrogate training space)
    y_target = torch.tensor([[float(cond_dict[c]) for c in surrogate.target_cols]], device=device, dtype=torch.float32)
    y_target_t = apply_log_cols_torch(y_target, surrogate.target_cols, surrogate.log_cols)
    y_target_z = surrogate.scaler.transform_torch(y_target_t)

    results = []
    total = 0
    n_values_per_rd: dict[float | None, list[int]] = {}
    rd_for_prior_default: float
    if args.rd is not None:
        rd_for_prior_default = float(args.rd)
    elif rd_mode_s == "solve":
        rd_for_prior_default = float(np.sqrt(float(rd_min_f) * float(rd_max_f)))
    else:
        rd_cands = [float(x) for x in list(args.rd_candidates)]
        if not rd_cands:
            rd_for_prior_default = 0.1
        else:
            rd_for_prior_default = float(rd_cands[len(rd_cands) // 2])

    for rd_cond in rd_loop_values:
        if args.n_nodes is not None:
            n_values = [int(args.n_nodes)]
        elif list(args.n_candidates):
            n_values = clamp_and_unique([int(x) for x in list(args.n_candidates)], min_n=int(args.min_n), max_n=int(args.max_n))
        else:
            if n_prior is None or cond_vals_prior_raw is None:
                raise RuntimeError("Internal error: NPrior expected but not loaded")
            rd_for_prior = float(rd_cond) if rd_cond is not None else float(rd_for_prior_default)
            n_values = sample_n_candidates_from_prior(
                n_prior,
                rd=float(rd_for_prior),
                cond_vals_raw=cond_vals_prior_raw,
                n_samples=int(args.n_prior_samples),
                min_n=int(args.min_n),
                max_n=int(args.max_n),
                device=device,
            )
        if not n_values:
            raise RuntimeError("No N candidates produced")
        n_values_per_rd[rd_cond] = n_values
        total += int(len(n_values) * int(k_per_combo))

    with tqdm(
        total=int(total),
        desc="generate",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        for rd_cond in rd_loop_values:
            n_values = n_values_per_rd[rd_cond]
            for n_nodes in n_values:
                logn_t = torch.tensor([[float(np.log(float(n_nodes)))]], device=device, dtype=torch.float32)
                rd_t = None
                if bool(node_bundle.use_rd):
                    if rd_cond is None:
                        raise RuntimeError("Internal error: rd_cond is None but node diffusion requires RD")
                    rd_t = torch.tensor([[float(rd_cond)]], device=device, dtype=torch.float32)
                    full = torch.cat([rd_t, logn_t, cond_vals], dim=-1)  # (1, 2+Dc)
                else:
                    full = torch.cat([logn_t, cond_vals], dim=-1)  # (1, 1+Dc)
                cond_z = node_bundle.cond_scaler.transform_torch(full).squeeze(0)

                for _ in range(int(k_per_combo)):
                    rd_desc = f"{float(rd_cond):.4g}" if rd_cond is not None else "NA"
                    pbar.set_postfix_str(f"step=ddpm rd_cond={rd_desc} N={int(n_nodes)}", refresh=True)
                    t_ddpm0 = time.perf_counter()
                    coords01 = ddpm_sample_coords(
                        schedule=node_bundle.schedule,
                        denoiser=node_bundle.denoiser,
                        cond_z=cond_z,
                        n_nodes=int(n_nodes),
                        k_nn=node_bundle.k_nn,
                        device=device,
                    )
                    t_ddpm = float(time.perf_counter() - t_ddpm0)

                    pbar.set_postfix_str(f"step=edge rd_cond={rd_desc} N={int(n_nodes)} ddpm={t_ddpm:.3g}s", refresh=True)
                    t_edge0 = time.perf_counter()
                    edges_uv = sample_edges_from_coords(
                        edge_bundle=edge_bundle,
                        coords01=coords01,
                        deg_cap=int(args.deg_cap),
                        edge_thr=float(args.edge_thr),
                        ensure_connected=True,
                        device=device,
                    )
                    t_edge = float(time.perf_counter() - t_edge0)

                    edge_index = undirected_to_directed_edge_index(edges_uv).to(device)

                    pbar.set_postfix_str(
                        f"step=surrogate rd_cond={rd_desc} N={int(n_nodes)} ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s",
                        refresh=True,
                    )
                    t_sur0 = time.perf_counter()
                    rd_used = None
                    rd_solved = None
                    rd_hit_bound = None
                    if rd_mode_s == "fixed":
                        if rd_cond is None:
                            raise RuntimeError("Internal error: rd_cond is None but rd_mode=fixed")
                        rd_used = float(rd_cond)
                        rd_t_score = torch.tensor([[float(rd_used)]], device=device, dtype=torch.float32)
                        batch = Batch(
                            x=coords01,
                            edge_index=edge_index,
                            batch=torch.zeros((int(n_nodes),), device=device, dtype=torch.long),
                            rd=rd_t_score,
                            y=y_target_z,
                            n_nodes=torch.tensor([int(n_nodes)], device=device, dtype=torch.long),
                            n_edges=torch.tensor([int(edges_uv.shape[0])], device=device, dtype=torch.long),
                        )
                        pred_z = surrogate.model(batch)
                        err = torch.mean((pred_z - y_target_z) ** 2).item()
                        pred_t = surrogate.scaler.inverse_transform_torch(pred_z)
                        pred = (
                            invert_log_cols_torch(pred_t, surrogate.target_cols, surrogate.log_cols)
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        rd_final = float(rd_used)
                    else:
                        solved = solve_rd_for_target(
                            surrogate=surrogate,
                            x=coords01,
                            edge_index=edge_index,
                            n_nodes=int(n_nodes),
                            n_edges=int(edges_uv.shape[0]),
                            y_target_z=y_target_z,
                            rd_min=float(rd_min_f),
                            rd_max=float(rd_max_f),
                            grid_steps=int(args.rd_grid_steps),
                            refine_iters=int(args.rd_refine_iters),
                            device=device,
                        )
                        err = float(solved["err_best"])
                        pred = np.asarray(solved["pred_vec_best"], dtype=np.float32)
                        rd_solved = float(solved["rd_best"])
                        rd_hit_bound = bool(solved["hit_bound"])
                        rd_final = float(rd_solved)
                    t_sur = float(time.perf_counter() - t_sur0)

                    results.append(
                        {
                            "err": float(err),
                            "rd": float(rd_final),
                            "rd_cond": float(rd_cond) if rd_cond is not None else None,
                            "rd_used": float(rd_used) if rd_used is not None else None,
                            "rd_solved": float(rd_solved) if rd_solved is not None else None,
                            "rd_hit_bound": bool(rd_hit_bound) if rd_hit_bound is not None else None,
                            "n_nodes": int(n_nodes),
                            "n_edges": int(edges_uv.shape[0]),
                            "coords01": coords01.detach().cpu().numpy(),
                            "edges_uv": edges_uv,
                            "pred": {c: float(pred[i]) for i, c in enumerate(surrogate.target_cols)},
                        }
                    )

                    pbar.set_postfix_str(
                        f"step=done rd={rd_final:.4g} N={int(n_nodes)} ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s sur={t_sur:.3g}s err_mse_z={err:.3g}",
                        refresh=True,
                    )
                    pbar.update(1)

    results.sort(key=lambda d: d["err"])
    keep = results[: int(args.top_m)]

    for i, r in enumerate(keep, start=1):
        coords01 = r["coords01"]
        coords = coords01 * COORD_RANGE + COORD_MIN
        node_path = run_dir / f"Node_gen_{i}.txt"
        conn_path = run_dir / f"Connection_gen_{i}.txt"
        meta_path = run_dir / f"meta_{i}.json"
        fig_png = run_dir / "figures" / f"graph_gen_{i}.png"
        fig_svg = run_dir / "figures" / f"graph_gen_{i}.svg"

        with open(node_path, "w") as f:
            for nid, (x, y) in enumerate(coords, start=1):
                f.write(f"{nid}, {x:.6e}, {y:.6e}\n")

        with open(conn_path, "w") as f:
            for eid, (u, v) in enumerate(r["edges_uv"].tolist(), start=1):
                f.write(f"{eid}, {int(u)+1}, {int(v)+1}\n")

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "rank": i,
                    "err_mse_z": r["err"],
                    "rd": float(r["rd"]),
                    "rd_mode": str(rd_mode_s),
                    "rd_cond": r.get("rd_cond"),
                    "rd_used": r.get("rd_used"),
                    "rd_solved": r.get("rd_solved"),
                    "rd_hit_bound": r.get("rd_hit_bound"),
                    "cond": cond_dict,
                    "n_nodes": r["n_nodes"],
                    "n_edges": r["n_edges"],
                    "deg_cap": int(args.deg_cap),
                    "edge_thr": float(args.edge_thr),
                    "edge_k": int(edge_bundle.k),
                    "pred": r["pred"],
                },
                f,
                indent=2,
            )

        save_graph_figure(
            coords=coords,
            edges_uv=r["edges_uv"],
            out_png=str(fig_png),
            out_svg=str(fig_svg),
            title=(
                f"gen {i} | rd={r['rd']:.4g} N={r['n_nodes']} E={r['n_edges']} err_mse_z={r['err']:.3g}\n"
                f"RS_true={cond_dict.get('RS', float('nan')):.4g} RS_pred={r['pred'].get('RS', float('nan')):.4g}"
            ),
        )

    best_rd = keep[0]["rd"] if keep else None
    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(
        json.dumps(
            {
                "saved": len(keep),
                "base_out_dir": str(base_dir),
                "run_dir": str(run_dir),
                "best_err_mse_z": keep[0]["err"] if keep else None,
                "best_rd": float(best_rd) if best_rd is not None else None,
                "rd_mode": str(rd_mode_s),
                "rd_fixed": float(args.rd) if args.rd is not None else None,
                "rd_search": (rd_mode_s == "fixed" and args.rd is None) or (rd_mode_s == "solve"),
                "rd_candidates": [float(x) for x in list(args.rd_candidates)] if (args.rd is None and list(args.rd_candidates)) else None,
                "rd_min": float(rd_min_f) if rd_mode_s == "solve" else None,
                "rd_max": float(rd_max_f) if rd_mode_s == "solve" else None,
                "rd_grid_steps": int(args.rd_grid_steps) if rd_mode_s == "solve" else None,
                "rd_refine_iters": int(args.rd_refine_iters) if rd_mode_s == "solve" else None,
                "k_per_combo": int(k_per_combo),
                "deg_cap": int(args.deg_cap),
                "edge_thr": float(args.edge_thr),
                "n_nodes": int(args.n_nodes) if args.n_nodes is not None else None,
                "n_candidates": [int(x) for x in list(args.n_candidates)] if list(args.n_candidates) else None,
                "n_prior_ckpt": str(args.n_prior_ckpt) if str(args.n_prior_ckpt) else None,
                "n_prior_samples": int(args.n_prior_samples) if str(args.n_prior_ckpt) else None,
            }
        )
    )


if __name__ == "__main__":
    main()
