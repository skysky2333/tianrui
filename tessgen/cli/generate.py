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
from ..transforms import apply_log_cols_torch, invert_log_cols_torch
from ..utils import Batch, device_from_arg, ensure_dir, set_seed
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
    p = argparse.ArgumentParser(description="Generate graphs conditioned on metrics (and optionally RD)")
    p.add_argument(
        "--rd",
        type=float,
        default=None,
        help="If provided, fix RD. If omitted, search over --rd_candidates and output the best RD in meta_*.json.",
    )
    p.add_argument(
        "--rd_candidates",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.15, 0.2],
        help="Candidate RD values used when --rd is omitted.",
    )
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
    ensure_dir(args.out_dir)
    set_seed(args.seed)
    device = device_from_arg(args.device)

    if not (0.0 <= float(args.edge_thr) <= 1.0):
        raise SystemExit(f"--edge_thr must be in [0,1]; got {args.edge_thr}")

    cond_dict = _parse_kv_list(list(args.cond))

    surrogate = load_surrogate(args.surrogate_ckpt, device=device)
    node_bundle = load_node_diffusion(args.node_ckpt, device=device)
    edge_bundle = load_edge_model(args.edge_ckpt, device=device)
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

    # RD grid (either fixed, or search over candidates)
    if args.rd is not None:
        rd_values = [float(args.rd)]
    else:
        rd_values = [float(x) for x in list(args.rd_candidates)]
        if not rd_values:
            raise SystemExit("No RD provided: pass --rd, or provide at least one value in --rd_candidates.")
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
    n_values_per_rd: dict[float, list[int]] = {}
    for rd in rd_values:
        if args.n_nodes is not None:
            n_values = [int(args.n_nodes)]
        elif list(args.n_candidates):
            n_values = clamp_and_unique([int(x) for x in list(args.n_candidates)], min_n=int(args.min_n), max_n=int(args.max_n))
        else:
            if n_prior is None or cond_vals_prior_raw is None:
                raise RuntimeError("Internal error: NPrior expected but not loaded")
            n_values = sample_n_candidates_from_prior(
                n_prior,
                rd=float(rd),
                cond_vals_raw=cond_vals_prior_raw,
                n_samples=int(args.n_prior_samples),
                min_n=int(args.min_n),
                max_n=int(args.max_n),
                device=device,
            )
        if not n_values:
            raise RuntimeError("No N candidates produced")
        n_values_per_rd[float(rd)] = n_values
        total += int(len(n_values) * int(k_per_combo))

    with tqdm(
        total=int(total),
        desc="generate",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        for rd in rd_values:
            n_values = n_values_per_rd[float(rd)]
            for n_nodes in n_values:
                rd_t = torch.tensor([[float(rd)]], device=device, dtype=torch.float32)
                logn_t = torch.tensor([[float(np.log(float(n_nodes)))]], device=device, dtype=torch.float32)
                full = torch.cat([rd_t, logn_t, cond_vals], dim=-1)  # (1, 2+Dc)
                cond_z = node_bundle.cond_scaler.transform_torch(full).squeeze(0)

                for _ in range(int(k_per_combo)):
                    pbar.set_postfix_str(f"step=ddpm rd={rd:.4g} N={int(n_nodes)}", refresh=True)
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

                    pbar.set_postfix_str(f"step=edge rd={rd:.4g} N={int(n_nodes)} ddpm={t_ddpm:.3g}s", refresh=True)
                    t_edge0 = time.perf_counter()
                    edges_uv = sample_edges_from_coords(
                        edge_model=edge_bundle.model,
                        coords01=coords01,
                        k=edge_bundle.k,
                        deg_cap=int(args.deg_cap),
                        edge_thr=float(args.edge_thr),
                        ensure_connected=True,
                        device=device,
                    )
                    t_edge = float(time.perf_counter() - t_edge0)

                    edge_index = undirected_to_directed_edge_index(edges_uv)
                    batch = Batch(
                        x=coords01,
                        edge_index=edge_index.to(device),
                        batch=torch.zeros((int(n_nodes),), device=device, dtype=torch.long),
                        rd=rd_t,
                        y=y_target_z,
                        n_nodes=torch.tensor([int(n_nodes)], device=device, dtype=torch.long),
                        n_edges=torch.tensor([int(edges_uv.shape[0])], device=device, dtype=torch.long),
                    )

                    pbar.set_postfix_str(
                        f"step=surrogate rd={rd:.4g} N={int(n_nodes)} ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s",
                        refresh=True,
                    )
                    t_sur0 = time.perf_counter()
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
                    t_sur = float(time.perf_counter() - t_sur0)

                    results.append(
                        {
                            "err": float(err),
                            "rd": float(rd),
                            "n_nodes": int(n_nodes),
                            "n_edges": int(edges_uv.shape[0]),
                            "coords01": coords01.detach().cpu().numpy(),
                            "edges_uv": edges_uv,
                            "pred": {c: float(pred[i]) for i, c in enumerate(surrogate.target_cols)},
                        }
                    )

                    pbar.set_postfix_str(
                        f"step=done rd={rd:.4g} N={int(n_nodes)} ddpm={t_ddpm:.3g}s edge={t_edge:.3g}s sur={t_sur:.3g}s err_mse_z={err:.3g}",
                        refresh=True,
                    )
                    pbar.update(1)

    results.sort(key=lambda d: d["err"])
    keep = results[: int(args.top_m)]

    for i, r in enumerate(keep, start=1):
        coords01 = r["coords01"]
        coords = coords01 * COORD_RANGE + COORD_MIN
        node_path = Path(args.out_dir) / f"Node_gen_{i}.txt"
        conn_path = Path(args.out_dir) / f"Connection_gen_{i}.txt"
        meta_path = Path(args.out_dir) / f"meta_{i}.json"
        fig_png = Path(args.out_dir) / "figures" / f"graph_gen_{i}.png"
        fig_svg = Path(args.out_dir) / "figures" / f"graph_gen_{i}.svg"

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
    print(
        json.dumps(
            {
                "saved": len(keep),
                "out_dir": args.out_dir,
                "best_err_mse_z": keep[0]["err"] if keep else None,
                "best_rd": float(best_rd) if best_rd is not None else None,
                "rd_search": args.rd is None,
                "rd_candidates": rd_values if args.rd is None else None,
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
