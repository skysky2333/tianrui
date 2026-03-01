from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..ckpt import load_edge_model, load_surrogate
from ..data import GraphStore, discover_graph_ids, rows_for_graph_ids, train_val_test_split_graph_ids, undirected_to_directed_edge_index
from ..generation import sample_edges_from_coords
from ..metrics import mae, pearson_r, rmse
from ..reporting import save_histogram_both, save_scatter_plot_both, write_json, write_jsonl
from ..transforms import invert_log_cols_torch
from ..utils import Batch, device_from_arg, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate: true coords -> edge sampling -> surrogate RS (with calibration plots)")
    p.add_argument("--edge_ckpt", type=str, default="runs/edge/edge_model.pt")
    p.add_argument("--surrogate_ckpt", type=str, default="runs/surrogate/surrogate.pt")
    p.add_argument("--data_csv", type=str, default="data/Data_2.csv")
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--epoch_rows", type=int, default=50, help="-1 = all val rows")

    p.add_argument("--edge_thr", type=float, default=0.5)
    p.add_argument("--deg_cap", type=int, default=12)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--out_dir", type=str, required=True, help="Output directory (overwritten)")
    return p.parse_args()


def _compute_rs_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {
            "n": 0,
            "pearson_r": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias_mean": float("nan"),
            "bias_median": float("nan"),
            "frac_over": float("nan"),
        }
    diff = y_pred - y_true
    return {
        "n": int(y_true.size),
        "pearson_r": float(pearson_r(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "bias_mean": float(np.mean(diff)),
        "bias_median": float(np.median(diff)),
        "frac_over": float(np.mean(y_pred > y_true)),
    }


def _log10_safe(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log10(np.clip(x, float(eps), None))


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))

    if not (0.0 < float(args.val_frac) < 1.0):
        raise SystemExit(f"--val_frac must be in (0,1); got {args.val_frac}")
    if not (0.0 < float(args.test_frac) < 1.0):
        raise SystemExit(f"--test_frac must be in (0,1); got {args.test_frac}")
    if float(args.val_frac) + float(args.test_frac) >= 1.0:
        raise SystemExit("--val_frac + --test_frac must be < 1.0")
    if not (0.0 <= float(args.edge_thr) <= 1.0):
        raise SystemExit(f"--edge_thr must be in [0,1]; got {args.edge_thr}")
    if int(args.deg_cap) <= 0:
        raise SystemExit("--deg_cap must be > 0")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    figs_dir = out / "figures"

    device = device_from_arg(str(args.device))
    edge_bundle = load_edge_model(str(args.edge_ckpt), device=device)
    surrogate = load_surrogate(str(args.surrogate_ckpt), device=device)
    rs_col = "RS" if "RS" in surrogate.target_cols else surrogate.target_cols[0]
    rs_idx = int(surrogate.target_cols.index(rs_col))

    df = pd.read_csv(str(args.data_csv))
    if "RD" not in df.columns:
        raise SystemExit("CSV missing required column 'RD'")
    if rs_col not in df.columns:
        raise SystemExit(f"CSV missing required RS column {rs_col!r}")

    graph_ids = discover_graph_ids(str(args.tess_root))
    _, val_g, _ = train_val_test_split_graph_ids(
        graph_ids,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    val_rows = rows_for_graph_ids(len(df), val_g)
    if int(args.epoch_rows) != -1:
        val_rows = val_rows[: int(args.epoch_rows)]
    if not val_rows:
        raise SystemExit("No validation rows selected.")

    store = GraphStore(str(args.tess_root))

    y_true: list[float] = []
    y_pred_sampled: list[float] = []
    y_pred_true_edges: list[float] = []
    edges_cache: dict[int, np.ndarray] = {}
    row_records: list[dict] = []

    t0 = time.time()
    for row_idx in val_rows:
        row_idx = int(row_idx)
        r = df.iloc[row_idx]
        graph_id = (row_idx // 5) + 1

        rd = float(r["RD"])
        rs_true = float(r[rs_col])

        g = store.get(int(graph_id))
        coords01_cpu = g.coords01.to(dtype=torch.float32)
        n_nodes = int(coords01_cpu.shape[0])
        n_edges_true = int(g.edges_undirected.shape[0])

        if int(graph_id) not in edges_cache:
            edges_uv = sample_edges_from_coords(
                edge_bundle=edge_bundle,
                coords01=coords01_cpu,
                deg_cap=int(args.deg_cap),
                edge_thr=float(args.edge_thr),
                ensure_connected=True,
                device=device,
            )
            edges_cache[int(graph_id)] = edges_uv
        else:
            edges_uv = edges_cache[int(graph_id)]

        coords01 = coords01_cpu.to(device=device)
        edge_index = undirected_to_directed_edge_index(edges_uv).to(device)

        batch_sampled = Batch(
            x=coords01,
            edge_index=edge_index,
            batch=torch.zeros((n_nodes,), device=device, dtype=torch.long),
            rd=torch.tensor([[float(rd)]], device=device, dtype=torch.float32),
            y=torch.zeros((1, int(len(surrogate.target_cols))), device=device, dtype=torch.float32),
            n_nodes=torch.tensor([int(n_nodes)], device=device, dtype=torch.long),
            n_edges=torch.tensor([int(edges_uv.shape[0])], device=device, dtype=torch.long),
        )
        pred_z = surrogate.model(batch_sampled)
        pred_t = surrogate.scaler.inverse_transform_torch(pred_z)
        pred_raw = invert_log_cols_torch(pred_t, surrogate.target_cols, surrogate.log_cols)
        rs_pred_sampled = float(pred_raw[0, rs_idx].detach().cpu().item())

        batch_true = Batch(
            x=coords01,
            edge_index=g.edge_index.to(device=device),
            batch=torch.zeros((n_nodes,), device=device, dtype=torch.long),
            rd=torch.tensor([[float(rd)]], device=device, dtype=torch.float32),
            y=torch.zeros((1, int(len(surrogate.target_cols))), device=device, dtype=torch.float32),
            n_nodes=torch.tensor([int(n_nodes)], device=device, dtype=torch.long),
            n_edges=torch.tensor([int(n_edges_true)], device=device, dtype=torch.long),
        )
        pred_z_true = surrogate.model(batch_true)
        pred_t_true = surrogate.scaler.inverse_transform_torch(pred_z_true)
        pred_raw_true = invert_log_cols_torch(pred_t_true, surrogate.target_cols, surrogate.log_cols)
        rs_pred_true = float(pred_raw_true[0, rs_idx].detach().cpu().item())

        y_true.append(float(rs_true))
        y_pred_sampled.append(float(rs_pred_sampled))
        y_pred_true_edges.append(float(rs_pred_true))

        row_records.append(
            {
                "row_idx": int(row_idx),
                "graph_id": int(graph_id),
                "rd": float(rd),
                "rs_col": str(rs_col),
                "rs_true": float(rs_true),
                "rs_pred_sampled": float(rs_pred_sampled),
                "rs_pred_true_edges": float(rs_pred_true),
                "n_nodes": int(n_nodes),
                "n_edges_true": int(n_edges_true),
                "n_edges_sampled": int(edges_uv.shape[0]),
            }
        )

    write_jsonl(str(rows_path), row_records)

    yt = np.asarray(y_true, dtype=np.float64)
    yp_s = np.asarray(y_pred_sampled, dtype=np.float64)
    yp_t = np.asarray(y_pred_true_edges, dtype=np.float64)

    save_scatter_plot_both(
        out_png=str(figs_dir / "rs_scatter_sampled.png"),
        out_svg=str(figs_dir / "rs_scatter_sampled.svg"),
        x=yt.tolist(),
        y=yp_s.tolist(),
        title=f"{rs_col}: true vs predicted (edge-sampled)",
        xlabel=f"true {rs_col}",
        ylabel=f"pred {rs_col}",
    )
    save_scatter_plot_both(
        out_png=str(figs_dir / "rs_scatter_true_edges.png"),
        out_svg=str(figs_dir / "rs_scatter_true_edges.svg"),
        x=yt.tolist(),
        y=yp_t.tolist(),
        title=f"{rs_col}: true vs predicted (true edges baseline)",
        xlabel=f"true {rs_col}",
        ylabel=f"pred {rs_col}",
    )
    save_scatter_plot_both(
        out_png=str(figs_dir / "rs_scatter_sampled_log10.png"),
        out_svg=str(figs_dir / "rs_scatter_sampled_log10.svg"),
        x=_log10_safe(yt).tolist(),
        y=_log10_safe(yp_s).tolist(),
        title=f"log10({rs_col}): true vs predicted (edge-sampled)",
        xlabel=f"log10(true {rs_col})",
        ylabel=f"log10(pred {rs_col})",
    )
    save_scatter_plot_both(
        out_png=str(figs_dir / "rs_scatter_true_edges_log10.png"),
        out_svg=str(figs_dir / "rs_scatter_true_edges_log10.svg"),
        x=_log10_safe(yt).tolist(),
        y=_log10_safe(yp_t).tolist(),
        title=f"log10({rs_col}): true vs predicted (true edges baseline)",
        xlabel=f"log10(true {rs_col})",
        ylabel=f"log10(pred {rs_col})",
    )
    save_histogram_both(
        out_png=str(figs_dir / "rs_error_hist_sampled.png"),
        out_svg=str(figs_dir / "rs_error_hist_sampled.svg"),
        values=(yp_s - yt).tolist(),
        title=f"{rs_col} error (pred-true) (edge-sampled)",
        xlabel=f"{rs_col}_pred - {rs_col}_true",
        bins=80,
    )
    save_histogram_both(
        out_png=str(figs_dir / "rs_error_hist_true_edges.png"),
        out_svg=str(figs_dir / "rs_error_hist_true_edges.svg"),
        values=(yp_t - yt).tolist(),
        title=f"{rs_col} error (pred-true) (true edges baseline)",
        xlabel=f"{rs_col}_pred - {rs_col}_true",
        bins=80,
    )

    metrics_sampled = _compute_rs_metrics(y_true=yt, y_pred=yp_s)
    metrics_true = _compute_rs_metrics(y_true=yt, y_pred=yp_t)

    summary = {
        "task": "eval_edge_surrogate_rs",
        "rs_col": str(rs_col),
        "n_rows": int(len(val_rows)),
        "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac), "seed": int(args.seed)},
        "eval": {"epoch_rows": int(args.epoch_rows)},
        "ckpts": {"edge": str(args.edge_ckpt), "surrogate": str(args.surrogate_ckpt)},
        "edge": {
            "variant": str(edge_bundle.variant),
            "cand_mode": str(edge_bundle.cand_mode),
            "k": int(edge_bundle.k),
            "k_msg": int(edge_bundle.k_msg) if edge_bundle.k_msg is not None else None,
            "edge_thr": float(args.edge_thr),
            "deg_cap": int(args.deg_cap),
        },
        "metrics": {"sampled": metrics_sampled, "true_edges": metrics_true},
        "artifacts": {"rows_jsonl": str(rows_path), "figures_dir": str(figs_dir)},
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(str(out / "surrogate_rs_summary.json"), summary)
    print(json.dumps({"out_dir": str(out), "pearson_r": float(metrics_sampled["pearson_r"]), "elapsed_sec": float(summary["elapsed_sec"])}))


if __name__ == "__main__":
    main()
