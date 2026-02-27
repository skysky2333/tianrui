from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..cycle_eval import run_cycle_eval
from ..ckpt import load_edge_model, load_n_prior, load_node_diffusion, load_surrogate
from ..data import discover_graph_ids, rows_for_graph_ids, train_val_test_split_graph_ids
from ..reporting import write_json
from ..utils import device_from_arg, ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end benchmark: metrics -> (diffusion+edge) -> surrogate -> metrics")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_rows", type=int, default=0, help="0 = all test rows")

    p.add_argument("--surrogate_ckpt", type=str, required=True)
    p.add_argument("--node_ckpt", type=str, required=True)
    p.add_argument("--edge_ckpt", type=str, required=True)

    p.add_argument("--k_best", type=int, default=8, help="How many samples per row for best-of-k evaluation")
    p.add_argument("--deg_cap", type=int, default=12)
    p.add_argument("--edge_thr", type=float, default=0.5, help="Edge probability threshold used during edge sampling")
    p.add_argument("--min_n", type=int, default=64)
    p.add_argument("--max_n", type=int, default=5000)
    p.add_argument("--n_mode", type=str, default="true", help="N selection: true|fixed|candidates|prior")
    p.add_argument("--n_fixed", type=int, default=0)
    p.add_argument("--n_candidates", type=int, nargs="*", default=[])
    p.add_argument("--n_prior_ckpt", type=str, default="")
    p.add_argument("--n_prior_samples", type=int, default=12)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_row_figs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_graph_files", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dir(args.out_dir)
    set_seed(int(args.seed))

    if not (0.0 <= float(args.edge_thr) <= 1.0):
        raise SystemExit(f"--edge_thr must be in [0,1]; got {args.edge_thr}")

    device = device_from_arg(args.device)
    surrogate = load_surrogate(args.surrogate_ckpt, device=device)
    node_bundle = load_node_diffusion(args.node_ckpt, device=device)
    edge_bundle = load_edge_model(args.edge_ckpt, device=device)
    n_prior = None
    if str(args.n_mode) == "prior":
        if not str(args.n_prior_ckpt):
            raise SystemExit("For n_mode=prior, provide --n_prior_ckpt.")
        n_prior = load_n_prior(args.n_prior_ckpt, device=device)

    df = pd.read_csv(args.data_csv)
    graph_ids = discover_graph_ids(args.tess_root)
    _, _, test_g = train_val_test_split_graph_ids(
        graph_ids,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    test_rows = rows_for_graph_ids(len(df), test_g)
    if int(args.max_rows) > 0:
        test_rows = test_rows[: int(args.max_rows)]
    if not test_rows:
        raise SystemExit("No test rows selected.")

    cycle = run_cycle_eval(
        df=df,
        row_indices=[int(x) for x in test_rows],
        tess_root=args.tess_root,
        surrogate=surrogate,
        node_bundle=node_bundle,
        edge_bundle=edge_bundle,
        n_prior=n_prior,
        device=device,
        k_best=int(args.k_best),
        deg_cap=int(args.deg_cap),
        min_n=int(args.min_n),
        max_n=int(args.max_n),
        edge_thr=float(args.edge_thr),
        n_mode=str(args.n_mode),
        n_fixed=int(args.n_fixed),
        n_candidates=[int(x) for x in list(args.n_candidates)],
        n_prior_samples=int(args.n_prior_samples),
        out_dir=args.out_dir,
        save_row_figs=bool(args.save_row_figs),
        save_graph_files=bool(args.save_graph_files),
        progress_prefix="cycle/test",
    )

    report = {
        "task": "benchmark_cycle",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac), "seed": int(args.seed)},
        "ckpts": {"surrogate": args.surrogate_ckpt, "node_diffusion": args.node_ckpt, "edge": args.edge_ckpt},
        "device": str(device),
        "cycle": cycle,
    }
    write_json(str(Path(args.out_dir) / "report.json"), report)
    print(json.dumps({"saved": args.out_dir, "test_rows": int(len(test_rows)), "elapsed_sec": float(cycle["elapsed_sec"])}))


if __name__ == "__main__":
    main()
