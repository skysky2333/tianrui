from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tessgen.utils import device_from_arg, set_seed

from ..checkpoint import load_lit, torch_load
from ..data import write_connection_txt, write_seed_txt
from ..graph_ops import sample_empirical_n
from ..viz import save_graph_grid


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Seed/Connection graph pairs from a trained modular graph generator")
    p.add_argument("--ckpt", type=str, default="runs/graph_gen/gan_model.pt")
    p.add_argument("--out_dir", type=str, default="out/gan_generated")
    p.add_argument("--n_samples", type=int, default=12)
    p.add_argument("--n_nodes", type=int, default=0, help="0 = sample N from training distribution")
    p.add_argument("--edge_threshold", type=float, default=0.5)
    p.add_argument("--sample_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))
    device = device_from_arg(args.device)
    payload = torch_load(args.ckpt, map_location=device)
    lit = load_lit(args.ckpt, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    if int(args.n_nodes) > 0:
        ns = [int(args.n_nodes)] * int(args.n_samples)
    else:
        ns = sample_empirical_n(list(map(int, payload["n_values"])), rng=rng, n_samples=int(args.n_samples))
    grid = []
    for i, n in enumerate(ns):
        if payload["approach"] == "diffusion_critic":
            xyr, edges = lit.sample_graph(n_nodes=int(n), edge_threshold=float(args.edge_threshold), sample_steps=int(args.sample_steps))
        else:
            xyr, edges = lit.sample_graph(n_nodes=int(n), edge_threshold=float(args.edge_threshold))
        xyr_np = xyr.detach().cpu().numpy()
        edges_np = edges.detach().cpu().numpy()
        write_seed_txt(out_dir / f"Seed_gen_{i}.txt", xyr_np)
        write_connection_txt(out_dir / f"Connection_gen_{i}.txt", edges_np)
        grid.append((xyr_np, edges_np, f"gen {i}"))
    save_graph_grid(graphs=grid, out_png=str(out_dir / "synthetic_grid.png"), cols=4)
    print(json.dumps({"out_dir": str(out_dir), "n_samples": len(ns), "n_nodes": ns[:20]}, indent=2))


if __name__ == "__main__":
    main()
