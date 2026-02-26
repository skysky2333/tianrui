from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..constants import RD_LEVELS
from ..data import GraphStore, discover_graph_ids
from ..graph_utils import is_connected


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check dataset structure and basic stats")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--n_graphs", type=int, default=50, help="How many graphs to inspect (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data_csv)
    if len(df) % 5 != 0:
        raise SystemExit(f"CSV rows must be multiple of 5; got {len(df)}")

    n_graphs_csv = len(df) // 5
    ids = discover_graph_ids(args.tess_root)
    if ids and max(ids) != n_graphs_csv:
        print(json.dumps({"warning": "max graph id != n_graphs_csv", "max_id": max(ids), "n_graphs_csv": n_graphs_csv}))

    # RD pattern check
    bad = 0
    for g in range(1, min(n_graphs_csv, 50) + 1):
        rds = df.iloc[(g - 1) * 5 : (g - 1) * 5 + 5]["RD"].tolist()
        if tuple(rds) != RD_LEVELS:
            bad += 1
    print(json.dumps({"csv_rows": len(df), "csv_graphs": n_graphs_csv, "rd_pattern_bad_first50": bad}))

    rng = np.random.default_rng(args.seed)
    inspect = ids if args.n_graphs == 0 else rng.choice(ids, size=min(int(args.n_graphs), len(ids)), replace=False).tolist()

    store = GraphStore(args.tess_root)
    node_counts = []
    edge_counts = []
    max_degs = []
    connected_frac = []
    coord_min = np.array([1e9, 1e9], dtype=np.float32)
    coord_max = np.array([-1e9, -1e9], dtype=np.float32)

    for gid in inspect:
        g = store.get(int(gid))
        n = g.n_nodes
        e = g.n_edges
        node_counts.append(n)
        edge_counts.append(e)
        coord_min = np.minimum(coord_min, g.coords01.min(dim=0).values.numpy())
        coord_max = np.maximum(coord_max, g.coords01.max(dim=0).values.numpy())

        deg = np.zeros(n, dtype=np.int64)
        for u, v in g.edges_undirected.numpy().tolist():
            deg[int(u)] += 1
            deg[int(v)] += 1
        max_degs.append(int(deg.max()) if n > 0 else 0)
        connected_frac.append(1.0 if is_connected(n, g.edges_undirected.numpy()) else 0.0)

    out = {
        "inspect_graphs": len(inspect),
        "nodes_min_mean_max": [int(np.min(node_counts)), float(np.mean(node_counts)), int(np.max(node_counts))] if node_counts else None,
        "edges_min_mean_max": [int(np.min(edge_counts)), float(np.mean(edge_counts)), int(np.max(edge_counts))] if edge_counts else None,
        "max_degree_min_mean_max": [int(np.min(max_degs)), float(np.mean(max_degs)), int(np.max(max_degs))] if max_degs else None,
        "connected_fraction": float(np.mean(connected_frac)) if connected_frac else None,
        "coord01_min": coord_min.tolist(),
        "coord01_max": coord_max.tolist(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()

