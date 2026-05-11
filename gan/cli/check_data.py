from __future__ import annotations

import argparse
import json
import statistics

from ..data import discover_graph_ids, validate_graph_pair


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Seed_*.txt + Connection_*.txt graph pairs for graph-generation training")
    p.add_argument("--data_root", type=str, default="data/data_for_gan")
    p.add_argument("--max_graphs", type=int, default=512, help="0 = validate all discovered matched graph IDs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ids = discover_graph_ids(args.data_root, max_graphs=int(args.max_graphs))
    rows = [validate_graph_pair(args.data_root, gid) for gid in ids]
    ns = [int(r["n_nodes"]) for r in rows]
    es = [int(r["n_edges"]) for r in rows]
    rs_max = [float(r["r_max"]) for r in rows]
    report = {
        "data_root": args.data_root,
        "matched_graphs_checked": len(rows),
        "n_nodes": {
            "min": min(ns) if ns else None,
            "median": statistics.median(ns) if ns else None,
            "max": max(ns) if ns else None,
        },
        "n_edges": {
            "min": min(es) if es else None,
            "median": statistics.median(es) if es else None,
            "max": max(es) if es else None,
        },
        "r_max_seen": max(rs_max) if rs_max else None,
        "first_ids": ids[:10],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
