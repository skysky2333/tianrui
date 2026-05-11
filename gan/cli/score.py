from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tessgen.utils import device_from_arg

from ..checkpoint import load_lit
from ..data import discover_graph_ids
from ..report import score_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score real graph pairs with a trained GAN discriminator or diffusion critic")
    p.add_argument("--ckpt", type=str, default="runs/graph_gen/gan_model.pt")
    p.add_argument("--data_root", type=str, default="data/data_for_gan")
    p.add_argument("--out_csv", type=str, default="out/gan_scores.csv")
    p.add_argument("--max_graphs", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = device_from_arg(args.device)
    lit = load_lit(args.ckpt, device=device)
    ids = discover_graph_ids(args.data_root, max_graphs=int(args.max_graphs))
    rows = score_dataset(lit, data_root=args.data_root, graph_ids=ids, max_graphs=int(args.max_graphs), num_workers=0)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_id", "n_nodes", "n_edges", "realism_logit", "realism_prob"])
        writer.writeheader()
        writer.writerows(rows)
    probs = [float(r["realism_prob"]) for r in rows]
    summary = {
        "out_csv": str(out_csv),
        "n_scored": len(rows),
        "prob_min": min(probs) if probs else None,
        "prob_mean": sum(probs) / len(probs) if probs else None,
        "prob_max": max(probs) if probs else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
