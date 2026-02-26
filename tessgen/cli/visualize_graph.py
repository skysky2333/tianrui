from __future__ import annotations

import argparse
import json

from ..viz import load_graph_from_files, save_graph_figure


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a Node_*.txt + Connection_*.txt graph")
    p.add_argument("--node_path", type=str, required=True)
    p.add_argument("--conn_path", type=str, required=True)
    p.add_argument("--out_png", type=str, required=True)
    p.add_argument("--out_svg", type=str, required=True)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--node_size", type=float, default=8.0)
    p.add_argument("--edge_lw", type=float, default=0.4)
    p.add_argument("--edge_alpha", type=float, default=0.35)
    p.add_argument("--equal_axis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hide_axes", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    g = load_graph_from_files(node_path=args.node_path, conn_path=args.conn_path)
    save_graph_figure(
        coords=g.coords,
        edges_uv=g.edges_uv,
        out_png=args.out_png,
        out_svg=args.out_svg,
        title=args.title,
        dpi=int(args.dpi),
        node_size=float(args.node_size),
        edge_lw=float(args.edge_lw),
        edge_alpha=float(args.edge_alpha),
        equal_axis=bool(args.equal_axis),
        hide_axes=bool(args.hide_axes),
    )

    print(
        json.dumps(
            {
                "node_path": args.node_path,
                "conn_path": args.conn_path,
                "out_png": args.out_png,
                "out_svg": args.out_svg,
                "n_nodes": int(g.coords.shape[0]),
                "n_edges": int(g.edges_uv.shape[0]),
            }
        )
    )


if __name__ == "__main__":
    main()

