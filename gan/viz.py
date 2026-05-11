from __future__ import annotations

from pathlib import Path

import numpy as np

from tessgen.reporting import mpl_setup


def save_graph_figure(
    *,
    xyr: np.ndarray,
    edges_uv: np.ndarray,
    out_png: str,
    title: str | None = None,
    dpi: int = 160,
    show_edges: bool = False,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if show_edges and edges_uv.size > 0:
        ax.add_collection(LineCollection(xyr[edges_uv, :2], colors="black", linewidths=0.25, alpha=0.35))
    r = xyr[:, 2].astype(np.float64, copy=False)
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    r01 = (r - r_min) / max(r_max - r_min, 1e-12)
    sizes = 8.0 + 55.0 * r01
    sc = ax.scatter(xyr[:, 0], xyr[:, 1], s=sizes, c=r, cmap="viridis", alpha=0.9, linewidths=0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(xyr[:, 0].min()) - 0.03, float(xyr[:, 0].max()) + 0.03)
    ax.set_ylim(float(xyr[:, 1].min()) - 0.03, float(xyr[:, 1].max()) + 0.03)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.01,
        0.01,
        f"N={xyr.shape[0]}  r={float(np.mean(r)):.4g}+/-{float(np.std(r)):.2g}",
        transform=ax.transAxes,
        fontsize=9,
    )
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="r")
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def save_graph_grid(
    *,
    graphs: list[tuple[np.ndarray, np.ndarray, str]],
    out_png: str,
    cols: int = 4,
    dpi: int = 160,
    show_edges: bool = False,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not graphs:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(graphs) / cols))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False, constrained_layout=True)
    all_r = np.concatenate([g[0][:, 2].astype(np.float64, copy=False) for g in graphs if g[0].shape[0] > 0])
    r_min = float(np.min(all_r))
    r_max = float(np.max(all_r))
    if r_max <= r_min:
        r_max = r_min + 1e-12
    last_sc = None
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (xyr, edges_uv, title) in zip(axes.ravel(), graphs):
        ax.set_visible(True)
        if show_edges and edges_uv.size > 0:
            ax.add_collection(LineCollection(xyr[edges_uv, :2], colors="black", linewidths=0.18, alpha=0.35))
        r = xyr[:, 2].astype(np.float64, copy=False)
        r01 = (r - r_min) / max(r_max - r_min, 1e-12)
        sizes = 5.0 + 46.0 * r01
        last_sc = ax.scatter(
            xyr[:, 0],
            xyr[:, 1],
            s=sizes,
            c=r,
            cmap="viridis",
            vmin=r_min,
            vmax=r_max,
            alpha=0.9,
            linewidths=0.0,
        )
        ax.set_aspect("equal", adjustable="box")
        pad = 0.03
        ax.set_xlim(float(xyr[:, 0].min()) - pad, float(xyr[:, 0].max()) + pad)
        ax.set_ylim(float(xyr[:, 1].min()) - pad, float(xyr[:, 1].max()) + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{title}\nN={xyr.shape[0]} r={float(np.mean(r)):.4g}+/-{float(np.std(r)):.2g}",
            fontsize=8,
        )
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="r")
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def save_score_histogram(
    *,
    out_png: str,
    real_scores: list[float],
    fake_scores: list[float],
    corrupt_scores: list[float] | None = None,
    title: str,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    bins = 40
    plt.hist(real_scores, bins=bins, alpha=0.55, label="real")
    plt.hist(fake_scores, bins=bins, alpha=0.55, label="synthetic")
    if corrupt_scores:
        plt.hist(corrupt_scores, bins=bins, alpha=0.45, label="corrupt")
    plt.title(title)
    plt.xlabel("realism probability")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
