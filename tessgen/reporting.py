from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mpl_setup() -> None:
    """
    Ensure matplotlib runs headless (safe in CLI/servers).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)


def save_line_plot(
    *,
    out_path: str,
    x: list[float] | list[int],
    ys: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-3,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    if y_scale not in {"linear", "log", "symlog"}:
        raise ValueError(f"Unsupported y_scale={y_scale!r} (expected 'linear', 'log', or 'symlog')")
    if y_scale == "log":
        for name, y in ys.items():
            if not y:
                continue
            if min(y) <= 0.0:
                raise ValueError(f"Log-scale requested but series {name!r} has non-positive values (min={min(y)})")
    if y_scale == "symlog" and float(symlog_linthresh) <= 0.0:
        raise ValueError("symlog_linthresh must be > 0")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, y in ys.items():
        if not y:
            continue
        plt.plot(x[: len(y)], y, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_scale == "symlog":
        plt.yscale("symlog", linthresh=float(symlog_linthresh))
    else:
        plt.yscale(y_scale)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_scatter_plot(
    *,
    out_path: str,
    x: list[float],
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(x, y, s=6, alpha=0.35)
    mn = min(min(x), min(y)) if x and y else 0.0
    mx = max(max(x), max(y)) if x and y else 1.0
    plt.plot([mn, mx], [mn, mx], color="black", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_scatter_plot_both(
    *,
    out_png: str,
    out_svg: str,
    x: list[float],
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int = 160,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(x, y, s=6, alpha=0.35)
    mn = min(min(x), min(y)) if x and y else 0.0
    mx = max(max(x), max(y)) if x and y else 1.0
    plt.plot([mn, mx], [mn, mx], color="black", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi))
    plt.savefig(out_svg)
    plt.close()


def save_histogram(
    *,
    out_path: str,
    values: list[float],
    title: str,
    xlabel: str,
    bins: int = 60,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_histogram_both(
    *,
    out_png: str,
    out_svg: str,
    values: list[float],
    title: str,
    xlabel: str,
    bins: int = 60,
    dpi: int = 160,
) -> None:
    mpl_setup()
    import matplotlib.pyplot as plt

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=int(bins), alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi))
    plt.savefig(out_svg)
    plt.close()
