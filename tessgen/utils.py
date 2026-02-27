from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but torch.cuda.is_available() is False")
    if dev.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested MPS device but torch.backends.mps.is_available() is False")
    return dev


@dataclass
class Batch:
    x: torch.Tensor  # (N, 2) coordinates
    edge_index: torch.Tensor  # (2, E) edges, 0-based, directed (both directions present)
    batch: torch.Tensor  # (N,) graph id per node
    rd: torch.Tensor  # (B, 1)
    y: torch.Tensor  # (B, Dy)
    n_nodes: torch.Tensor  # (B,)
    n_edges: torch.Tensor  # (B,)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_progress_bar(*, done: int, total: int, start_time: float, prefix: str = "", width: int = 28) -> str:
    done = int(done)
    total = int(total)
    width = int(width)
    if total <= 0:
        total = 1
    if width <= 0:
        raise ValueError("width must be > 0")
    frac = float(done) / float(total)
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * float(width)))
    filled = min(max(filled, 0), width)
    bar = ("=" * filled) + ("-" * (width - filled))

    elapsed = float(time.time() - float(start_time))
    rate = float(done) / elapsed if elapsed > 1e-9 else 0.0
    eta = (float(total - done) / rate) if rate > 1e-12 else float("inf")
    eta_min = eta / 60.0

    pre = f"{prefix} " if prefix else ""
    return f"{pre}[{bar}] {done}/{total} ({100.0*frac:5.1f}%) | {rate:6.2f}/s | ETA {eta_min:6.1f}m"
