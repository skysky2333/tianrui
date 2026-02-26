from __future__ import annotations

import os
import random
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
