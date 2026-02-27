from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NPriorRowDataset(Dataset):
    """
    Each row in the CSV corresponds to (graph_id, RD, cond_metrics) -> log(N).

    Uses the same row->graph mapping convention:
      graph_id = row_idx//5 + 1
    """

    def __init__(
        self,
        *,
        data_csv: str,
        tess_root: str,
        cond_cols: Sequence[str],
        row_indices: Sequence[int],
    ):
        self.df = pd.read_csv(data_csv)
        self.data_csv = data_csv
        self.tess_root = Path(tess_root)
        if not self.tess_root.exists():
            raise FileNotFoundError(self.tess_root)
        self.cond_cols = list(cond_cols)
        if "RD" not in self.df.columns:
            raise ValueError("CSV missing required column RD")
        for c in self.cond_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV missing cond column {c}")
        self.row_indices = list(map(int, row_indices))

    def __len__(self) -> int:
        return len(self.row_indices)

    def graph_id_for_row(self, row_idx: int) -> int:
        return (int(row_idx) // 5) + 1

    def _node_path(self, graph_id: int) -> Path:
        return self.tess_root / f"Node_{int(graph_id)}.txt"

    @lru_cache(maxsize=4096)
    def _node_count(self, graph_id: int) -> int:
        path = self._node_path(graph_id)
        n = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    n += 1
        if n <= 0:
            raise ValueError(f"Empty node file: {path}")
        return int(n)

    def __getitem__(self, i: int) -> dict:
        row_idx = int(self.row_indices[i])
        r = self.df.iloc[row_idx]
        graph_id = self.graph_id_for_row(row_idx)

        n = self._node_count(graph_id)
        logn = float(math.log(float(n)))
        rd = float(r["RD"])
        cond = np.array([float(r[c]) for c in self.cond_cols], dtype=np.float32)

        return {
            "row_idx": int(row_idx),
            "graph_id": int(graph_id),
            "rd": torch.tensor([rd], dtype=torch.float32),
            "cond": torch.from_numpy(cond),
            "logn": torch.tensor([logn], dtype=torch.float32),
            "n_nodes": torch.tensor([int(n)], dtype=torch.long),
        }

