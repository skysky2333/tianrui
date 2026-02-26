from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, *, eps: float = 1e-8) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def inverse_transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return x * std + mean

