from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch


def _to_float(x: Any) -> float | None:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, (float, int)):
        return float(x)
    return None


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps":
        torch.mps.empty_cache()
        return
    if device.type == "cpu":
        return
    raise RuntimeError(f"Unsupported device type for cache clearing: {device.type}")


def _device_memory_mb(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        if device.index is None:
            idx = torch.cuda.current_device()
        else:
            idx = int(device.index)
        return {
            "cuda_alloc_mb": float(torch.cuda.memory_allocated(idx)) / (1024.0**2),
            "cuda_reserved_mb": float(torch.cuda.memory_reserved(idx)) / (1024.0**2),
        }
    if device.type == "mps":
        return {
            "mps_alloc_mb": float(torch.mps.current_allocated_memory()) / (1024.0**2),
            "mps_driver_mb": float(torch.mps.driver_allocated_memory()) / (1024.0**2),
        }
    if device.type == "cpu":
        return {}
    raise RuntimeError(f"Unsupported device type for memory stats: {device.type}")


class EmptyCacheCallback(pl.Callback):
    """
    Clears the backend device cache to mitigate runaway reserved memory (especially on MPS).
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        _empty_device_cache(pl_module.device)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        _empty_device_cache(pl_module.device)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        _empty_device_cache(pl_module.device)


class DeviceMemoryCallback(pl.Callback):
    """
    Logs backend memory usage to the progress bar every N steps.
    """

    def __init__(self, *, every_n_steps: int = 50, prog_bar: bool = False):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be > 0")
        self.prog_bar = bool(prog_bar)

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking:
            return
        step = int(trainer.global_step)
        if (step + 1) % self.every_n_steps != 0:
            return
        stats = _device_memory_mb(pl_module.device)
        for k, v in stats.items():
            pl_module.log(f"mem/{k}", float(v), on_step=True, on_epoch=False, prog_bar=self.prog_bar, logger=False, batch_size=1)


class JsonlMetricsCallback(pl.Callback):
    """
    Writes a compact `history.jsonl` with epoch-level metrics from `trainer.callback_metrics`.
    """

    def __init__(self, out_path: str):
        super().__init__()
        self.out_path = out_path
        self._first_write = True
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch) + 1
        row: dict[str, Any] = {"epoch": epoch}
        for k, v in metrics.items():
            fk = str(k)
            fv = _to_float(v)
            if fv is not None:
                if fk.startswith("train/") or fk.startswith("val/"):
                    row[fk] = fv

        if not any(k.startswith("train/") for k in row):
            raise RuntimeError(f"Missing train/* metrics for epoch={epoch}. Ensure training logs on_epoch metrics.")
        if not any(k.startswith("val/") for k in row):
            raise RuntimeError(f"Missing val/* metrics for epoch={epoch}. Ensure validation logs on_epoch metrics.")

        mode = "w" if self._first_write else "a"
        self._first_write = False
        with open(self.out_path, mode) as f:
            f.write(json.dumps(row) + "\n")
