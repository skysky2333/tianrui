from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LightningDevice:
    accelerator: str
    devices: int | list[int]


def lightning_device_from_arg(device: str) -> LightningDevice:
    """
    Maps our `--device` arg to PyTorch Lightning accelerator/devices.

    - auto: cuda -> mps -> cpu
    - cuda / cuda:0 / cuda:1: accelerator=gpu
    - mps: accelerator=mps
    - cpu: accelerator=cpu
    """
    if device == "auto":
        if torch.cuda.is_available():
            return LightningDevice(accelerator="gpu", devices=1)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return LightningDevice(accelerator="mps", devices=1)
        return LightningDevice(accelerator="cpu", devices=1)

    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but torch.cuda.is_available() is False")
        if dev.index is not None:
            if int(dev.index) < 0 or int(dev.index) >= int(torch.cuda.device_count()):
                raise ValueError(
                    f"Requested CUDA device index {dev.index} but torch.cuda.device_count()={torch.cuda.device_count()}"
                )
            return LightningDevice(accelerator="gpu", devices=[int(dev.index)])
        return LightningDevice(accelerator="gpu", devices=1)
    if dev.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested MPS but torch.backends.mps.is_available() is False")
        return LightningDevice(accelerator="mps", devices=1)
    if dev.type == "cpu":
        return LightningDevice(accelerator="cpu", devices=1)
    raise ValueError(f"Unsupported device: {device}")
