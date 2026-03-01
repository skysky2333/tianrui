from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .edge_model import EdgeModel, EdgeModelConfig
from .models.edge_3 import Edge3Model, Edge3ModelConfig
from .n_prior import NPriorConfig, NPriorModel
from .node_diffusion import DiffusionConfig, DiffusionSchedule, NodeDenoiser, NodeDenoiserConfig
from .scaler import StandardScaler
from .surrogate import SurrogateConfig, SurrogateModel


@dataclass(frozen=True)
class SurrogateBundle:
    model: SurrogateModel
    target_cols: list[str]
    log_cols: set[str]
    scaler: StandardScaler


def load_surrogate(ckpt_path: str, *, device: torch.device) -> SurrogateBundle:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = SurrogateConfig(**ckpt["cfg"])
    target_cols = list(ckpt["target_cols"])
    log_cols = set(ckpt["log_cols"])
    scaler = StandardScaler(mean=np.array(ckpt["scaler_mean"], dtype=np.float32), std=np.array(ckpt["scaler_std"], dtype=np.float32))
    model = SurrogateModel(y_dim=len(target_cols), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return SurrogateBundle(model=model, target_cols=target_cols, log_cols=log_cols, scaler=scaler)


@dataclass(frozen=True)
class EdgeBundle:
    model: EdgeModel | Edge3Model
    variant: str
    cand_mode: str
    k: int
    k_msg: int | None


def load_edge_model(ckpt_path: str, *, device: torch.device) -> EdgeBundle:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    variant = str(ckpt["variant"])
    cand_mode = str(ckpt["cand_mode"])
    k = int(ckpt["k"])
    if variant == "edge":
        cfg = EdgeModelConfig(**ckpt["cfg"])
        model = EdgeModel(cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return EdgeBundle(model=model, variant=variant, cand_mode=cand_mode, k=k, k_msg=None)
    if variant == "edge_3":
        cfg = Edge3ModelConfig(**ckpt["cfg"])
        k_msg = int(ckpt["k_msg"])
        model = Edge3Model(cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return EdgeBundle(model=model, variant=variant, cand_mode=cand_mode, k=k, k_msg=k_msg)
    raise ValueError(f"Unsupported edge variant={variant!r} (expected 'edge'|'edge_3')")


@dataclass(frozen=True)
class NPriorBundle:
    model: NPriorModel
    cond_cols: list[str]
    log_cols: set[str]
    scaler: StandardScaler


def load_n_prior(ckpt_path: str, *, device: torch.device) -> NPriorBundle:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = NPriorConfig(**ckpt["cfg"])
    cond_cols = list(ckpt["cond_cols"])
    log_cols = set(ckpt["log_cols"])
    scaler = StandardScaler(mean=np.array(ckpt["scaler_mean"], dtype=np.float32), std=np.array(ckpt["scaler_std"], dtype=np.float32))
    model = NPriorModel(x_dim=1 + len(cond_cols), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return NPriorBundle(model=model, cond_cols=cond_cols, log_cols=log_cols, scaler=scaler)


@dataclass(frozen=True)
class NodeDiffusionBundle:
    denoiser: NodeDenoiser
    schedule: DiffusionSchedule
    cond_cols: list[str]
    log_cols: set[str]
    cond_scaler: StandardScaler
    k_nn: int


def load_node_diffusion(ckpt_path: str, *, device: torch.device) -> NodeDiffusionBundle:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    den_cfg = NodeDenoiserConfig(**ckpt["denoiser_cfg"])
    schedule_cfg = DiffusionConfig(**ckpt["schedule_cfg"])
    schedule = DiffusionSchedule(schedule_cfg).to(device)
    denoiser = NodeDenoiser(den_cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    denoiser.eval()
    cond_cols = list(ckpt["cond_cols"])
    log_cols = set(ckpt["log_cols"])
    cond_scaler = StandardScaler(
        mean=np.array(ckpt["cond_scaler_mean"], dtype=np.float32),
        std=np.array(ckpt["cond_scaler_std"], dtype=np.float32),
    )
    k_nn = int(ckpt["k_nn"])
    return NodeDiffusionBundle(
        denoiser=denoiser,
        schedule=schedule,
        cond_cols=cond_cols,
        log_cols=log_cols,
        cond_scaler=cond_scaler,
        k_nn=k_nn,
    )
