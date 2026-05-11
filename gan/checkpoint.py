from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models.diffusion_critic import DiffusionCriticLitModule
from .models.hybrid_gan import HybridGANLitModule


def torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_lit_from_payload(payload: dict[str, Any], *, device: str | torch.device = "cpu"):
    approach = str(payload.get("approach", ""))
    h = dict(payload["hparams"])
    if approach == "hybrid_gan":
        lit = HybridGANLitModule(
            model_cfg=dict(payload["model_cfg"]),
            k_msg=int(h["k_msg"]),
            k_edge=int(h["k_edge"]),
            candidate_mode=str(h["candidate_mode"]),
            r_scale=float(payload["r_scale"]),
            lr_g=float(h["lr_g"]),
            lr_d=float(h["lr_d"]),
            weight_decay=float(h["weight_decay"]),
            lambda_stats=float(h.get("lambda_stats", 10.0)),
            edge_temperature=float(h.get("edge_temperature", 1.0)),
            g_steps_per_batch=int(h.get("g_steps_per_batch", 1)),
            d_every_n_steps=int(h.get("d_every_n_steps", 1)),
            real_label_smooth=float(h.get("real_label_smooth", 0.9)),
            fake_label_smooth=float(h.get("fake_label_smooth", 0.0)),
            instance_noise=float(h.get("instance_noise", 0.0)),
            learn_edges=bool(h.get("learn_edges", False)),
            lambda_adv=float(h.get("lambda_adv", 1.0)),
            lambda_seed=float(h.get("lambda_seed", 50.0)),
            seed_mmd_points=int(h.get("seed_mmd_points", 256)),
            g_pretrain_steps=int(h.get("g_pretrain_steps", 5000)),
            d_loss_floor=float(h.get("d_loss_floor", 0.05)),
        )
        lit.generator.load_state_dict(payload["generator_state_dict"])
        lit.discriminator.load_state_dict(payload["discriminator_state_dict"])
    elif approach == "diffusion_critic":
        lit = DiffusionCriticLitModule(
            model_cfg=dict(payload["model_cfg"]),
            diffusion_cfg=dict(payload["diffusion_cfg"]),
            k_msg=int(h["k_msg"]),
            k_edge=int(h["k_edge"]),
            candidate_mode=str(h["candidate_mode"]),
            r_scale=float(payload["r_scale"]),
            lr_g=float(h["lr_g"]),
            lr_d=float(h["lr_d"]),
            weight_decay=float(h["weight_decay"]),
            lambda_edge=float(h.get("lambda_edge", 0.0)),
            critic_sample_steps=int(h.get("critic_sample_steps", 8)),
            learn_edges=bool(h.get("learn_edges", False)),
        )
        lit.generator.load_state_dict(payload["generator_state_dict"])
        lit.critic.load_state_dict(payload["critic_state_dict"])
    else:
        raise ValueError(f"Unsupported checkpoint approach={approach!r}")
    lit.to(device)
    lit.eval()
    return lit


def load_lit(path: str | Path, *, device: str | torch.device = "cpu"):
    return load_lit_from_payload(torch_load(path, map_location=device), device=device)


def save_approach_artifacts(payload: dict[str, Any], *, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, run_dir / "gan_model.pt")
    torch.save(
        {
            "approach": payload["approach"],
            "model_cfg": payload["model_cfg"],
            "diffusion_cfg": payload.get("diffusion_cfg"),
            "hparams": payload["hparams"],
            "r_scale": payload["r_scale"],
            "n_values": payload["n_values"],
            "state_dict": payload["generator_state_dict"],
        },
        run_dir / "generator.pt",
    )
    if payload["approach"] == "hybrid_gan":
        torch.save(
            {
                "approach": payload["approach"],
                "model_cfg": payload["model_cfg"],
                "hparams": payload["hparams"],
                "r_scale": payload["r_scale"],
                "state_dict": payload["discriminator_state_dict"],
            },
            run_dir / "discriminator.pt",
        )
    if payload["approach"] == "diffusion_critic":
        torch.save(
            {
                "approach": payload["approach"],
                "model_cfg": payload["model_cfg"],
                "hparams": payload["hparams"],
                "r_scale": payload["r_scale"],
                "state_dict": payload["critic_state_dict"],
            },
            run_dir / "critic.pt",
        )
