from __future__ import annotations

import torch

from .models import HEAD_KEYS, trainable_parameters


def optimizer_params(model: torch.nn.Module, cfg: dict, mode: str) -> list[dict] | list[torch.nn.Parameter]:
    decay = float(cfg["train"].get("layer_decay", 1.0))
    if mode not in {"partial_finetune", "full_finetune"} or decay >= 1.0:
        return trainable_parameters(model)
    max_d = max_depth(model)
    groups: dict[float, list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = decay ** (max_d - param_depth(name, max_d))
        groups.setdefault(scale, []).append(p)
    base_lr = float(cfg["train"]["lr"])
    return [{"params": params, "lr": base_lr * scale} for scale, params in sorted(groups.items())]


def scheduler(opt, cfg: dict, mode: str, epochs: int):
    warmup = int(cfg["train"].get("warmup_epochs", 0)) if mode in {"partial_finetune", "full_finetune"} else 0
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    warmup = min(warmup, max(epochs - 1, 1))
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=float(cfg["train"].get("warmup_start_factor", 0.01)), total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1)),
        ],
        milestones=[warmup],
    )


def max_depth(model: torch.nn.Module) -> int:
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        return max_depth(backbone)
    for attr in ("blocks", "stages", "layers"):
        blocks = getattr(model, attr, None)
        if blocks is not None and hasattr(blocks, "__len__"):
            return len(blocks) + 1
    return 5 if hasattr(model, "layer4") else 1


def param_depth(name: str, max_d: int) -> int:
    name = strip_backbone_prefix(name)
    first = name.split(".", 1)[0]
    if first.lower() in HEAD_KEYS:
        return max_d
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]) + 1
    if parts[0] in {"stages", "layers"} and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]) + 1
    if first.startswith("layer") and first[5:].isdigit():
        return int(first[5:])
    return 0


def strip_backbone_prefix(name: str) -> str:
    return name[len("backbone.") :] if name.startswith("backbone.") else name
