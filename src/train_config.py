from __future__ import annotations

import torch

from .data import normalize_domain


def batch_size(cfg: dict, mode: str) -> int:
    value = cfg["train"].get("physical_batch_size", cfg["data"]["batch_size"])
    if isinstance(value, dict):
        value = value.get(mode, cfg["data"]["batch_size"])
    return int(value)


def feature_pool(cfg: dict, model_name: str, mode: str) -> str:
    value = cfg.get("model", {}).get("feature_pool", "default")
    if isinstance(value, dict):
        value = value.get(mode, "default")
    if value == "auto":
        return "cls_patch_mean" if "vit" in model_name else "default"
    if value != "default" and "vit" not in model_name:
        return "default"
    return str(value)


def accumulation(cfg: dict, mode: str, physical: int | None = None) -> tuple[int, int]:
    physical = int(physical or batch_size(cfg, mode))
    effective = cfg["train"].get("effective_batch_size", physical)
    if isinstance(effective, dict):
        effective = effective.get(mode, physical)
    target = int(effective)
    if target <= physical:
        return 1, physical
    accum = (target + physical - 1) // physical
    return accum, physical * accum


def run_tag(cfg: dict) -> str | None:
    ds = cfg["dataset"]
    protocol = ds["protocol"]
    if protocol == "lodo":
        return normalize_domain(ds["heldout_domain"])
    if protocol == "in_domain":
        return normalize_domain(ds["domain"])
    if protocol == "cross_domain":
        source = "".join(normalize_domain(d) for d in ds["source_domains"])
        target = "".join(normalize_domain(d) for d in ds["target_domains"])
        return f"{source}_to_{target}"
    if protocol == "semantic_ood":
        return "known_unknown"
    return None


def metadata(cfg: dict, model_name: str, mode: str, seed: int, tag: str | None) -> dict[str, str | int]:
    return {
        "model": model_name,
        "mode": mode,
        "protocol": cfg["dataset"]["protocol"],
        "domain_or_heldout": tag or "",
        "seed": seed,
    }


def checkpoint(model: torch.nn.Module, class_to_idx: dict[str, int], cfg: dict, model_name: str, mode: str, seed: int, tag: str | None) -> dict:
    return {
        "model": model.state_dict(),
        "class_to_idx": class_to_idx,
        "cfg": cfg,
        "meta": metadata(cfg, model_name, mode, seed, tag),
    }
