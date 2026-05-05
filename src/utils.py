from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str | Path, map_location=None):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        return


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def run_id(model: str, mode: str, protocol: str, seed: int, tag: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [stamp, protocol]
    if tag:
        parts.append(tag.replace(" ", ""))
    parts += [model, mode, f"seed{seed}"]
    return "_".join(parts)


def output_dirs(root: str | Path, name: str) -> dict[str, Path]:
    root = Path(root)
    dirs = {k: root / k / name for k in ("checkpoints", "logs", "predictions", "metrics", "tables", "figures")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_config(config_path: str | Path, dst_dir: str | Path) -> None:
    shutil.copy2(config_path, Path(dst_dir) / "config_used.yaml")


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
