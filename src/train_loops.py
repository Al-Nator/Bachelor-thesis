from __future__ import annotations

import torch
from tqdm import tqdm

from .models import freeze_frozen_batchnorm_stats
from .utils import amp_dtype


def train_epoch(model, loader, criterion, opt, scaler, dev, use_amp: bool, accum_steps: int) -> float:
    model.train()
    freeze_frozen_batchnorm_stats(model)
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    total_steps = len(loader)
    for step, (x, y, _, _) in enumerate(tqdm(loader, desc="train", leave=False), 1):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype(), enabled=use_amp and dev.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss / _window_steps(step, total_steps, accum_steps)).backward()
        if step % accum_steps == 0 or step == total_steps:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        total += float(loss.detach()) * len(y)
        n += len(y)
    return total / max(n, 1)


def _window_steps(step: int, total_steps: int, accum_steps: int) -> int:
    window_start = ((step - 1) // accum_steps) * accum_steps + 1
    return min(accum_steps, total_steps - window_start + 1)
