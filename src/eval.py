from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
from tqdm import tqdm

from .utils import amp_dtype


@torch.inference_mode()
def predict(model: torch.nn.Module, loader, device: torch.device, desc: str = "eval", use_amp: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels, domains, paths = [], [], [], []
    for x, y, d, p in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype(), enabled=use_amp and device.type == "cuda"):
            out = model(x).float().cpu().numpy()
        logits.append(out)
        labels.append(y.numpy())
        domains.extend(d)
        paths.extend(p)
    return np.concatenate(logits), np.concatenate(labels), np.array(domains), np.array(paths)


def save_predictions(path: str | Path, logits: np.ndarray, labels: np.ndarray, domains: np.ndarray, paths: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), logits=logits, labels=labels, domains=domains, paths=paths)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "domain", "label", "prediction"])
        writer.writeheader()
        for row in zip(paths, domains, labels, logits.argmax(axis=1)):
            writer.writerow(dict(zip(writer.fieldnames, row)))
