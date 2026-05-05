from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .data import OfficeHomeDataset, make_loader
from .eval import save_predictions
from .metrics import classification_metrics, domain_metrics
from .models import build_transforms, profile_model
from .train_config import accumulation, checkpoint, metadata
from .train_console import image_size, print_epoch, print_run_footer, write_history
from .train_optim import scheduler
from .utils import amp_dtype, load_checkpoint, save_json


def run_cached_linear_probe(
    cfg: dict,
    model: torch.nn.Module,
    loaders: dict,
    splits: dict,
    class_to_idx: dict,
    dirs: dict[str, Path],
    name: str,
    model_name: str,
    mode: str,
    seed: int,
    tag: str | None,
    dev: torch.device,
    started: float,
    batch_size: int,
) -> dict[str, float]:
    print("linear_probe: caching frozen backbone embeddings")
    eval_tf = build_transforms(model, False, cfg.get("data", {}).get("image_size"))
    feature_loaders = {
        "train": make_loader(OfficeHomeDataset(splits["train"], class_to_idx, eval_tf), batch_size, False, cfg["data"]["workers"]),
        "val": make_loader(OfficeHomeDataset(splits["val"], class_to_idx, eval_tf), batch_size, False, cfg["data"]["workers"]),
    }
    for split in [s for s in ("id_test", "target_test", "unknown_test") if s in splits]:
        feature_loaders[split] = make_loader(OfficeHomeDataset(splits[split], class_to_idx, eval_tf), batch_size, False, cfg["data"]["workers"])
    features = {
        split: cached_features(model, loader, dev, dirs["predictions"] / f"embeddings_{split}.npz", split, cfg["train"].get("amp", True))
        for split, loader in feature_loaders.items()
    }
    head = nn.Linear(features["train"][0].shape[1], len(class_to_idx)).to(dev)
    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    epochs = int(cfg["train"]["epochs"])
    accum_steps, _ = accumulation(cfg, mode, batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = scheduler(opt, cfg, mode, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_head_epoch(head, features["train"][0], features["train"][1], criterion, opt, dev, batch_size, accum_steps)
        val_logits = head_logits(head, features["val"][0], dev, batch_size)
        val = classification_metrics(features["val"][1], val_logits)
        sched.step()
        lr = max(sched.get_last_lr())
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        write_history(dirs["logs"] / "history.csv", history)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            load_head_into_model(model, head)
            ckpt = checkpoint(model, class_to_idx, cfg, model_name, mode, seed, tag)
            ckpt["cached_embeddings"] = True
            torch.save(ckpt, dirs["checkpoints"] / "best.pt")
        else:
            bad_epochs += 1
        print_epoch(epoch, epochs, train_loss, val, lr, best_f1, best_epoch, bad_epochs, cfg["train"]["patience"], time.perf_counter() - epoch_started, improved)
        if bad_epochs >= cfg["train"]["patience"]:
            print(f"early stopping: no val Macro-F1 improvement for {bad_epochs} epochs")
            break
    ckpt = load_checkpoint(dirs["checkpoints"] / "best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    head = model.get_classifier() if hasattr(model, "get_classifier") else head
    report = {**metadata(cfg, model_name, mode, seed, tag), "best_val_macro_f1": best_f1}
    for split in [s for s in ("id_test", "target_test", "unknown_test") if s in features]:
        x, labels, domains, paths = features[split]
        logits = head_logits(head, x, dev, batch_size)
        save_predictions(dirs["predictions"] / split, logits, labels, domains, paths)
        if split == "unknown_test":
            continue
        metrics = classification_metrics(labels, logits)
        if split == "target_test":
            metrics |= domain_metrics(labels, logits, domains, report.get("id_test_macro_f1"))
        report |= {f"{split}_{k}": v for k, v in metrics.items()}
    if cfg["train"].get("profile", True):
        report |= {f"compute_{k}": v for k, v in profile_model(model, image_size(next(iter(loaders["val"]))[0]), dev).items()}
    save_json(report, dirs["metrics"] / "metrics.json")
    print_run_footer(name, report, time.perf_counter() - started)
    return report


@torch.inference_mode()
def cached_features(model: torch.nn.Module, loader, dev: torch.device, path: Path, desc: str, use_amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path, allow_pickle=True)
        print(f"embeddings cache hit: {desc} -> {path}")
        return data["features"], data["labels"], data["domains"], data["paths"]
    model.eval()
    xs, ys, domains, paths = [], [], [], []
    for x, y, d, p in tqdm(loader, desc=f"embed_{desc}", leave=False):
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype(), enabled=use_amp and dev.type == "cuda"):
            feats = model.forward_head(model.forward_features(x), pre_logits=True)
        xs.append(feats.float().cpu().numpy())
        ys.append(y.numpy())
        domains.extend(d)
        paths.extend(p)
    out = (np.concatenate(xs), np.concatenate(ys), np.array(domains), np.array(paths))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=out[0], labels=out[1], domains=out[2], paths=out[3])
    print(f"embeddings cached: {desc} {out[0].shape} -> {path}")
    return out


def train_head_epoch(head: nn.Module, features: np.ndarray, labels: np.ndarray, criterion, opt, dev: torch.device, batch_size: int, accum_steps: int) -> float:
    head.train()
    order = np.random.permutation(len(labels))
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    starts = list(range(0, len(order), batch_size))
    for step, start in enumerate(starts, 1):
        idx = order[start : start + batch_size]
        x = torch.from_numpy(features[idx]).to(dev)
        y = torch.from_numpy(labels[idx]).long().to(dev)
        loss = criterion(head(x), y)
        (loss / _window_steps(step, len(starts), accum_steps)).backward()
        if step % accum_steps == 0 or step == len(starts):
            opt.step()
            opt.zero_grad(set_to_none=True)
        total += float(loss.detach()) * len(idx)
        n += len(idx)
    return total / max(n, 1)


def _window_steps(step: int, total_steps: int, accum_steps: int) -> int:
    window_start = ((step - 1) // accum_steps) * accum_steps + 1
    return min(accum_steps, total_steps - window_start + 1)


@torch.inference_mode()
def head_logits(head: nn.Module, features: np.ndarray, dev: torch.device, batch_size: int) -> np.ndarray:
    head.eval()
    logits = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start : start + batch_size]).to(dev)
        logits.append(head(x).float().cpu().numpy())
    return np.concatenate(logits)


def load_head_into_model(model: torch.nn.Module, head: nn.Module) -> None:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if not isinstance(classifier, nn.Module):
        raise RuntimeError("Model does not expose a classifier module for cached linear probing")
    classifier.load_state_dict(head.state_dict())
