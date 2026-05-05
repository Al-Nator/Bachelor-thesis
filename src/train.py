from __future__ import annotations

import time

import torch
from torch import nn

from .data import OfficeHomeDataset, build_splits, make_loader
from .eval import predict, save_predictions
from .metrics import classification_metrics, domain_metrics
from .models import build_transforms, configure_trainable, create_classifier, profile_model
from .train_config import accumulation as _accumulation
from .train_config import batch_size
from .train_config import checkpoint as _checkpoint
from .train_config import feature_pool as resolve_feature_pool
from .train_config import metadata as _metadata
from .train_config import run_tag as _run_tag
from .train_console import image_size as _image_size
from .train_console import print_epoch as _print_epoch
from .train_console import print_run_footer as _print_run_footer
from .train_console import print_run_header as _print_run_header
from .train_console import write_history as _write_history
from .train_linear import run_cached_linear_probe as _run_cached_linear_probe
from .train_loops import train_epoch as _train_epoch
from .train_optim import max_depth as _max_depth
from .train_optim import optimizer_params as _optimizer_params
from .train_optim import param_depth as _param_depth
from .train_optim import scheduler as _scheduler
from .train_optim import strip_backbone_prefix as _strip_backbone_prefix
from .utils import amp_dtype, device as get_device, load_checkpoint, output_dirs, run_id, save_json, save_yaml, set_seed


def run_training(cfg: dict, model_name: str, mode: str, seed: int) -> dict[str, float]:
    started = time.perf_counter()
    set_seed(seed)
    dev = get_device()
    splits, class_to_idx = build_splits(cfg, seed)
    feature_pool = resolve_feature_pool(cfg, model_name, mode)
    model = create_classifier(model_name, len(class_to_idx), cfg["model"].get("pretrained", True), feature_pool).to(dev)
    configure_trainable(model, mode, cfg["model"].get("last_blocks", 2))
    image_size = cfg.get("data", {}).get("image_size")
    train_tf = build_transforms(model, True, image_size)
    eval_tf = build_transforms(model, False, image_size)
    batch = batch_size(cfg, mode)
    loaders = {
        "train": make_loader(OfficeHomeDataset(splits["train"], class_to_idx, train_tf), batch, True, cfg["data"]["workers"]),
        "val": make_loader(OfficeHomeDataset(splits["val"], class_to_idx, eval_tf), batch, False, cfg["data"]["workers"]),
    }
    tag = _run_tag(cfg)
    name = run_id(model_name, mode, cfg["dataset"]["protocol"], seed, tag)
    dirs = output_dirs(cfg["output_dir"], name)
    save_yaml(cfg, dirs["logs"] / "config_used.yaml")
    epochs = int(cfg["train"]["epochs"])
    accum_steps, effective_batch = _accumulation(cfg, mode, batch)
    opt = torch.optim.AdamW(_optimizer_params(model, cfg, mode), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = _scheduler(opt, cfg, mode, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and amp_dtype() == torch.float16)
    _print_run_header(name, cfg, splits, class_to_idx, model, dev, mode, batch, accum_steps, effective_batch, len(opt.param_groups), feature_pool)
    if mode == "linear_probe" and cfg["train"].get("cache_embeddings", True):
        return _run_cached_linear_probe(cfg, model, loaders, splits, class_to_idx, dirs, name, model_name, mode, seed, tag, dev, started, batch)
    best_f1 = _fit_model(cfg, model, loaders, criterion, opt, sched, scaler, dev, epochs, accum_steps, dirs, class_to_idx, model_name, mode, seed, tag)
    report = _evaluate_best(cfg, model, splits, class_to_idx, eval_tf, batch, dev, dirs, model_name, mode, seed, tag, best_f1, loaders)
    _print_run_footer(name, report, time.perf_counter() - started)
    return report


def _fit_model(cfg, model, loaders, criterion, opt, sched, scaler, dev, epochs: int, accum_steps: int, dirs, class_to_idx, model_name: str, mode: str, seed: int, tag: str | None):
    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _train_epoch(model, loaders["train"], criterion, opt, scaler, dev, cfg["train"].get("amp", True), accum_steps)
        val_logits, val_y, _, _ = predict(model, loaders["val"], dev, desc="val")
        val = classification_metrics(val_y, val_logits)
        sched.step()
        lr = max(sched.get_last_lr())
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        _write_history(dirs["logs"] / "history.csv", history)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            torch.save(_checkpoint(model, class_to_idx, cfg, model_name, mode, seed, tag), dirs["checkpoints"] / "best.pt")
        else:
            bad_epochs += 1
        _print_epoch(epoch, epochs, train_loss, val, lr, best_f1, best_epoch, bad_epochs, cfg["train"]["patience"], time.perf_counter() - epoch_started, improved)
        if bad_epochs >= cfg["train"]["patience"]:
            print(f"early stopping: no val Macro-F1 improvement for {bad_epochs} epochs")
            break
    return best_f1


def _evaluate_best(cfg, model, splits, class_to_idx, eval_tf, batch: int, dev: torch.device, dirs, model_name: str, mode: str, seed: int, tag: str | None, best_f1: float, loaders) -> dict[str, float]:
    ckpt = load_checkpoint(dirs["checkpoints"] / "best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    report = {**_metadata(cfg, model_name, mode, seed, tag), "best_val_macro_f1": best_f1}
    for split in [s for s in ("id_test", "target_test", "unknown_test") if s in splits]:
        loader = make_loader(OfficeHomeDataset(splits[split], class_to_idx, eval_tf), batch, False, cfg["data"]["workers"])
        logits, labels, domains, paths = predict(model, loader, dev, desc=split)
        save_predictions(dirs["predictions"] / split, logits, labels, domains, paths)
        if split == "unknown_test":
            continue
        metrics = classification_metrics(labels, logits)
        if split == "target_test":
            metrics |= domain_metrics(labels, logits, domains, report.get("id_test_macro_f1"))
        report |= {f"{split}_{k}": v for k, v in metrics.items()}
    if cfg["train"].get("profile", True):
        report |= {f"compute_{k}": v for k, v in profile_model(model, _image_size(next(iter(loaders["val"]))[0]), dev).items()}
    save_json(report, dirs["metrics"] / "metrics.json")
    return report
