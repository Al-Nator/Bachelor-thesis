from __future__ import annotations

import csv

import torch

from .models import trainable_parameters


def image_size(batch: torch.Tensor) -> tuple[int, int, int]:
    return tuple(batch.shape[1:])  # type: ignore[return-value]


def print_run_header(
    name: str,
    cfg: dict,
    splits: dict,
    class_to_idx: dict,
    model: torch.nn.Module,
    dev: torch.device,
    mode: str,
    batch: int,
    accum_steps: int,
    effective_batch: int,
    param_groups: int,
    feature_pool: str,
) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_parameters(model))
    split_sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    print()
    print(f"run: {name}")
    print(f"device: {dev} | classes: {len(class_to_idx)} | splits: {split_sizes}")
    print(f"model params: {total:,} | trainable: {trainable:,} ({trainable / max(total, 1):.2%})")
    print(
        "train: "
        f"epochs={cfg['train']['epochs']} patience={cfg['train']['patience']} "
        f"batch={batch} effective_batch={effective_batch} accum={accum_steps} "
        f"lr={cfg['train']['lr']} wd={cfg['train']['weight_decay']} "
        f"amp={cfg['train'].get('amp', True)}"
    )
    if feature_pool != "default":
        print(f"feature_pool: {feature_pool}")
    if mode in {"partial_finetune", "full_finetune"}:
        print(f"fine-tune schedule: warmup_epochs={cfg['train'].get('warmup_epochs', 0)} layer_decay={cfg['train'].get('layer_decay', 1.0)} param_groups={param_groups}")


def print_epoch(
    epoch: int,
    epochs: int,
    train_loss: float,
    val: dict[str, float],
    lr: float,
    best_f1: float,
    best_epoch: int,
    bad_epochs: int,
    patience: int,
    seconds: float,
    improved: bool,
) -> None:
    mark = "*" if improved else " "
    print(
        f"{mark} epoch {epoch:02d}/{epochs} "
        f"loss={train_loss:.4f} "
        f"val_f1={val['macro_f1']:.4f} "
        f"val_acc={val['accuracy']:.4f} "
        f"val_bal_acc={val['balanced_accuracy']:.4f} "
        f"val_ece={val['ece']:.4f} "
        f"lr={lr:.2e} "
        f"best_f1={best_f1:.4f}@{best_epoch} "
        f"patience={bad_epochs}/{patience} "
        f"time={seconds / 60:.1f}m"
    )


def print_run_footer(name: str, report: dict[str, float], seconds: float) -> None:
    keys = ["best_val_macro_f1", "id_test_macro_f1", "target_test_macro_f1", "target_test_worst_domain_f1", "target_test_relative_drop"]
    summary = " ".join(f"{k}={report[k]:.4f}" for k in keys if k in report)
    print(f"done: {name} | {summary} | total={seconds / 60:.1f}m")


def write_history(path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
