from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.corruptions import CORRUPTIONS, Corruption
from src.data import OfficeHomeDataset, build_splits, make_loader
from src.eval import predict
from src.metrics import classification_metrics
from src.models import build_transforms, create_classifier
from src.train_config import batch_size, feature_pool
from src.utils import device, load_checkpoint, load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="target_test")
    p.add_argument("--corruptions", nargs="+")
    p.add_argument("--severities", nargs="+", type=int)
    p.add_argument("--summary-name", default="ACS")
    p.add_argument("--out")
    args = p.parse_args()
    dev = device()
    ckpt = load_checkpoint(args.checkpoint, map_location=dev)
    if "cfg" not in ckpt and not args.config:
        raise ValueError("Checkpoint has no cfg; pass --config for legacy checkpoints")
    cfg = ckpt.get("cfg") or load_config(args.config)
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        raise KeyError("Checkpoint does not contain class_to_idx")
    ckpt_model = ckpt.get("meta", {}).get("model")
    model_name = ckpt_model or args.model
    if ckpt_model and args.model and args.model != ckpt_model:
        raise ValueError(f"--model={args.model} does not match checkpoint model={ckpt_model}")
    splits, split_class_to_idx = build_splits(cfg, int(ckpt.get("meta", {}).get("seed", args.seed)))
    if split_class_to_idx != class_to_idx:
        raise ValueError("class_to_idx from checkpoint does not match split config")
    mode = ckpt.get("meta", {}).get("mode", "linear_probe")
    model = create_classifier(model_name, len(class_to_idx), pretrained=False, feature_pool=feature_pool(cfg, model_name, mode)).to(dev)
    model.load_state_dict(ckpt["model"])
    tf = build_transforms(model, False, cfg.get("data", {}).get("image_size"))
    batch = batch_size(cfg, mode)
    rows = []
    corruptions = args.corruptions or CORRUPTIONS
    severities = args.severities or list(range(1, 6))
    unknown = sorted(set(corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruptions: {unknown}")
    for name in corruptions:
        for severity in severities:
            ds = OfficeHomeDataset(splits[args.split], class_to_idx, tf, Corruption(name, severity, args.seed))
            loader = make_loader(ds, batch, False, cfg["data"]["workers"])
            logits, labels, _, _ = predict(model, loader, dev)
            rows.append({"corruption": name, "severity": severity, **classification_metrics(labels, logits)})
    clean_ds = OfficeHomeDataset(splits[args.split], class_to_idx, tf)
    clean_loader = make_loader(clean_ds, batch, False, cfg["data"]["workers"])
    clean_logits, clean_labels, _, _ = predict(model, clean_loader, dev)
    clean_f1 = classification_metrics(clean_labels, clean_logits)["macro_f1"]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if isinstance(rows[0][k], float)}
    summary = {
        "corruption": args.summary_name,
        "severity": 0,
        **mean,
        "relative_corruption_drop": (clean_f1 - mean["macro_f1"]) / clean_f1 if clean_f1 else 0.0,
    }
    out = Path(args.out) if args.out else Path("outputs/metrics") / Path(args.checkpoint).parent.name / "corruptions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerows(rows + [summary])


if __name__ == "__main__":
    main()
