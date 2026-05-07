from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.corruptions import CORRUPTIONS
from src.data import build_splits, make_loader
from src.eval import predict
from src.metrics import classification_metrics
from src.models import build_transforms, create_classifier
from src.train_config import batch_size, feature_pool
from src.train_linear import head_logits
from src.utils import load_checkpoint, load_config, device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model")
    p.add_argument("--cache", default="outputs/corruption_cache/imagenet_c_s3_s5")
    p.add_argument("--severities", nargs="+", type=int, default=[3, 5])
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    p.add_argument("--split")
    p.add_argument("--out")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out) if args.out else Path("outputs/metrics") / Path(args.checkpoint).parent.name / "corruptions_s3_s5.csv"
    if out.exists() and not args.overwrite:
        print(f"skip existing: {out}")
        return
    rows, clean_f1 = evaluate(args)
    summary = summarize(rows, clean_f1, args.severities)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerows(rows + [summary])
    print(f"wrote {out}")


def evaluate(args) -> tuple[list[dict], float]:
    dev = device()
    ckpt = load_checkpoint(args.checkpoint, map_location=dev)
    cfg = ckpt.get("cfg") or load_config(args.config)
    class_to_idx = ckpt["class_to_idx"]
    meta = ckpt.get("meta", {})
    model_name = meta.get("model") or args.model
    if not model_name:
        raise ValueError("Checkpoint has no model metadata; pass --model")
    mode = meta.get("mode", "linear_probe")
    if args.model and model_name != args.model:
        raise ValueError(f"--model={args.model} does not match checkpoint model={model_name}")
    splits, split_class_to_idx = build_splits(cfg, int(meta.get("seed", 42)))
    if split_class_to_idx != class_to_idx:
        raise ValueError("class_to_idx from checkpoint does not match split config")
    split = args.split or default_split(cfg["dataset"]["protocol"])
    records = splits[split]
    clean_f1 = clean_macro_f1(args.checkpoint, split)
    model = create_classifier(model_name, len(class_to_idx), pretrained=False, feature_pool=feature_pool(cfg, model_name, mode)).to(dev)
    model.load_state_dict(ckpt["model"])
    settings = [(c, s) for c in args.corruptions for s in args.severities]
    rows = []
    for corruption, severity in tqdm(settings, desc=Path(args.checkpoint).parent.name, leave=True):
        if mode == "linear_probe":
            logits, labels = predict_from_embeddings(model, args.cache, model_name, records, class_to_idx, corruption, severity, dev)
        else:
            logits, labels = predict_from_shards(model, cfg, args.cache, records, class_to_idx, corruption, severity, dev, mode, args.batch_size)
        rows.append({"corruption": corruption, "severity": severity, **classification_metrics(labels, logits)})
    return rows, clean_f1


def predict_from_embeddings(model, cache: str, model_name: str, records, class_to_idx: dict[str, int], corruption: str, severity: int, dev) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cache) / "embeddings" / model_name / f"{corruption}_s{severity}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing linear embedding cache: {path}")
    data = np.load(path, allow_pickle=True)
    idx = {str(p): i for i, p in enumerate(data["paths"])}
    take = [idx[r.path] for r in records]
    features = data["features"][take].astype(np.float32, copy=False)
    labels = np.array([class_to_idx[r.label] for r in records])
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    if not isinstance(head, torch.nn.Module):
        raise RuntimeError("Linear model does not expose classifier head")
    return head_logits(head, features, dev, 4096), labels


def predict_from_shards(model, cfg: dict, cache: str, records, class_to_idx: dict[str, int], corruption: str, severity: int, dev, mode: str, batch_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    manifest = read_manifest(Path(cache), corruption, severity)
    tf = build_transforms(model, False, cfg.get("data", {}).get("image_size"))
    ds = ShardDataset(records, class_to_idx, manifest, tf)
    loader = make_loader(ds, batch_size or eval_batch_size(cfg, mode), False, cfg["data"]["workers"])
    logits, labels, _, _ = predict(model, loader, dev, desc=f"{corruption}/s{severity}", use_amp=cfg["train"].get("amp", True))
    return logits, labels


class ShardDataset:
    def __init__(self, records, class_to_idx: dict[str, int], manifest: dict[str, str], transform) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image = Image.open(self.manifest[r.path]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[r.label], r.domain, r.path


def read_manifest(root: Path, corruption: str, severity: int) -> dict[str, str]:
    path = root / "images" / corruption / f"s{severity}" / "manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["path"]: row["corrupted_path"] for row in csv.DictReader(f)}


def summarize(rows: list[dict], clean_f1: float, severities: list[int]) -> dict:
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if isinstance(rows[0][k], float)}
    return {
        "corruption": "ACS_" + "_".join(f"S{s}" for s in severities),
        "severity": 0,
        **mean,
        "relative_corruption_drop": (clean_f1 - mean["macro_f1"]) / clean_f1 if clean_f1 else 0.0,
    }


def clean_macro_f1(checkpoint: str, split: str) -> float:
    metrics_path = Path("outputs/metrics") / Path(checkpoint).parent.name / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = "target_test_macro_f1" if split == "target_test" else "id_test_macro_f1"
    return float(data[key])


def default_split(protocol: str) -> str:
    return "target_test" if protocol in {"lodo", "cross_domain"} else "id_test"


def eval_batch_size(cfg: dict, mode: str) -> int:
    batch = batch_size(cfg, mode)
    return max(batch, 128 if mode != "full_finetune" else 64)


if __name__ == "__main__":
    main()
