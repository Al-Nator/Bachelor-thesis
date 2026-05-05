from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import OfficeHomeDataset, build_splits, make_loader
from src.eval import predict, save_predictions
from src.models import build_transforms, create_classifier
from src.ood import semantic_ood_report
from src.train_config import batch_size, feature_pool
from src.utils import device, load_checkpoint, load_config, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    args = p.parse_args()
    dev = device()
    ckpt = load_checkpoint(args.checkpoint, map_location=dev)
    if "cfg" not in ckpt and not args.config:
        raise ValueError("Checkpoint has no cfg; pass --config for legacy checkpoints")
    cfg = ckpt.get("cfg") or load_config(args.config)
    if cfg["dataset"].get("protocol") != "semantic_ood":
        cfg = {**cfg, "dataset": {**cfg["dataset"], "protocol": "semantic_ood"}}
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
    id_loader = make_loader(OfficeHomeDataset(splits["id_test"], class_to_idx, tf), batch, False, cfg["data"]["workers"])
    unk_loader = make_loader(OfficeHomeDataset(splits["unknown_test"], class_to_idx, tf), batch, False, cfg["data"]["workers"])
    id_logits, id_labels, id_domains, id_paths = predict(model, id_loader, dev, desc="id_test")
    unk_logits, unk_labels, unk_domains, unk_paths = predict(model, unk_loader, dev, desc="unknown_test")
    out = Path(args.out) if args.out else Path("outputs/metrics") / Path(args.checkpoint).parent.name / "ood.json"
    pred_dir = _prediction_dir(out)
    save_predictions(pred_dir / "ood_id_test", id_logits, id_labels, id_domains, id_paths)
    save_predictions(pred_dir / "ood_unknown_test", unk_logits, unk_labels, unk_domains, unk_paths)
    save_json(semantic_ood_report(id_logits, unk_logits), out)


def _prediction_dir(out: Path) -> Path:
    parts = list(out.parts)
    if "metrics" in parts:
        parts[parts.index("metrics")] = "predictions"
        return Path(*parts).parent
    return out.parent


if __name__ == "__main__":
    main()
