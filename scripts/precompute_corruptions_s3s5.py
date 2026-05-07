from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.corruptions import CORRUPTIONS, Corruption
from src.data import OfficeHomeDataset, make_loader, scan_officehome
from src.models import build_transforms, create_classifier
from src.train_config import feature_pool
from src.utils import amp_dtype, device, load_checkpoint, load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/officehome_dinov3_base.yaml")
    p.add_argument("--out", default="outputs/corruption_cache/imagenet_c_s3_s5")
    p.add_argument("--severities", nargs="+", type=int, default=[3, 5])
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--embeddings", action="store_true")
    p.add_argument("--checkpoint-root", default="outputs/checkpoints")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    records = scan_officehome(cfg["dataset"]["root"])
    out = Path(args.out)
    _validate(args.corruptions, args.severities)
    for corruption in tqdm(args.corruptions, desc="corruption types"):
        for severity in tqdm(args.severities, desc=corruption, leave=False):
            precompute_images(records, out, corruption, severity, args)
    if args.embeddings:
        precompute_linear_embeddings(records, out, args)


def precompute_images(records, root: Path, corruption: str, severity: int, args) -> None:
    setting = root / "images" / corruption / f"s{severity}"
    manifest = setting / "manifest.csv"
    rows, jobs = [], []
    for idx, r in enumerate(records):
        dst = setting / f"shard_{idx // args.shard_size:05d}" / f"{idx:06d}_{_hash(r.path)}.jpg"
        rows.append({"path": r.path, "label": r.label, "domain": r.domain, "corrupted_path": str(dst)})
        if args.overwrite or not dst.exists():
            jobs.append((r.path, str(dst), corruption, severity, args.seed, args.jpeg_quality, args.image_size))
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(_write_corrupted, jobs), total=len(jobs), desc=f"{corruption}/s{severity}", leave=False))
    setting.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "domain", "corrupted_path"])
        writer.writeheader()
        writer.writerows(rows)


def _write_corrupted(job) -> None:
    src, dst, corruption, severity, seed, quality, image_size = job
    image = Image.open(src).convert("RGB")
    if image_size:
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    out = Corruption(corruption, severity, seed)(image, src)
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, format="JPEG", quality=quality, optimize=True)


def precompute_linear_embeddings(records, root: Path, args) -> None:
    ckpts = linear_backbone_checkpoints(Path(args.checkpoint_root))
    if not ckpts:
        print("no linear_probe checkpoints found")
        return
    dev = device()
    for model_name, ckpt_path in tqdm(ckpts.items(), desc="linear backbones"):
        ckpt = load_checkpoint(ckpt_path, map_location=dev)
        cfg = ckpt["cfg"]
        mode = ckpt.get("meta", {}).get("mode", "linear_probe")
        model = create_classifier(model_name, len(ckpt["class_to_idx"]), pretrained=False, feature_pool=feature_pool(cfg, model_name, mode)).to(dev)
        model.load_state_dict(ckpt["model"])
        tf = build_transforms(model, False, cfg.get("data", {}).get("image_size"))
        for corruption in tqdm(args.corruptions, desc=model_name, leave=False):
            for severity in tqdm(args.severities, desc=corruption, leave=False):
                path = root / "embeddings" / model_name / f"{corruption}_s{severity}.npz"
                if path.exists() and not args.overwrite:
                    continue
                manifest = read_manifest(root, corruption, severity)
                ds = ManifestImageDataset(manifest, tf)
                loader = make_loader(ds, args.batch_size, False, cfg["data"]["workers"])
                features, labels, domains, paths = embed(model, loader, dev, cfg["train"].get("amp", True), f"embed {model_name} {corruption}/s{severity}")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, features=features.astype(np.float16), labels=labels, domains=domains, paths=paths)


def linear_backbone_checkpoints(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.glob("*/best.pt")):
        ckpt = load_checkpoint(path, map_location="cpu")
        meta = ckpt.get("meta", {})
        if meta.get("mode") == "linear_probe":
            out.setdefault(str(meta["model"]), path)
    return out


@torch.inference_mode()
def embed(model, loader, dev, use_amp: bool, desc: str):
    model.eval()
    xs, labels, domains, paths = [], [], [], []
    for x, label, domain, path in tqdm(loader, desc=desc, leave=False):
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype(), enabled=use_amp and dev.type == "cuda"):
            feats = model.forward_head(model.forward_features(x), pre_logits=True)
        xs.append(feats.float().cpu().numpy())
        labels.extend(label)
        domains.extend(domain)
        paths.extend(path)
    return np.concatenate(xs), np.array(labels), np.array(domains), np.array(paths)


class ManifestImageDataset:
    def __init__(self, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["corrupted_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label"], row["domain"], row["path"]


def read_manifest(root: Path, corruption: str, severity: int) -> list[dict[str, str]]:
    path = root / "images" / corruption / f"s{severity}" / "manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hash(value: str) -> str:
    return blake2b(value.encode(), digest_size=8).hexdigest()


def _validate(corruptions: list[str], severities: list[int]) -> None:
    unknown = sorted(set(corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruptions: {unknown}")
    bad = [s for s in severities if s < 1 or s > 5]
    if bad:
        raise ValueError(f"Bad severities: {bad}")


if __name__ == "__main__":
    main()
