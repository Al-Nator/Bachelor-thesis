from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Callable

from PIL import Image

try:
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOMAINS = {"art": "Art", "clipart": "Clipart", "product": "Product", "realworld": "RealWorld", "real world": "RealWorld"}


@dataclass(frozen=True)
class Record:
    path: str
    label: str
    domain: str


class OfficeHomeDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
        corruption: Callable | None = None,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image = Image.open(r.path).convert("RGB")
        if self.corruption:
            image = self.corruption(image, r.path)
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx.get(r.label, -1), r.domain, r.path


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    if DataLoader is None:
        raise ModuleNotFoundError("torch is required for DataLoader; install dependencies with uv sync")
    import torch

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def build_splits(cfg: dict, seed: int) -> tuple[dict[str, list[Record]], dict[str, int]]:
    cfg = _normalized_cfg(cfg)
    records = scan_officehome(cfg["dataset"]["root"])
    protocol = cfg["dataset"].get("protocol", "lodo")
    if protocol == "lodo":
        return _lodo(records, cfg, seed)
    if protocol == "cross_domain":
        return _cross_domain(records, cfg, seed)
    if protocol == "in_domain":
        return _in_domain(records, cfg, seed)
    if protocol == "semantic_ood":
        return _semantic_ood(records, cfg, seed)
    raise ValueError(f"Unknown protocol: {protocol}")


def scan_officehome(root: str | Path) -> list[Record]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Office-Home root not found: {root}")
    out: list[Record] = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        domain = _domain_name(domain_dir.name)
        if not domain:
            continue
        for class_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            for img in sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
                out.append(Record(str(img), class_dir.name, domain))
    if not out:
        raise RuntimeError(f"No images found in Office-Home layout under {root}")
    return out


def dataset_summary(records: list[Record]) -> dict[str, object]:
    domains = sorted({r.domain for r in records})
    classes = sorted({r.label for r in records})
    by_domain = {d: sum(r.domain == d for r in records) for d in domains}
    by_domain_class = {
        d: {c: sum(r.domain == d and r.label == c for r in records) for c in classes}
        for d in domains
    }
    return {
        "images": len(records),
        "domains": domains,
        "classes": len(classes),
        "by_domain": by_domain,
        "min_images_per_domain_class": min(v for domain in by_domain_class.values() for v in domain.values()),
    }


def _lodo(records: list[Record], cfg: dict, seed: int):
    heldout = cfg["dataset"]["heldout_domain"]
    source = [r for r in records if r.domain != heldout]
    target = [r for r in records if r.domain == heldout]
    splits = dict(zip(("train", "val", "id_test"), _split_60_20_20(source, seed)))
    splits["target_test"] = target
    return splits, _classes(source)


def _cross_domain(records: list[Record], cfg: dict, seed: int):
    source_domains = set(cfg["dataset"]["source_domains"])
    target_domains = set(cfg["dataset"]["target_domains"])
    source = [r for r in records if r.domain in source_domains]
    target = [r for r in records if r.domain in target_domains]
    splits = dict(zip(("train", "val", "id_test"), _split_60_20_20(source, seed)))
    splits["target_test"] = target
    return splits, _classes(source)


def _in_domain(records: list[Record], cfg: dict, seed: int):
    domain = cfg["dataset"]["domain"]
    data = [r for r in records if r.domain == domain]
    return dict(zip(("train", "val", "id_test"), _split_60_20_20(data, seed))), _classes(data)


def _semantic_ood(records: list[Record], cfg: dict, seed: int):
    visible_domains = set(cfg["dataset"]["visible_domains"])
    records = [r for r in records if r.domain in visible_domains]
    labels = sorted({r.label for r in records})
    known = _semantic_known_classes(labels, cfg["dataset"], seed)
    known = set(known)
    known_records = [r for r in records if r.label in known]
    unknown_records = [r for r in records if r.label not in known]
    splits = dict(zip(("train", "val", "id_test"), _split_60_20_20(known_records, seed)))
    splits["unknown_test"] = unknown_records
    return splits, _classes(known_records)


def _split_60_20_20(records: list[Record], seed: int) -> tuple[list[Record], list[Record], list[Record]]:
    train, temp = _stratified_split(records, 0.4, seed)
    val, test = _stratified_split(temp, 0.5, seed)
    return train, val, test


def _stratified_split(records: list[Record], test_size: float, seed: int) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(r.label, []).append(r)
    left, right = [], []
    for label in sorted(groups):
        items = groups[label][:]
        rng.shuffle(items)
        n_right = max(1, round(len(items) * test_size)) if len(items) > 1 else 0
        n_right = min(n_right, len(items) - 1) if len(items) > 1 else n_right
        right.extend(items[:n_right])
        left.extend(items[n_right:])
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def _classes(records: list[Record]) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted({r.label for r in records}))}


def _semantic_known_classes(labels: list[str], ds_cfg: dict, seed: int) -> list[str]:
    labels_set = set(labels)
    if ds_cfg.get("known_classes"):
        known = set(ds_cfg["known_classes"])
        _validate_classes(known, labels_set, "known_classes")
        return sorted(known)
    if ds_cfg.get("unknown_classes"):
        unknown = set(ds_cfg["unknown_classes"])
        _validate_classes(unknown, labels_set, "unknown_classes")
        return sorted(labels_set - unknown)
    known_fraction = ds_cfg.get("known_fraction", 0.7)
    n_known = int(len(labels) * known_fraction)
    shuffled = labels[:]
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n_known])


def _validate_classes(classes: set[str], labels: set[str], field: str) -> None:
    missing = sorted(classes - labels)
    if missing:
        raise ValueError(f"Unknown classes in dataset.{field}: {missing}")


def _domain_name(name: str) -> str | None:
    return DOMAINS.get(name.replace("_", " ").lower())


def normalize_domain(name: str) -> str:
    domain = _domain_name(name)
    if not domain:
        raise ValueError(f"Unknown Office-Home domain: {name}")
    return domain


def _normalized_cfg(cfg: dict) -> dict:
    cfg = {**cfg, "dataset": {**cfg["dataset"]}}
    ds = cfg["dataset"]
    for key in ("heldout_domain", "domain"):
        if key in ds:
            ds[key] = normalize_domain(ds[key])
    for key in ("source_domains", "target_domains", "visible_domains"):
        if key in ds:
            ds[key] = [normalize_domain(d) for d in ds[key]]
    return cfg
