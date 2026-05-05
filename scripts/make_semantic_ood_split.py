from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import scan_officehome


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/OfficeHomeDataset_10072016")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unknown-count", type=int, default=10)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--min-images", type=int, default=100)
    p.add_argument("--out-json", default="configs/semantic_ood_split_seed42.json")
    p.add_argument("--out-yaml", default="configs/semantic_ood_split_seed42.yaml")
    args = p.parse_args()

    records = scan_officehome(args.data_root)
    counts = dict(sorted(Counter(r.label for r in records).items()))
    candidates = {c: n for c, n in counts.items() if n >= args.min_images}
    if len(candidates) < args.unknown_count:
        raise ValueError("Not enough semantic OOD candidates after min-images filtering")

    buckets = _frequency_groups(candidates, args.groups)
    unknown = _pick_unknown(buckets, args.unknown_count, args.seed)
    known = sorted(set(counts) - set(unknown))
    data = {
        "seed": args.seed,
        "unknown_count": args.unknown_count,
        "min_images": args.min_images,
        "class_counts": counts,
        "frequency_groups": buckets,
        "known_classes": known,
        "unknown_classes": unknown,
    }
    _write_json(data, args.out_json)
    _write_yaml(data, args.out_yaml)
    print(f"classes={len(counts)} known={len(known)} unknown={len(unknown)}")
    print("unknown:", ", ".join(unknown))


def _frequency_groups(counts: dict[str, int], groups: int) -> dict[str, list[str]]:
    names = ["low", "mid_low", "mid_high", "high"] if groups == 4 else [f"group_{i + 1}" for i in range(groups)]
    items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    buckets: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        lo = round(i * len(items) / groups)
        hi = round((i + 1) * len(items) / groups)
        buckets[name] = [c for c, _ in items[lo:hi]]
    return buckets


def _pick_unknown(buckets: dict[str, list[str]], total: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    names = list(buckets)
    quotas = {name: total // len(names) for name in names}
    for name in names[: total % len(names)]:
        quotas[name] += 1
    selected: list[str] = []
    for name in names:
        choices = buckets[name][:]
        rng.shuffle(choices)
        selected.extend(choices[: quotas[name]])
    return sorted(selected)


def _write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
