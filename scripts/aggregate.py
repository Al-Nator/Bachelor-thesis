from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

MODES = ("linear_probe", "partial_finetune", "full_finetune")
PROTOCOLS = ("semantic_ood", "cross_domain", "in_domain", "lodo")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-root", default="outputs/metrics")
    p.add_argument("--out", default="outputs/tables/metrics_long.csv")
    p.add_argument("--summary", default="outputs/tables/summary.csv")
    p.add_argument("--lodo-summary", default="outputs/tables/lodo_summary.csv")
    args = p.parse_args()
    root = Path(args.metrics_root)
    rows = []
    rows += _metrics_rows(root)
    rows += _ood_rows(root)
    rows += _corruption_rows(root)
    rows = _latest_rows(rows)
    _write_long(rows, Path(args.out))
    _write_summary(rows, Path(args.summary))
    _write_lodo_summary(rows, Path(args.lodo_summary))
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"wrote grouped summary -> {args.summary}")
    print(f"wrote LODO summary -> {args.lodo_summary}")


def _metrics_rows(root: Path) -> list[dict[str, str | int | float]]:
    out = []
    logs_root = root.parent / "logs"
    for path in root.rglob("metrics.json"):
        data = _load_json(path)
        if not _current_recipe(path.parent.name, data, logs_root):
            continue
        meta = _meta(path.parent.name, data)
        for metric, value in data.items():
            if isinstance(value, (int, float)) and metric != "seed":
                out.append({**meta, "metric": metric, "value": float(value), "source": "metrics.json"})
    return out


def _ood_rows(root: Path) -> list[dict[str, str | int | float]]:
    out = []
    logs_root = root.parent / "logs"
    for path in root.rglob("ood.json"):
        data = _load_json(path)
        if not _current_recipe(path.parent.name, data, logs_root):
            continue
        meta = _meta(path.parent.name, data)
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                out.append({**meta, "metric": f"ood/{metric}", "value": float(value), "source": "ood.json"})
    return out


def _corruption_rows(root: Path) -> list[dict[str, str | int | float]]:
    out = []
    logs_root = root.parent / "logs"
    for path in root.rglob("corruptions*.csv"):
        meta = _meta(path.parent.name, {})
        if not _current_recipe(path.parent.name, meta, logs_root):
            continue
        source = path.name
        prefix = _corruption_prefix(path.stem)
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                corruption = row["corruption"]
                severity = row["severity"]
                for metric, raw in row.items():
                    if metric in {"corruption", "severity"} or raw == "":
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    name = f"{prefix}/{corruption}/{metric}" if severity == "0" else f"{prefix}/{corruption}/s{severity}/{metric}"
                    out.append({**meta, "metric": name, "value": value, "source": source})
    return out


def _corruption_prefix(stem: str) -> str:
    if stem == "corruptions":
        return "corruption"
    if stem == "corruptions_subset":
        return "corruption_subset"
    return stem.replace("corruptions_", "corruption_")


def _write_long(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["protocol", "model", "mode", "domain_or_heldout", "metric", "seed", "value", "source", "run"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    latest: dict[tuple[str, str, str, str, str, int, str], dict[str, str | int | float]] = {}
    for row in rows:
        key = (
            str(row["protocol"]),
            str(row["model"]),
            str(row["mode"]),
            str(row["domain_or_heldout"]),
            str(row["metric"]),
            int(row["seed"]),
            str(row["source"]),
        )
        if key not in latest or str(row["run"]) > str(latest[key]["run"]):
            latest[key] = row
    return list(latest.values())


def _write_summary(rows: list[dict[str, str | int | float]], path: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["protocol"]), str(row["model"]), str(row["mode"]), str(row["domain_or_heldout"]), str(row["metric"]))
        groups.setdefault(key, []).append(float(row["value"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["protocol", "model", "mode", "domain_or_heldout", "metric", "n", "mean", "std", "mean_std"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            values = groups[key]
            avg = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            writer.writerow(
                {
                    "protocol": key[0],
                    "model": key[1],
                    "mode": key[2],
                    "domain_or_heldout": key[3],
                    "metric": key[4],
                    "n": len(values),
                    "mean": avg,
                    "std": sd,
                    "mean_std": f"{avg:.4f} ± {sd:.4f}",
                }
            )


def _write_lodo_summary(rows: list[dict[str, str | int | float]], path: Path) -> None:
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    groups: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        if row["protocol"] != "lodo" or row["metric"] != "target_test_macro_f1" or row["source"] != "metrics.json":
            continue
        key = (str(row["model"]), str(row["mode"]), int(row["seed"]))
        groups.setdefault(key, {})[str(row["domain_or_heldout"])] = float(row["value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "mode", "seed", *domains, "mean_lodo_f1", "worst_lodo_f1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            vals = groups[key]
            present = [vals[d] for d in domains if d in vals]
            if not present:
                continue
            writer.writerow(
                {
                    "model": key[0],
                    "mode": key[1],
                    "seed": key[2],
                    **{d: vals.get(d, "") for d in domains},
                    "mean_lodo_f1": mean(present),
                    "worst_lodo_f1": min(present),
                }
            )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _current_recipe(run: str, data: dict, logs_root: Path) -> bool:
    log = logs_root / run / "config_used.yaml"
    if not log.exists():
        return False
    text = log.read_text(encoding="utf-8")
    if "image_size: 256" not in text:
        return False
    model = str(data.get("model") or _parse_run(run)["model"])
    mode = str(data.get("mode") or _parse_run(run)["mode"])
    if "vit" in model and f"{mode}: cls_patch_mean" not in text:
        return False
    return True


def _meta(run: str, data: dict) -> dict[str, str | int]:
    parsed = _parse_run(run)
    return {
        "protocol": str(data.get("protocol") or parsed["protocol"]),
        "model": str(data.get("model") or parsed["model"]),
        "mode": str(data.get("mode") or parsed["mode"]),
        "domain_or_heldout": str(data.get("domain_or_heldout") or parsed["domain_or_heldout"]),
        "seed": int(data.get("seed") or parsed["seed"]),
        "run": run,
    }


def _parse_run(run: str) -> dict[str, str | int]:
    parts = run.split("_")
    seed_part = next((p for p in reversed(parts) if p.startswith("seed")), "seed0")
    seed = int(seed_part.replace("seed", "") or 0)
    seed_idx = parts.index(seed_part)
    before_seed = parts[:seed_idx]
    mode, mode_start = _find_mode(before_seed)
    protocol, protocol_end = _find_protocol(before_seed)
    model = "_".join(before_seed[protocol_end:mode_start])
    domain = ""
    if protocol_end < mode_start:
        # In run names with a domain tag, model starts after the tag. Use known model suffixes by finding mode from right.
        candidates = before_seed[protocol_end:mode_start]
        for i in range(len(candidates)):
            possible_model = "_".join(candidates[i:])
            if possible_model:
                model = possible_model
                domain = "_".join(candidates[:i])
                if _looks_like_model(possible_model):
                    break
    return {"protocol": protocol, "model": model, "mode": mode, "domain_or_heldout": domain, "seed": seed}


def _find_mode(parts: list[str]) -> tuple[str, int]:
    joined = "_".join(parts)
    for mode in MODES:
        marker = f"_{mode}"
        idx = joined.rfind(marker)
        if idx >= 0:
            prefix = joined[:idx]
            return mode, len(prefix.split("_"))
    return "unknown", max(len(parts) - 1, 0)


def _find_protocol(parts: list[str]) -> tuple[str, int]:
    joined = "_".join(parts)
    for protocol in PROTOCOLS:
        marker = f"_{protocol}_"
        idx = joined.find(marker)
        if idx >= 0:
            prefix = joined[:idx]
            return protocol, len(f"{prefix}_{protocol}".split("_"))
    return "unknown", 1


def _looks_like_model(name: str) -> bool:
    return name.startswith(("dinov3_", "resnet"))


if __name__ == "__main__":
    main()
