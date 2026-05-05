from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.train import run_training
from src.utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["linear_probe", "partial_finetune", "full_finetune"], required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--protocol")
    p.add_argument("--heldout-domain")
    p.add_argument("--domain")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.protocol:
        cfg["dataset"]["protocol"] = args.protocol
    if args.heldout_domain:
        cfg["dataset"]["heldout_domain"] = args.heldout_domain
    if args.domain:
        cfg["dataset"]["domain"] = args.domain
    seeds = cfg["seeds"] if args.all_seeds else [args.seed or cfg["seeds"][0]]
    for seed in seeds:
        report = run_training(cfg, args.model, args.mode, seed)
        print({"seed": seed, **report})


if __name__ == "__main__":
    main()
