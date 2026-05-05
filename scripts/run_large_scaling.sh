#!/usr/bin/env bash
set -euo pipefail

# Scaling check: DINOv3 ConvNeXt Large vs DINOv3 ViT-L.
# Runs 8 Office-Home LODO linear-probing trainings for one seed.
#
# Usage:
#   bash scripts/run_large_scaling.sh
#   SEED=123 bash scripts/run_large_scaling.sh

SEED="${SEED:-42}"
CONFIG="configs/officehome_dinov3_large.yaml"
HELDOUTS=(Art Clipart Product RealWorld)
MODELS=(dinov3_convnext_large dinov3_vit_l)

mkdir -p outputs/checkpoints outputs/metrics

run() {
  echo
  echo ">>> $*"
  "$@"
}

run_complete() {
  local pattern="$1" required="$2"
  local ckpt run_dir log
  while IFS= read -r ckpt; do
    run_dir="$(basename "$(dirname "$ckpt")")"
    [[ -f "outputs/metrics/${run_dir}/metrics.json" ]] || continue
    log="outputs/logs/${run_dir}/config_used.yaml"
    [[ -f "$log" ]] || continue
    rg -q "epochs: 50" "$log" || continue
    rg -q "image_size: 256" "$log" || continue
    rg -q "linear_probe: cls_patch_mean" "$log" || continue
    [[ -z "$required" ]] || rg -q "$required" "$log" || continue
    return 0
  done < <(find outputs/checkpoints -path "$pattern" -print)
  return 1
}

for heldout in "${HELDOUTS[@]}"; do
  for model in "${MODELS[@]}"; do
    pattern="outputs/checkpoints/*_lodo_${heldout}_${model}_linear_probe_seed${SEED}/best.pt"
    if run_complete "$pattern" "effective_batch_size:"; then
      echo "skip existing: lodo ${heldout} ${model} linear_probe seed${SEED}"
      continue
    fi
    run uv run python scripts/train_lodo.py \
      --config "$CONFIG" \
      --model "$model" \
      --mode linear_probe \
      --protocol lodo \
      --heldout-domain "$heldout" \
      --seed "$SEED"
  done
done

run uv run python scripts/aggregate.py
