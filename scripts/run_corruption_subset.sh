#!/usr/bin/env bash
set -euo pipefail

# Fast corruption subset:
# 16 checkpoints x 5 corruption types x severities 1/3/5 = 240 corrupted inference passes.
#
# Usage:
#   bash scripts/run_corruption_subset.sh

SEED="${SEED:-42}"
DOMAINS=(Clipart Product)
CORRUPTIONS=(gaussian_noise motion_blur fog contrast jpeg_compression)
SEVERITIES=(1 3 5)

LINEAR_MODELS=(
  dinov3_convnext_tiny
  dinov3_vit_s_plus
  dinov3_convnext_base
  dinov3_vit_b
  dinov3_convnext_large
  dinov3_vit_l
)
PARTIAL_MODELS=(
  dinov3_convnext_large
  dinov3_vit_l
)

run() {
  echo
  echo ">>> $*"
  "$@"
}

latest_ckpt() {
  local domain="$1" model="$2" mode="$3"
  find outputs/checkpoints -path "outputs/checkpoints/*_lodo_${domain}_${model}_${mode}_seed${SEED}/best.pt" | sort | tail -n 1
}

eval_one() {
  local domain="$1" model="$2" mode="$3" ckpt run_dir out
  ckpt="$(latest_ckpt "$domain" "$model" "$mode")"
  if [[ -z "$ckpt" ]]; then
    echo "missing checkpoint: ${domain} ${model} ${mode} seed${SEED}" >&2
    return 1
  fi
  run_dir="$(basename "$(dirname "$ckpt")")"
  out="outputs/metrics/${run_dir}/corruptions_subset.csv"
  if [[ -f "$out" ]]; then
    echo "skip existing: $out"
    return
  fi
  run uv run python scripts/eval_corruptions.py \
    --checkpoint "$ckpt" \
    --model "$model" \
    --split target_test \
    --corruptions "${CORRUPTIONS[@]}" \
    --severities "${SEVERITIES[@]}" \
    --summary-name subset_ACS \
    --out "$out"
}

for domain in "${DOMAINS[@]}"; do
  for model in "${LINEAR_MODELS[@]}"; do
    eval_one "$domain" "$model" linear_probe
  done
  for model in "${PARTIAL_MODELS[@]}"; do
    eval_one "$domain" "$model" partial_finetune
  done
done

run uv run python scripts/aggregate.py
