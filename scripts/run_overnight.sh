#!/usr/bin/env bash
set -euo pipefail

# Unified thesis training matrix, seed 42 by default.
#
# Usage:
#   bash scripts/run_overnight.sh
#   SEED=123 bash scripts/run_overnight.sh
#   STAGE=main_lodo bash scripts/run_overnight.sh
#   STAGE=semantic_ood bash scripts/run_overnight.sh
#   STAGE=small_all bash scripts/run_overnight.sh
#
# Stages:
#   main_lodo      rows 1-5
#   adaptation     rows 6-7
#   upper_bound    rows 8-9
#   semantic_ood   rows 10-14
#   small_partial  Small pair + ResNet partial FT on 4 LODO domains
#   small_full     Small pair + ResNet full FT on 4 LODO domains
#   small_all      small_partial + small_full
#   eval_ood       OOD metrics for rows 10-14
#   all            full 65 trainings + eval_ood + aggregate

STAGE="${STAGE:-all}"
SEED="${SEED:-42}"
HELDOUTS=(Art Clipart Product RealWorld)

mkdir -p outputs/checkpoints outputs/metrics

run() {
  echo
  echo ">>> $*"
  "$@"
}

run_complete() {
  local pattern="$1" required="${2:-}"
  local ckpt run_dir log
  while IFS= read -r ckpt; do
    run_dir="$(basename "$(dirname "$ckpt")")"
    [[ -f "outputs/metrics/${run_dir}/metrics.json" ]] || continue
    log="outputs/logs/${run_dir}/config_used.yaml"
    [[ -f "$log" ]] || continue
    rg -q "epochs: 50" "$log" || continue
    rg -q "image_size: 256" "$log" || continue
    rg -q "physical_batch_size:" "$log" || continue
    rg -q "effective_batch_size:" "$log" || continue
    [[ -z "$required" ]] || rg -q "$required" "$log" || continue
    return 0
  done < <(find outputs/checkpoints -path "$pattern" -print)
  return 1
}

train_one() {
  local config="$1" model="$2" mode="$3" protocol="$4" extra_name="${5:-}" extra_value="${6:-}"
  local tag=""
  if [[ "$protocol" == "lodo" ]]; then
    tag="$extra_value"
  elif [[ "$protocol" == "semantic_ood" ]]; then
    tag="known_unknown"
  elif [[ "$protocol" == "in_domain" ]]; then
    tag="$extra_value"
  elif [[ "$protocol" == "cross_domain" ]]; then
    tag="ProductRealWorld_to_ArtClipart"
  fi
  local pattern="outputs/checkpoints/*_${protocol}_${tag}_${model}_${mode}_seed${SEED}/best.pt"
  local required=""
  if [[ "$model" == *vit* ]]; then
    required="${mode}: cls_patch_mean"
  fi
  if run_complete "$pattern" "$required"; then
    echo "skip existing: ${protocol} ${tag} ${model} ${mode} seed${SEED}"
    return
  fi
  if [[ -n "$extra_name" ]]; then
    run uv run python scripts/train_lodo.py --config "$config" --model "$model" --mode "$mode" --protocol "$protocol" "$extra_name" "$extra_value" --seed "$SEED"
  else
    run uv run python scripts/train_lodo.py --config "$config" --model "$model" --mode "$mode" --protocol "$protocol" --seed "$SEED"
  fi
}

latest_ckpt() {
  local pattern="$1"
  find outputs/checkpoints -path "$pattern" | sort | tail -n 1
}

eval_ood_for() {
  local config="$1" model="$2"
  local ckpt run_dir
  ckpt="$(latest_ckpt "outputs/checkpoints/*_semantic_ood_known_unknown_${model}_linear_probe_seed${SEED}/best.pt")"
  if [[ -z "$ckpt" ]]; then
    echo "missing semantic OOD checkpoint: ${model} seed${SEED}"
    return
  fi
  run_dir="$(basename "$(dirname "$ckpt")")"
  run uv run python scripts/eval_ood.py --config "$config" --checkpoint "$ckpt" --model "$model" --seed "$SEED" --out "outputs/metrics/${run_dir}/ood.json"
}

run_main_lodo() {
  for heldout in "${HELDOUTS[@]}"; do
    train_one configs/officehome_resnet50.yaml resnet50 linear_probe lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_small.yaml dinov3_convnext_tiny linear_probe lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_small.yaml dinov3_vit_s_plus linear_probe lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base.yaml dinov3_convnext_base linear_probe lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base.yaml dinov3_vit_b linear_probe lodo --heldout-domain "$heldout"
  done
}

run_adaptation() {
  for heldout in "${HELDOUTS[@]}"; do
    train_one configs/officehome_dinov3_base.yaml dinov3_convnext_base partial_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base.yaml dinov3_vit_b partial_finetune lodo --heldout-domain "$heldout"
  done
}

run_upper_bound() {
  for heldout in "${HELDOUTS[@]}"; do
    train_one configs/officehome_dinov3_base_full_finetune.yaml dinov3_convnext_base full_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base_full_finetune.yaml dinov3_vit_b full_finetune lodo --heldout-domain "$heldout"
  done
}

run_small_partial() {
  for heldout in "${HELDOUTS[@]}"; do
    train_one configs/officehome_resnet50.yaml resnet50 partial_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_small.yaml dinov3_convnext_tiny partial_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_small.yaml dinov3_vit_s_plus partial_finetune lodo --heldout-domain "$heldout"
  done
}

run_small_full() {
  for heldout in "${HELDOUTS[@]}"; do
    train_one configs/officehome_dinov3_base_full_finetune.yaml resnet50 full_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base_full_finetune.yaml dinov3_convnext_tiny full_finetune lodo --heldout-domain "$heldout"
    train_one configs/officehome_dinov3_base_full_finetune.yaml dinov3_vit_s_plus full_finetune lodo --heldout-domain "$heldout"
  done
}

run_semantic_ood() {
  train_one configs/officehome_resnet50.yaml resnet50 linear_probe semantic_ood
  train_one configs/officehome_dinov3_small.yaml dinov3_convnext_tiny linear_probe semantic_ood
  train_one configs/officehome_dinov3_small.yaml dinov3_vit_s_plus linear_probe semantic_ood
  train_one configs/officehome_dinov3_base.yaml dinov3_convnext_base linear_probe semantic_ood
  train_one configs/officehome_dinov3_base.yaml dinov3_vit_b linear_probe semantic_ood
}

run_eval_ood() {
  eval_ood_for configs/officehome_resnet50.yaml resnet50
  eval_ood_for configs/officehome_dinov3_small.yaml dinov3_convnext_tiny
  eval_ood_for configs/officehome_dinov3_small.yaml dinov3_vit_s_plus
  eval_ood_for configs/officehome_dinov3_base.yaml dinov3_convnext_base
  eval_ood_for configs/officehome_dinov3_base.yaml dinov3_vit_b
  run uv run python scripts/aggregate.py
}

case "$STAGE" in
  main_lodo)
    run_main_lodo
    ;;
  adaptation)
    run_adaptation
    ;;
  upper_bound)
    run_upper_bound
    ;;
  small_partial)
    run_small_partial
    ;;
  small_full)
    run_small_full
    ;;
  small_all)
    run_small_partial
    run_small_full
    ;;
  semantic_ood)
    run_semantic_ood
    ;;
  eval_ood)
    run_eval_ood
    ;;
  all)
    run_main_lodo
    run_adaptation
    run_upper_bound
    run_small_partial
    run_small_full
    run_semantic_ood
    run_eval_ood
    ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    exit 2
    ;;
esac
