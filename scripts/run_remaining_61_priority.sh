#!/usr/bin/env bash
set -euo pipefail

# Remaining all-protocol trainings for ResNet-50 and DINOv3 Large pair.
# Ordered by priority: useful linear protocols first, expensive full FT last.
#
# Missing matrix from the current seed-42 state:
#   linear_probe:     17 trainings
#   partial_finetune: 18 trainings
#   resnet full FT:    6 trainings
#   large full FT:    20 trainings
#   total:            61 trainings
#
# Usage:
#   bash scripts/run_remaining_61_priority.sh
#   SEED=123 bash scripts/run_remaining_61_priority.sh

SEED="${SEED:-42}"
DOMAINS=(Art Clipart Product RealWorld)
LARGE_MODELS=(dinov3_convnext_large dinov3_vit_l)

RESNET_CFG="configs/officehome_resnet50.yaml"
RESNET_FULL_CFG="configs/officehome_dinov3_base_full_finetune.yaml"
LARGE_CFG="configs/officehome_dinov3_large.yaml"

mkdir -p outputs/checkpoints outputs/metrics

run() {
  echo
  echo ">>> $*"
  "$@"
}

tag_for() {
  local protocol="$1" domain="${2:-}"
  case "$protocol" in
    lodo|in_domain) echo "$domain" ;;
    cross_domain) echo "ProductRealWorld_to_ArtClipart" ;;
    semantic_ood) echo "known_unknown" ;;
    *) echo "" ;;
  esac
}

run_complete() {
  local pattern="$1" model="$2" mode="$3"
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
    if [[ "$model" == *vit* ]]; then
      rg -q "${mode}: cls_patch_mean" "$log" || continue
    fi
    return 0
  done < <(find outputs/checkpoints -path "$pattern" -print)
  return 1
}

train_one() {
  local config="$1" model="$2" mode="$3" protocol="$4" domain="${5:-}"
  local tag pattern
  tag="$(tag_for "$protocol" "$domain")"
  pattern="outputs/checkpoints/*_${protocol}_${tag}_${model}_${mode}_seed${SEED}/best.pt"
  if run_complete "$pattern" "$model" "$mode"; then
    echo "skip existing: ${protocol} ${tag} ${model} ${mode} seed${SEED}"
    return
  fi
  case "$protocol" in
    lodo)
      run uv run python scripts/train_lodo.py --config "$config" --model "$model" --mode "$mode" --protocol lodo --heldout-domain "$domain" --seed "$SEED"
      ;;
    in_domain)
      run uv run python scripts/train_lodo.py --config "$config" --model "$model" --mode "$mode" --protocol in_domain --domain "$domain" --seed "$SEED"
      ;;
    cross_domain|semantic_ood)
      run uv run python scripts/train_lodo.py --config "$config" --model "$model" --mode "$mode" --protocol "$protocol" --seed "$SEED"
      ;;
    *)
      echo "unknown protocol: $protocol" >&2
      exit 2
      ;;
  esac
}

echo "== Priority 1/5: missing linear protocols, 17 trainings =="
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" linear_probe semantic_ood
done
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" linear_probe cross_domain
done
for domain in "${DOMAINS[@]}"; do
  for model in "${LARGE_MODELS[@]}"; do
    train_one "$LARGE_CFG" "$model" linear_probe in_domain "$domain"
  done
done
train_one "$RESNET_CFG" resnet50 linear_probe cross_domain
for domain in "${DOMAINS[@]}"; do
  train_one "$RESNET_CFG" resnet50 linear_probe in_domain "$domain"
done

echo "== Priority 2/5: missing partial fine-tuning protocols, 18 trainings =="
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" partial_finetune semantic_ood
done
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" partial_finetune cross_domain
done
for domain in "${DOMAINS[@]}"; do
  for model in "${LARGE_MODELS[@]}"; do
    train_one "$LARGE_CFG" "$model" partial_finetune in_domain "$domain"
  done
done
train_one "$RESNET_CFG" resnet50 partial_finetune semantic_ood
train_one "$RESNET_CFG" resnet50 partial_finetune cross_domain
for domain in "${DOMAINS[@]}"; do
  train_one "$RESNET_CFG" resnet50 partial_finetune in_domain "$domain"
done

echo "== Priority 3/5: ResNet full fine-tuning outside LODO, 6 trainings =="
train_one "$RESNET_FULL_CFG" resnet50 full_finetune semantic_ood
train_one "$RESNET_FULL_CFG" resnet50 full_finetune cross_domain
for domain in "${DOMAINS[@]}"; do
  train_one "$RESNET_FULL_CFG" resnet50 full_finetune in_domain "$domain"
done

echo "== Priority 4/5: Large full fine-tuning on LODO, 8 trainings =="
for domain in "${DOMAINS[@]}"; do
  for model in "${LARGE_MODELS[@]}"; do
    train_one "$LARGE_CFG" "$model" full_finetune lodo "$domain"
  done
done

echo "== Priority 5/5: Large full fine-tuning on secondary protocols, 12 trainings =="
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" full_finetune semantic_ood
done
for model in "${LARGE_MODELS[@]}"; do
  train_one "$LARGE_CFG" "$model" full_finetune cross_domain
done
for domain in "${DOMAINS[@]}"; do
  for model in "${LARGE_MODELS[@]}"; do
    train_one "$LARGE_CFG" "$model" full_finetune in_domain "$domain"
  done
done

run uv run python scripts/aggregate.py
