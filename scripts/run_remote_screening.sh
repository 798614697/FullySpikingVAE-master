#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${ASC_CODE_DIR:-/home/zhaozihao/ASC-FSVAE}"
RUN_ROOT="${ASC_RUN_ROOT:-/mnt/ssd/C/ZZH/ASC-FSVAE-runs}"
DEVICE="${ASC_DEVICE:-cuda:0}"
SEED="${ASC_SEED:-2024}"
PYTHON_BIN="${ASC_PYTHON:-python}"

cd "$CODE_DIR"
mkdir -p "$RUN_ROOT/attribute_classifier"

CLASSIFIER="$RUN_ROOT/attribute_classifier/best_valid.pt"
if [[ ! -f "$RUN_ROOT/attribute_classifier/complete.json" ]]; then
  CLASSIFIER_RESUME=()
  if [[ -f "$RUN_ROOT/attribute_classifier/latest_resume.pt" ]]; then
    CLASSIFIER_RESUME=(--resume "$RUN_ROOT/attribute_classifier/latest_resume.pt")
  fi
  "$PYTHON_BIN" train_attribute_classifier.py \
    --config NetworkConfigs/ASC_CelebA_D.yaml \
    --output-dir "$RUN_ROOT/attribute_classifier" \
    --device "$DEVICE" --seed "$SEED" "${CLASSIFIER_RESUME[@]}"
fi
test -f "$CLASSIFIER"

for EXPERIMENT in A B C D; do
  case "$EXPERIMENT" in
    A) RUN_NAME="A_fsvae" ;;
    B) RUN_NAME="B_conditional" ;;
    C) RUN_NAME="C_image_attr" ;;
    D) RUN_NAME="D_full" ;;
  esac
  OUTPUT="$RUN_ROOT/$RUN_NAME/seed$SEED"
  mkdir -p "$OUTPUT"
  EXTRA=()
  if [[ "$EXPERIMENT" == "C" || "$EXPERIMENT" == "D" ]]; then
    EXTRA=(--classifier-checkpoint "$CLASSIFIER")
  fi
  RESUME=()
  if [[ -f "$OUTPUT/latest_resume.pt" ]]; then
    RESUME=(--resume "$OUTPUT/latest_resume.pt")
  fi
  "$PYTHON_BIN" main_asc_fsvae.py --experiment "$EXPERIMENT" \
    --config "NetworkConfigs/ASC_CelebA_${EXPERIMENT}.yaml" \
    --output-dir "$OUTPUT" --device "$DEVICE" --seed "$SEED" \
    "${EXTRA[@]}" "${RESUME[@]}"
done
