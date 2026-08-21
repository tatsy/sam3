#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="./data/mvtec_loco_anno/cvat_coco"

uv run python finetune.py \
  --coco-json "${DATA_DIR}/annotations/instances_default.json" \
  --image-root "${DATA_DIR}/images/default" \
  --output-dir ./models/sam3_mvtec_loco_lora \
  --lora-rank 8 \
