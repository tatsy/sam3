#!/usr/bin/env bash

set -euo pipefail

uv run torchrun \         
  --standalone \
  --nproc_per_node=2 \
  finetune.py \
  --config ./configs/all_categories.yml \
  --coco-json \
    ./data/mvtec_loco_anno/cvat_coco/annotations/instances_default.json \
  --image-root \
    ./data/mvtec_loco_anno/cvat_coco/images/default \
  --output-dir \
    ./models/sam3_mvtec_loco_all_categories_lora \
  --lora-rank 8 \
  --num-workers 4 \
  --epochs 10
