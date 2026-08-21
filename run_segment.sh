#!/usr/bin/env bash

set -euo pipefail

uv run python segment.py \
  --input-dir data/mvtec_loco_anno/cvat_coco/images/default \
  --output-dir outputs/sam3_ft_all_categories_validation \
  --config configs/all_categories.yml \
  --base-model facebook/sam3 \
  --adapter-dir ./models/sam3_mvtec_loco_all_categories_lora/adapter \
  --ground-truth-json data/mvtec_loco_anno/cvat_coco/annotations/instances_default.json \
  --split-json models/sam3_mvtec_loco_all_categories_lora/split.json \
  --eval-split validation \
  --num-images 100 \
  --use-presence-filter \
  --overwrite


