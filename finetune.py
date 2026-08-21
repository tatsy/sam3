from __future__ import annotations

import json
import random
import argparse
from typing import Any, TypedDict
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn
from pycocotools import mask as mask_utils
from transformers import (
    Trainer,
    Sam3Model,
    Sam3Processor,
    EvalPrediction,
    TrainingArguments,
)
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import Dataset


# キーはinstances_default.jsonのcategories[].nameと一致させます。
PROMPT_BY_CATEGORY = {
    'orange': 'orange fruit',
    'nectarine': 'nectarine fruit',
    'cereals': 'cereal flakes',
    'banana_chips': 'dried banana slices',
    'almonds': 'almonds',
    'white_box': 'white plastic box',
}


class Sam3BatchData(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def decode_coco_segmentation(
    segmentation: Any,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode polygon, compressed RLE, or uncompressed RLE."""

    if isinstance(segmentation, list):
        # COCO polygon
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)

    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get('counts'), list):
            # Uncompressed RLE
            rle = mask_utils.frPyObjects(
                segmentation,
                height,
                width,
            )
        else:
            # Compressed RLE
            rle = segmentation
    else:
        raise TypeError(f'Unsupported segmentation type: {type(segmentation)}')

    decoded = mask_utils.decode(rle)

    # Multiple polygons can produce [H, W, N].
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)

    return decoded.astype(bool)


def resolve_image_path(
    image_root: Path,
    file_name: str,
) -> Path:
    """Handle both plain filenames and CVAT-style relative paths."""

    candidates = [
        image_root / file_name,
        image_root / Path(file_name).name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not locate image '{file_name}' under {image_root}")


class CocoPromptSegmentationDataset(Dataset):
    """Convert COCO instances to image-prompt semantic masks."""

    def __init__(
        self,
        *,
        coco_data: dict[str, Any],
        image_root: Path,
        image_ids: set[int],
        prompt_by_category: dict[str, str],
        include_empty_masks: bool = False,
    ) -> None:
        self.image_root = image_root
        self.prompt_by_category = prompt_by_category

        self.images_by_id = {int(image['id']): image for image in coco_data['images'] if int(image['id']) in image_ids}

        self.categories_by_id = {int(category['id']): category['name'] for category in coco_data['categories']}

        self.annotations_by_key: dict[
            tuple[int, int],
            list[dict[str, Any]],
        ] = defaultdict(list)

        for annotation in coco_data['annotations']:
            image_id = int(annotation['image_id'])
            category_id = int(annotation['category_id'])

            if image_id in self.images_by_id:
                self.annotations_by_key[(image_id, category_id)].append(annotation)

        self.samples: list[tuple[int, int]] = []

        for image_id in sorted(self.images_by_id):
            for category_id, category_name in self.categories_by_id.items():
                if category_name not in prompt_by_category:
                    continue

                has_annotations = bool(self.annotations_by_key.get((image_id, category_id)))

                # アノテーション漏れをnegativeと誤認しないため、
                # 最初は正例だけを使用します。
                if has_annotations or include_empty_masks:
                    self.samples.append((image_id, category_id))

        if not self.samples:
            raise RuntimeError('No training samples were created. Check category names in PROMPT_BY_CATEGORY.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, category_id = self.samples[index]

        image_info = self.images_by_id[image_id]
        width = int(image_info['width'])
        height = int(image_info['height'])

        image_path = resolve_image_path(
            self.image_root,
            image_info['file_name'],
        )
        image = Image.open(image_path).convert('RGB')

        union_mask = np.zeros((height, width), dtype=bool)

        for annotation in self.annotations_by_key.get(
            (image_id, category_id),
            [],
        ):
            instance_mask = decode_coco_segmentation(
                annotation['segmentation'],
                height=height,
                width=width,
            )
            union_mask |= instance_mask

        category_name = self.categories_by_id[category_id]

        return {
            'image': image,
            # 0/255 uint8にしてprocessorへ渡します。
            'mask': union_mask.astype(np.uint8) * 255,
            'prompt': self.prompt_by_category[category_name],
            'category': category_name,
            'image_id': image_id,
        }


class Sam3DataCollator:
    def __init__(self, processor: Sam3Processor) -> None:
        self.processor = processor

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        encoded = self.processor(
            images=[sample['image'] for sample in samples],
            text=[sample['prompt'] for sample in samples],
            segmentation_maps=[sample['mask'] for sample in samples],
            return_tensors='pt',
        )
        labels = encoded['labels']

        if labels.ndim == 4:
            if labels.shape[1] != 1:
                raise ValueError(f'Unexpected labels shape: {labels.shape}')

            labels = labels[:, 0]

        # Pass only the data needed by SAM3
        batch = {
            'pixel_values': encoded['pixel_values'],
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }

        return batch


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

    loss = 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))

    return loss.mean()


class Sam3ForPromptSegmentation(nn.Module):
    """Add a segmentation loss to Sam3Model."""

    def __init__(
        self,
        sam3: nn.Module,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.sam3 = sam3
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [B, 1, Hmask, Wmask]
        logits = outputs.semantic_seg

        result = {'logits': logits}

        if labels is None:
            return result

        # Processor return the tensor with shape [B, Hmask, Wmask]
        targets = (labels > 0).float().unsqueeze(1)

        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(
                targets,
                size=logits.shape[-2:],
                mode='nearest',
            )

        # Calculate loss with float32 for numerical stability
        loss_logits = logits.float()
        targets = targets.float()

        focal = sigmoid_focal_loss(
            loss_logits,
            targets,
            alpha=0.25,
            gamma=2.0,
            reduction='mean',
        )

        dice = dice_loss(
            loss_logits,
            targets,
        )

        loss = self.focal_weight * focal + self.dice_weight * dice

        return {'loss': loss, 'logits': logits}


def preprocess_logits_for_metrics(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
    # Trainer may supply a tuple for models with multiple outputs.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if isinstance(logits, dict):
        logits = logits['logits']

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f'Unexpected logits type: {type(logits)}')

    return logits.float()


def find_lora_target_modules(model: nn.Module) -> list[str]:
    """Select attention projections outside the text encoder."""

    target_prefixes = (
        'vision_encoder.',
        'detr_encoder.',
        'mask_decoder.',
    )

    target_modules: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not name.startswith(target_prefixes):
            continue

        lower_name = name.lower()

        # Extract q_proj/v_proj and out_proj of DETR
        if 'attn' not in lower_name and 'attention' not in lower_name:
            continue

        target_modules.append(name)

    if not target_modules:
        raise RuntimeError(
            'No LoRA target modules were found. Inspect model.named_modules() for your Transformers version.'
        )

    return target_modules


def build_model(
    *,
    model_name: str,
    dtype: torch.dtype,
    rank: int,
) -> Sam3ForPromptSegmentation:
    base_model = Sam3Model.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    target_modules = find_lora_target_modules(base_model)

    print(f'LoRA target modules: {len(target_modules)}')
    for name in target_modules[:20]:
        print(f'  {name}')
    if len(target_modules) > 20:
        print('  ...')

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias='none',
        modules_to_save=['semantic_projection'],
    )

    peft_model = get_peft_model(
        base_model,
        lora_config,
    )
    peft_model.print_trainable_parameters()

    return Sam3ForPromptSegmentation(peft_model)


def compute_metrics(
    eval_prediction: EvalPrediction,
) -> dict[str, float]:
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    # Defensive handling for older/current Trainer behavior.
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    logits = torch.as_tensor(
        predictions,
        dtype=torch.float32,
    )
    targets = torch.as_tensor(labels)

    if logits.ndim != 4:
        raise ValueError(f'Expected logits [B, 1, H, W], got {logits.shape}')

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    elif targets.ndim != 4:
        raise ValueError(f'Unexpected target shape: {targets.shape}')

    targets = targets > 0

    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(
            targets.float(),
            size=logits.shape[-2:],
            mode='nearest',
        ).bool()

    predicted_masks = logits.sigmoid() > 0.5

    intersection = (predicted_masks & targets).flatten(1).sum(dim=1).float()

    union = (predicted_masks | targets).flatten(1).sum(dim=1).float()

    predicted_area = predicted_masks.flatten(1).sum(dim=1).float()
    target_area = targets.flatten(1).sum(dim=1).float()

    iou = intersection / union.clamp_min(1.0)

    dice = 2.0 * intersection / (predicted_area + target_area).clamp_min(1.0)

    return {
        'mean_iou': iou.mean().item(),
        'mean_dice': dice.mean().item(),
    }


def split_image_ids(
    image_ids: list[int],
    validation_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)

    num_validation = max(
        1,
        round(len(shuffled) * validation_ratio),
    )

    validation_ids = set(shuffled[:num_validation])
    train_ids = set(shuffled[num_validation:])

    if not train_ids:
        raise ValueError('The training split is empty.')

    return train_ids, validation_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco-json', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', type=str, default='facebook/sam3')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.coco_json.open('r', encoding='utf-8') as file:
        coco_data = json.load(file)

    print('COCO categories:')
    for category in coco_data['categories']:
        print(f'  id={category["id"]}: {category["name"]}')

    all_image_ids = [int(image['id']) for image in coco_data['images']]

    train_ids, validation_ids = split_image_ids(
        all_image_ids,
        validation_ratio=0.2,
        seed=args.seed,
    )

    print(f'Train images: {len(train_ids)}')
    print(f'Validation images: {len(validation_ids)}')

    processor = Sam3Processor.from_pretrained(args.model_name)

    train_dataset = CocoPromptSegmentationDataset(
        coco_data=coco_data,
        image_root=args.image_root,
        image_ids=train_ids,
        prompt_by_category=PROMPT_BY_CATEGORY,
    )
    validation_dataset = CocoPromptSegmentationDataset(
        coco_data=coco_data,
        image_root=args.image_root,
        image_ids=validation_ids,
        prompt_by_category=PROMPT_BY_CATEGORY,
    )

    print(f'Train image-prompt pairs: {len(train_dataset)}')
    print(f'Validation image-prompt pairs: {len(validation_dataset)}')

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = build_model(
        model_name=args.model_name,
        dtype=model_dtype,
        rank=args.lora_rank,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / 'trainer'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=0.1,
        lr_scheduler_type='cosine',
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        eval_strategy='epoch',
        save_strategy='no',
        logging_steps=5,
        report_to=['no'],
        remove_unused_columns=False,
        label_names=['labels'],
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=Sam3DataCollator(processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print(final_metrics)

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

    adapter_dir = args.output_dir / 'adapter'
    unwrapped_model.sam3.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    split_info = {
        'train_image_ids': sorted(train_ids),
        'validation_image_ids': sorted(validation_ids),
        'prompts': PROMPT_BY_CATEGORY,
    }

    with (args.output_dir / 'split.json').open(
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(split_info, file, indent=2)

    print(f'Adapter saved to: {adapter_dir}')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
