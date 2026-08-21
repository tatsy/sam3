from __future__ import annotations

import os
import re
import json
import random
import argparse
from typing import Any, TypedDict
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn
from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from transformers import (
    Trainer,
    Sam3Model,
    Sam3Processor,
    EvalPrediction,
    TrainingArguments,
)
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import Dataset


PromptMap = dict[str, tuple[str, ...]]


def load_prompt_map(config_path: Path) -> PromptMap:
    with config_path.open(
        'r',
        encoding='utf-8',
    ) as file:
        config = yaml.safe_load(file) or {}

    label_configs = config.get('labels')

    if not isinstance(label_configs, list):
        raise ValueError(f'No labels were found in {config_path}')

    prompt_map: PromptMap = {}

    for label_config in label_configs:
        if not isinstance(label_config, dict):
            raise TypeError('Each labels entry must be a mapping.')

        name = label_config.get('name')
        prompts = label_config.get('prompts')

        if not isinstance(name, str) or not name:
            raise ValueError(f'Invalid category name: {name!r}')

        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f'No prompts were specified for {name!r}')

        valid_prompts = tuple(prompt for prompt in prompts if isinstance(prompt, str) and prompt)

        if not valid_prompts:
            raise ValueError(f'No valid prompts were specified for {name!r}')

        if name in prompt_map:
            raise ValueError(f'Duplicate category in config: {name}')

        prompt_map[name] = valid_prompts

    return prompt_map


def extract_dataset_name(file_name: str) -> str:
    match = re.fullmatch(
        r'(.+)_\d{4}_\d+\.[^.]+',
        Path(file_name).name,
    )

    if match is None:
        raise ValueError(f'Could not infer dataset name from {file_name}')

    return match.group(1)


def split_image_ids_stratified(
    images: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    image_ids_by_dataset: dict[str, list[int]] = defaultdict(list)

    for image in images:
        dataset_name = extract_dataset_name(str(image['file_name']))
        image_ids_by_dataset[dataset_name].append(int(image['id']))

    train_ids: set[int] = set()
    validation_ids: set[int] = set()

    for dataset_name, image_ids in sorted(image_ids_by_dataset.items()):
        shuffled = list(image_ids)

        # datasetごとに違うが再現可能なseed
        dataset_seed = seed + sum(ord(char) for char in dataset_name)
        random.Random(dataset_seed).shuffle(shuffled)

        num_validation = max(
            1,
            round(len(shuffled) * validation_ratio),
        )

        validation_ids.update(shuffled[:num_validation])
        train_ids.update(shuffled[num_validation:])

        print(f'{dataset_name}: train={len(shuffled) - num_validation}, validation={num_validation}')

    if not train_ids or not validation_ids:
        raise ValueError('Train or validation split is empty.')

    return train_ids, validation_ids


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


class Sam3SegmentationAugment:
    def __init__(self) -> None:
        self.geometry_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=(-5.0, 5.0),
                            translate=(0.03, 0.03),
                            scale=(0.95, 1.05),
                            interpolation=v2.InterpolationMode.BILINEAR,
                            fill={
                                tv_tensors.Image: (0, 0, 0),
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )

        self.image_transform = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.08,
                            hue=0.02,
                        ),
                    ],
                    p=0.8,
                ),
                v2.RandomApply(
                    [
                        v2.GaussianBlur(
                            kernel_size=3,
                            sigma=(0.1, 1.0),
                        ),
                    ],
                    p=0.1,
                ),
            ]
        )

    def __call__(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        image_tensor = tv_tensors.Image(image)
        mask_tensor = tv_tensors.Mask(torch.as_tensor(mask.astype(np.uint8)))

        image_tensor, mask_tensor = self.geometry_transform(image_tensor, mask_tensor)
        image_tensor = self.image_transform(image_tensor)

        augmented_image = TF.to_pil_image(image_tensor)
        augmented_mask = mask_tensor.as_subclass(torch.Tensor).cpu().numpy() > 0

        return augmented_image, augmented_mask


class CocoPromptSegmentationDataset(Dataset):
    """Convert COCO instances to image-prompt semantic masks."""

    def __init__(
        self,
        *,
        coco_data: dict[str, Any],
        image_root: Path,
        image_ids: set[int],
        prompt_by_category: PromptMap,
        augment: Sam3SegmentationAugment | None = None,
        negative_ratio: float = 0.0,
        include_all_negatives: bool = False,
        randomize_prompts: bool = False,
        seed: int = 42,
    ) -> None:
        if negative_ratio < 0:
            raise ValueError(f'negative_ratio must be >= 0, got {negative_ratio}')

        self.image_root = image_root
        self.prompt_by_category = prompt_by_category
        self.augment = augment
        self.randomize_prompts = randomize_prompts

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

        positive_samples: list[tuple[int, int]] = []
        negative_candidates: list[tuple[int, int]] = []
        for image_id in sorted(self.images_by_id):
            for category_id, category_name in self.categories_by_id.items():
                if category_name not in prompt_by_category:
                    continue

                sample = (image_id, category_id)
                has_annotations = bool(self.annotations_by_key.get((image_id, category_id)))

                if has_annotations:
                    positive_samples.append(sample)
                else:
                    negative_candidates.append(sample)

        rng = random.Random(seed)

        if include_all_negatives:
            selected_negatives = negative_candidates
        else:
            num_negatives = min(len(negative_candidates), round(len(positive_samples) * negative_ratio))
            selected_negatives = rng.sample(negative_candidates, num_negatives)

        self.samples = positive_samples + selected_negatives
        rng.shuffle(self.samples)

        if not self.samples:
            raise RuntimeError('No samples were created. Check COCO categories and YAML config file.')

        print(f'positive={len(positive_samples)}, negative={len(selected_negatives)}')

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

        if self.augment is not None:
            image, union_mask = self.augment(image, union_mask)

        prompts = self.prompt_by_category[category_name]
        if self.randomize_prompts:
            prompt = random.choice(prompts)
        else:
            prompt = prompts[0]

        return {
            'image': image,
            'mask': union_mask.astype(np.uint8) * 255,
            'prompt': prompt,
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


def find_lora_target_modules(
    model: nn.Module,
    vision_last_n_layers: int | None = 8,
) -> list[str]:
    """Select attention projections outside the text encoder."""

    allowed_projections = {'q_proj', 'v_proj'}
    vision_prefix = 'vision_encoder.backbone.layers.'
    detr_prefix = 'detr_encoder.layers.'

    num_vision_layers = len(model.vision_encoder.backbone.layers)  # type: ignore
    if vision_last_n_layers is None:
        first_vision_layer = 0
    else:
        first_vision_layer = max(0, num_vision_layers - vision_last_n_layers)

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        project_name = name.rsplit('.', maxsplit=1)[-1]
        if project_name not in allowed_projections:
            continue

        # Vision tower
        if name.startswith(vision_prefix):
            remainder = name[len(vision_prefix) :]
            layer_index_text = remainder.split('.', maxsplit=1)[0]
            if not layer_index_text.isdigit():
                continue

            layer_index = int(layer_index_text)
            if layer_index >= first_vision_layer:
                target_modules.append(name)

            continue

        # DETR encoder
        if name.startswith(detr_prefix):
            target_modules.append(name)
            continue

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
    vision_last_n_layers: int | None = 8,
) -> Sam3ForPromptSegmentation:
    base_model = Sam3Model.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    target_modules = find_lora_target_modules(
        base_model,
        vision_last_n_layers=vision_last_n_layers,
    )

    vision_targets = [name for name in target_modules if name.startswith('vision_encoder.')]
    detr_targets = [name for name in target_modules if name.startswith('detr_encoder.')]

    print(f'  Vision encoder targets: {len(vision_targets)}')
    print(f'  DETR encoder targets: {len(detr_targets)}')

    for name in target_modules:
        print(f'  {name}')

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

    if targets.shape[1] != 1:
        raise ValueError(f'Expected one mask channel, got {targets.shape}')

    targets = targets > 0

    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(
            targets.float(),
            size=logits.shape[-2:],
            mode='nearest',
        ).bool()

    probabilities = logits.sigmoid()
    predicted_masks = probabilities > 0.5

    intersection = (predicted_masks & targets).flatten(1).sum(dim=1).float()

    union = (predicted_masks | targets).flatten(1).sum(dim=1).float()

    predicted_area = predicted_masks.flatten(1).sum(dim=1).float()
    target_area = targets.flatten(1).sum(dim=1).float()

    num_pixels = predicted_masks[0].numel()

    positive_indices = target_area > 0
    negative_indices = target_area == 0

    metrics: dict[str, float] = {}

    # Positive evaluation
    if positive_indices.any():
        positive_iou = intersection[positive_indices] / union[positive_indices].clamp_min(1.0)

        positive_dice = (
            2.0
            * intersection[positive_indices]
            / (predicted_area[positive_indices] + target_area[positive_indices]).clamp_min(1.0)
        )

        metrics['positive_mean_iou'] = positive_iou.mean().item()
        metrics['positive_mean_dice'] = positive_dice.mean().item()
    else:
        metrics['positive_mean_iou'] = 0.0
        metrics['positive_mean_dice'] = 0.0

    # Negative evaluation
    if negative_indices.any():
        negative_predicted_area = predicted_area[negative_indices]

        fp_pixel_rate = (negative_predicted_area / num_pixels).mean()
        fp_image_rate = (negative_predicted_area / num_pixels > 0.001).float().mean()
        negative_mean_probability = probabilities[negative_indices].mean()

        metrics['negative_fp_pixel_rate'] = fp_pixel_rate.item()
        metrics['negative_fp_image_rate'] = fp_image_rate.item()
        metrics['negative_mean_probability'] = negative_mean_probability.item()
    else:
        metrics['negative_fp_pixel_rate'] = 0.0
        metrics['negative_fp_image_rate'] = 0.0
        metrics['negative_mean_probability'] = 0.0

    metrics['balanced_score'] = 0.5 * (metrics['positive_mean_iou'] + 1.0 - metrics['negative_fp_image_rate'])

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco-json', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', type=str, default='facebook/sam3')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--config', type=Path, default=Path('configs/all_categories.yml'))

    return parser.parse_args()


def initialize_distributed_if_needed() -> bool:
    local_rank_value = os.environ.get('LOCAL_RANK')

    if local_rank_value is None:
        return False

    local_rank = int(local_rank_value)

    if local_rank < 0:
        return False

    if not torch.cuda.is_available():
        raise RuntimeError('Distributed NCCL training requires CUDA.')

    device = torch.device(
        'cuda',
        local_rank,
    )

    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            device_id=device,
        )
        return True

    return False


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initialize_distributed_if_needed()

    # Load COCO JSON / category prompts and check their correspondence
    with args.coco_json.open('r', encoding='utf-8') as file:
        coco_data = json.load(file)

    prompt_by_category = load_prompt_map(args.config)

    coco_category_names = {str(category['name']) for category in coco_data['categories']}

    configured_category_names = set(prompt_by_category)

    missing_in_config = coco_category_names - configured_category_names

    missing_in_coco = configured_category_names - coco_category_names

    if missing_in_config:
        raise ValueError(f'COCO categories missing from config: {sorted(missing_in_config)}')

    if missing_in_coco:
        raise ValueError(f'Config categories missing from COCO: {sorted(missing_in_coco)}')

    train_ids, validation_ids = split_image_ids_stratified(
        coco_data['images'],
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
        prompt_by_category=prompt_by_category,
        augment=Sam3SegmentationAugment(),
        negative_ratio=0.5,
        include_all_negatives=False,
        randomize_prompts=True,
        seed=args.seed,
    )
    validation_dataset = CocoPromptSegmentationDataset(
        coco_data=coco_data,
        image_root=args.image_root,
        image_ids=validation_ids,
        prompt_by_category=prompt_by_category,
        augment=None,
        negative_ratio=0.0,
        include_all_negatives=True,
        randomize_prompts=False,
        seed=args.seed,
    )

    print(f'Train image-prompt pairs: {len(train_dataset)}')
    print(f'Validation image-prompt pairs: {len(validation_dataset)}')

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = build_model(
        model_name=args.model_name,
        dtype=model_dtype,
        rank=args.lora_rank,
        vision_last_n_layers=8,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / 'trainer'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=1.0e-4,
        warmup_steps=0.1,
        lr_scheduler_type='cosine',
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        logging_strategy='steps',
        logging_steps=10,
        report_to=['none'],
        remove_unused_columns=False,
        label_names=['labels'],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model='balanced_score',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=Sam3DataCollator(processor),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        print(final_metrics)

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        adapter_dir = args.output_dir / 'adapter'
        unwrapped_model.sam3.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)

        split_info = {
            'train_image_ids': sorted(train_ids),
            'validation_image_ids': sorted(validation_ids),
            'prompts': {category: list(prompts) for category, prompts in prompt_by_category.items()},
        }

        with (args.output_dir / 'split.json').open('w', encoding='utf-8') as file:
            json.dump(split_info, file, indent=2)

        print(f'Adapter saved to: {adapter_dir}')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
