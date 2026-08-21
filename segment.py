from __future__ import annotations

import csv
import json
import random
import shutil
import zipfile
import argparse
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from pycocotools import mask as mask_utils
from transformers import Sam3Model, Sam3Processor


OVERLAY_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (106, 76, 147),
    (42, 157, 143),
    (244, 162, 97),
    (233, 196, 106),
]


@dataclass(frozen=True)
class PromptSpec:
    """SAM v3 prompt and filtering configuration."""

    label: str
    prompt: str
    confidence_threshold: float
    mask_threshold: float
    min_area_ratio: float
    max_area_ratio: float


@dataclass(frozen=True)
class EvaluationData:
    """COCO ground truth indexed for semantic-segmentation evaluation."""

    image_by_id: dict[int, dict[str, Any]]
    category_id_by_label: dict[str, int]
    annotations_by_key: dict[tuple[int, int], list[dict[str, Any]]]
    selected_image_ids: set[int]


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'all_categories.yml'


def load_prompt_specs(config_path: Path) -> list[PromptSpec]:
    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}

    label_configs = config.get('labels')
    if not isinstance(label_configs, list) or not label_configs:
        raise ValueError(f'No labels were found in config: {config_path}')

    prompt_specs: list[PromptSpec] = []

    for label_config in label_configs:
        if not isinstance(label_config, dict):
            raise TypeError('Each labels entry in the config must be a mapping.')

        label = label_config.get('name')
        prompts = label_config.get('prompts')

        if not isinstance(label, str) or not label:
            raise ValueError("Each labels entry must define a non-empty 'name'.")

        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"Label '{label}' must define at least one prompt in 'prompts'.")

        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Label '{label}' contains an invalid prompt: {prompt!r}")

            confidence_threshold = float(label_config.get('confidence_threshold', 0.0))
            mask_threshold = float(label_config.get('mask_threshold', 0.5))
            min_area_ratio = float(label_config['min_area_ratio'])
            max_area_ratio = float(label_config['max_area_ratio'])

            prompt_specs.append(
                PromptSpec(
                    label=label,
                    prompt=prompt,
                    confidence_threshold=confidence_threshold,
                    mask_threshold=mask_threshold,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                )
            )

    return prompt_specs


def validate_prompts_against_split(
    prompt_specs: list[PromptSpec],
    split_json: Path,
) -> None:
    """Ensure evaluation uses the same labels and prompts as fine-tuning."""

    with split_json.open('r', encoding='utf-8') as file:
        split_data = json.load(file)

    saved_prompts = split_data.get('prompts')
    if not isinstance(saved_prompts, dict):
        raise ValueError(f"No 'prompts' mapping was found in {split_json}")

    configured_prompts: dict[str, list[str]] = {}
    for spec in prompt_specs:
        configured_prompts.setdefault(spec.label, []).append(spec.prompt)

    normalized_saved_prompts: dict[str, list[str]] = {}
    for label, prompts in saved_prompts.items():
        if isinstance(prompts, str):
            # Compatibility with split.json files produced by older finetune.py.
            normalized_saved_prompts[str(label)] = [prompts]
        elif isinstance(prompts, list) and all(
            isinstance(prompt, str) and prompt for prompt in prompts
        ):
            normalized_saved_prompts[str(label)] = prompts
        else:
            raise ValueError(
                f'Invalid prompts for label {label!r} in {split_json}: {prompts!r}'
            )

    configured_labels = set(configured_prompts)
    saved_labels = set(normalized_saved_prompts)
    if configured_labels != saved_labels:
        raise ValueError(
            'Config labels do not match the fine-tuning split. '
            f'Missing from config: {sorted(saved_labels - configured_labels)}; '
            f'not used in fine-tuning: {sorted(configured_labels - saved_labels)}'
        )

    mismatched_prompts = {
        label: {
            'fine_tuning': normalized_saved_prompts[label],
            'segmentation': configured_prompts[label],
        }
        for label in sorted(configured_labels)
        if normalized_saved_prompts[label] != configured_prompts[label]
    }
    if mismatched_prompts:
        raise ValueError(
            'Config prompts do not match the prompts used for fine-tuning: '
            f'{mismatched_prompts}'
        )


def find_images(input_dir: Path) -> list[Path]:
    extensions = {
        '.png',
        '.jpg',
        '.jpeg',
        '.bmp',
        '.tif',
        '.tiff',
    }

    images = sorted(path for path in input_dir.rglob('*') if path.is_file() and path.suffix.lower() in extensions)

    if not images:
        raise FileNotFoundError(f'No images were found under: {input_dir}')

    return images


def sample_images(
    image_paths: list[Path],
    num_images: int,
    seed: int,
) -> list[Path]:
    if len(image_paths) <= num_images:
        return image_paths

    rng = random.Random(seed)
    selected = rng.sample(image_paths, num_images)

    # COCO内の画像順を毎回固定します。
    return sorted(selected)


def resolve_input_image_path(
    input_dir: Path,
    file_name: str,
) -> Path:
    candidates = [
        input_dir / file_name,
        input_dir / Path(file_name).name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not locate image '{file_name}' under {input_dir}")


def decode_coco_segmentation(
    segmentation: Any,
    height: int,
    width: int,
) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get('counts'), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise TypeError(f'Unsupported segmentation type: {type(segmentation)}')

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)

    return decoded.astype(bool)


def load_evaluation_data(
    ground_truth_json: Path,
    labels: list[str],
    split_json: Path | None,
    eval_split: str,
) -> EvaluationData:
    with ground_truth_json.open('r', encoding='utf-8') as file:
        coco_data = json.load(file)

    image_by_id = {int(image['id']): image for image in coco_data['images']}
    category_id_by_label = {
        str(category['name']): int(category['id']) for category in coco_data['categories']
    }

    missing_labels = sorted(set(labels) - set(category_id_by_label))
    if missing_labels:
        raise ValueError(
            f'Labels from the config are missing in ground truth: {missing_labels}'
        )

    annotations_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for annotation in coco_data['annotations']:
        key = (int(annotation['image_id']), int(annotation['category_id']))
        annotations_by_key.setdefault(key, []).append(annotation)

    if split_json is None:
        selected_image_ids = set(image_by_id)
    else:
        with split_json.open('r', encoding='utf-8') as file:
            split_data = json.load(file)

        if eval_split == 'train':
            split_ids = split_data.get('train_image_ids')
        elif eval_split == 'validation':
            split_ids = split_data.get('validation_image_ids')
        else:
            split_ids = list(image_by_id)

        if not isinstance(split_ids, list):
            raise ValueError(
                f"Split '{eval_split}' was not found in {split_json}"
            )

        selected_image_ids = {int(image_id) for image_id in split_ids}
        unknown_ids = selected_image_ids - set(image_by_id)
        if unknown_ids:
            raise ValueError(
                f'Split contains image IDs absent from ground truth: {sorted(unknown_ids)}'
            )

    if not selected_image_ids:
        raise ValueError('No images were selected for evaluation.')

    return EvaluationData(
        image_by_id=image_by_id,
        category_id_by_label=category_id_by_label,
        annotations_by_key=annotations_by_key,
        selected_image_ids=selected_image_ids,
    )


def find_evaluation_images(
    input_dir: Path,
    evaluation_data: EvaluationData,
) -> tuple[list[Path], dict[Path, int]]:
    image_paths: list[Path] = []
    image_id_by_path: dict[Path, int] = {}

    for image_id in sorted(evaluation_data.selected_image_ids):
        image_info = evaluation_data.image_by_id[image_id]
        image_path = resolve_input_image_path(input_dir, str(image_info['file_name'])).resolve()

        if image_path in image_id_by_path:
            raise ValueError(f'Duplicate ground-truth image path: {image_path}')

        image_paths.append(image_path)
        image_id_by_path[image_path] = image_id

    return image_paths, image_id_by_path


def binary_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def mask_nms(
    candidates: list[dict[str, Any]],
    iou_threshold: float,
    max_instances: int,
) -> list[dict[str, Any]]:
    """Greedy NMS based on binary-mask IoU."""

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate['score'],
        reverse=True,
    )

    kept: list[dict[str, Any]] = []

    for candidate in candidates:
        overlaps_existing = any(
            binary_mask_iou(candidate['mask'], previous['mask']) >= iou_threshold for previous in kept
        )

        if overlaps_existing:
            continue

        kept.append(candidate)

        if len(kept) >= max_instances:
            break

    return kept


def merge_masks(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    merged_mask = np.logical_or.reduce([candidate['mask'] for candidate in candidates])
    merged_prompts = sorted({candidate['prompt'] for candidate in candidates})

    return {
        'mask': merged_mask,
        'score': max(float(candidate['score']) for candidate in candidates),
        'prompt': ' | '.join(merged_prompts),
    }


def make_masks_exclusive(
    merged_candidates_by_label: dict[str, dict[str, Any]],
    image_shape: tuple[int, int],
) -> list[tuple[str, dict[str, Any]]]:
    occupied_mask = np.zeros(image_shape, dtype=bool)
    exclusive_candidates: list[tuple[str, dict[str, Any]]] = []

    sorted_candidates = sorted(
        merged_candidates_by_label.items(),
        key=lambda item: item[1]['score'],
        reverse=True,
    )

    for label, candidate in sorted_candidates:
        exclusive_mask = np.logical_and(candidate['mask'], ~occupied_mask)
        if not exclusive_mask.any():
            continue

        occupied_mask = np.logical_or(occupied_mask, exclusive_mask)
        exclusive_candidates.append(
            (
                label,
                {
                    **candidate,
                    'mask': exclusive_mask,
                },
            )
        )

    return exclusive_candidates


def encode_coco_rle(
    binary_mask: np.ndarray,
) -> tuple[dict[str, Any], float, list[float]]:
    """Convert a binary mask to compressed COCO RLE."""

    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    raw_rle = mask_utils.encode(binary_mask)

    area = float(mask_utils.area(raw_rle))
    bbox = [float(value) for value in mask_utils.toBbox(raw_rle).tolist()]

    # pycocotools returns bytes, but JSON requires str.
    json_rle = {
        'size': [
            int(raw_rle['size'][0]),
            int(raw_rle['size'][1]),
        ],
        'counts': raw_rle['counts'].decode('ascii'),
    }

    return json_rle, area, bbox


def create_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        mode='w',
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in package_dir.rglob('*'):
            if path.is_file():
                archive.write(
                    path,
                    arcname=path.relative_to(package_dir),
                )


def save_overlay_preview(
    image: Image.Image,
    destination_path: Path,
    labels: list[str],
    masks_by_label: dict[str, list[np.ndarray]],
) -> None:
    base_image = image.convert('RGBA')
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    alpha = 96

    for index, label in enumerate(labels):
        color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]

        for mask in masks_by_label.get(label, []):
            overlay[mask] = (*color, alpha)

    preview = Image.alpha_composite(
        base_image,
        Image.fromarray(overlay, mode='RGBA'),
    )
    preview.convert('RGB').save(destination_path)


def save_binary_mask(mask: np.ndarray, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode='L').save(destination_path)


def make_ground_truth_mask(
    evaluation_data: EvaluationData,
    image_id: int,
    label: str,
    height: int,
    width: int,
) -> np.ndarray:
    category_id = evaluation_data.category_id_by_label[label]
    mask = np.zeros((height, width), dtype=bool)

    for annotation in evaluation_data.annotations_by_key.get(
        (image_id, category_id),
        [],
    ):
        mask |= decode_coco_segmentation(
            annotation['segmentation'],
            height=height,
            width=width,
        )

    return mask


def make_evaluation_row(
    *,
    image_id: int,
    file_name: str,
    label: str,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    mean_probability: float,
) -> dict[str, Any]:
    intersection = int(np.logical_and(predicted_mask, ground_truth_mask).sum())
    union = int(np.logical_or(predicted_mask, ground_truth_mask).sum())
    predicted_area = int(predicted_mask.sum())
    ground_truth_area = int(ground_truth_mask.sum())
    num_pixels = int(predicted_mask.size)
    is_positive = ground_truth_area > 0

    if is_positive:
        iou = intersection / max(union, 1)
        dice = 2.0 * intersection / max(predicted_area + ground_truth_area, 1)
    else:
        iou = None
        dice = None

    predicted_area_ratio = predicted_area / num_pixels

    return {
        'image_id': image_id,
        'file_name': file_name,
        'label': label,
        'is_positive': is_positive,
        'iou': iou,
        'dice': dice,
        'predicted_area': predicted_area,
        'ground_truth_area': ground_truth_area,
        'predicted_area_ratio': predicted_area_ratio,
        'false_positive_image': (not is_positive and predicted_area_ratio > 0.001),
        'mean_probability': mean_probability,
    }


def summarize_evaluation_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        positive_rows = [row for row in subset if row['is_positive']]
        negative_rows = [row for row in subset if not row['is_positive']]

        positive_mean_iou = (
            float(np.mean([row['iou'] for row in positive_rows])) if positive_rows else 0.0
        )
        positive_mean_dice = (
            float(np.mean([row['dice'] for row in positive_rows])) if positive_rows else 0.0
        )
        negative_fp_pixel_rate = (
            float(np.mean([row['predicted_area_ratio'] for row in negative_rows]))
            if negative_rows
            else 0.0
        )
        negative_fp_image_rate = (
            float(np.mean([row['false_positive_image'] for row in negative_rows]))
            if negative_rows
            else 0.0
        )
        negative_mean_probability = (
            float(np.mean([row['mean_probability'] for row in negative_rows]))
            if negative_rows
            else 0.0
        )

        return {
            'num_samples': len(subset),
            'num_positive': len(positive_rows),
            'num_negative': len(negative_rows),
            'positive_mean_iou': positive_mean_iou,
            'positive_mean_dice': positive_mean_dice,
            'negative_fp_pixel_rate': negative_fp_pixel_rate,
            'negative_fp_image_rate': negative_fp_image_rate,
            'negative_mean_probability': negative_mean_probability,
            'balanced_score': 0.5 * (
                positive_mean_iou + 1.0 - negative_fp_image_rate
            ),
        }

    labels = sorted({str(row['label']) for row in rows})

    return {
        'overall': summarize_subset(rows),
        'by_label': {
            label: summarize_subset([row for row in rows if row['label'] == label])
            for label in labels
        },
    }


def save_evaluation_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = summarize_evaluation_rows(rows)

    metrics_path = output_dir / 'evaluation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)

    rows_path = output_dir / 'evaluation_per_sample.csv'
    with rows_path.open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'image_id',
            'file_name',
            'label',
            'is_positive',
            'iou',
            'dice',
            'predicted_area',
            'ground_truth_area',
            'predicted_area_ratio',
            'false_positive_image',
            'mean_probability',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return metrics


def load_model(
    base_model_name: str,
    adapter_dir: Path | None,
    device: torch.device,
) -> tuple[
    Sam3Model | PeftModel,
    Sam3Processor,
    torch.dtype,
]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if adapter_dir is not None and not adapter_dir.is_dir():
        raise FileNotFoundError(f'LoRA adapter directory does not exist: {adapter_dir}')

    base_model = Sam3Model.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        attn_implementation='sdpa',
    )

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

        try:
            processor = Sam3Processor.from_pretrained(str(adapter_dir))
        except (OSError, ValueError):
            processor = Sam3Processor.from_pretrained(base_model_name)

        print(f'Using LoRA adapter: {adapter_dir}')
    else:
        model = base_model
        processor = Sam3Processor.from_pretrained(base_model_name)

        print('Using the pretrained SAM 3 without LoRA.')

    model = model.to(device).eval()

    return model, processor, dtype


@torch.inference_mode()
def generate_annotations(
    input_dir: Path,
    output_dir: Path,
    num_images: int,
    seed: int,
    overwrite: bool,
    config_path: Path,
    base_model_name: str,
    adapter_dir: Path | None,
    use_presence_filter: bool,
    make_exclusive: bool,
    ground_truth_json: Path | None,
    split_json: Path | None,
    eval_split: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this script.')

    prompt_specs = load_prompt_specs(config_path)
    labels = list(dict.fromkeys(spec.label for spec in prompt_specs))

    if split_json is not None:
        validate_prompts_against_split(prompt_specs, split_json)

    evaluation_data: EvaluationData | None = None
    evaluation_image_id_by_path: dict[Path, int] = {}

    if ground_truth_json is None:
        if split_json is not None:
            raise ValueError('--split-json requires --ground-truth-json.')
        all_images = find_images(input_dir)
    else:
        evaluation_data = load_evaluation_data(
            ground_truth_json=ground_truth_json,
            labels=labels,
            split_json=split_json,
            eval_split=eval_split,
        )
        all_images, evaluation_image_id_by_path = find_evaluation_images(
            input_dir=input_dir,
            evaluation_data=evaluation_data,
        )

    selected_images = sample_images(
        all_images,
        num_images=num_images,
        seed=seed,
    )

    package_dir = output_dir / 'cvat_coco'
    overlay_output_dir = output_dir / 'overlay_previews'
    semantic_mask_output_dir = output_dir / 'semantic_masks'
    image_output_dir = package_dir / 'images' / 'default'
    annotation_output_dir = package_dir / 'annotations'

    removable_dirs = [package_dir, overlay_output_dir, semantic_mask_output_dir]
    for directory in removable_dirs:
        if directory.exists():
            if not overwrite:
                raise FileExistsError(f'{directory} already exists. Use --overwrite to replace it.')
            shutil.rmtree(directory)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    annotation_output_dir.mkdir(parents=True, exist_ok=True)
    overlay_output_dir.mkdir(parents=True, exist_ok=True)
    semantic_mask_output_dir.mkdir(parents=True, exist_ok=True)

    category_id_by_label = {label: index + 1 for index, label in enumerate(labels)}
    categories = [
        {
            'id': category_id,
            'name': label,
            'supercategory': 'component',
        }
        for label, category_id in category_id_by_label.items()
    ]

    device = torch.device('cuda')
    model, processor, autocast_dtype = load_model(
        base_model_name=base_model_name,
        adapter_dir=adapter_dir,
        device=device,
    )

    if isinstance(model, PeftModel):
        sam3_model = model.get_base_model()
    else:
        sam3_model = model

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    annotation_id = 1

    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
        for image_id, source_path in enumerate(
            tqdm(selected_images, desc='Processing images'),
            start=1,
        ):
            destination_name = f'{image_id:04d}_{source_path.name}'
            destination_path = image_output_dir / destination_name
            shutil.copy2(source_path, destination_path)

            image = Image.open(source_path).convert('RGB')
            width, height = image.size
            image_area = width * height

            coco_images.append(
                {
                    'id': image_id,
                    'file_name': destination_name,
                    'width': width,
                    'height': height,
                }
            )
            selected_rows.append(
                {
                    'image_id': image_id,
                    'source_path': str(source_path),
                    'cvat_file_name': destination_name,
                }
            )

            # Reuse the LoRA-adapted vision features for all text prompts.
            image_inputs = processor(
                images=image,
                return_tensors='pt',
            ).to(device)
            vision_embeds = sam3_model.get_vision_features(
                pixel_values=image_inputs['pixel_values'],
            )

            candidates_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
            preview_masks_by_label: dict[str, list[np.ndarray]] = {label: [] for label in labels}
            raw_masks_by_label = {
                label: np.zeros((height, width), dtype=bool) for label in labels
            }
            raw_probabilities_by_label = {
                label: np.zeros((height, width), dtype=np.float32) for label in labels
            }

            for spec in prompt_specs:
                text_inputs = processor(
                    text=spec.prompt,
                    return_tensors='pt',
                ).to(device)

                outputs = model(
                    vision_embeds=vision_embeds,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs.get('attention_mask'),
                )

                semantic_probability = F.interpolate(
                    outputs.semantic_seg.float(),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                ).sigmoid()[0, 0]

                probability = semantic_probability.detach().cpu().numpy()
                mask = probability > spec.mask_threshold

                raw_masks_by_label[spec.label] |= mask
                np.maximum(
                    raw_probabilities_by_label[spec.label],
                    probability,
                    out=raw_probabilities_by_label[spec.label],
                )

                presence_score = float(outputs.presence_logits.float().sigmoid().reshape(-1).max().item())

                area = int(mask.sum())
                area_ratio = area / image_area

                if area == 0:
                    continue
                if use_presence_filter and presence_score < spec.confidence_threshold:
                    continue
                if not spec.min_area_ratio <= area_ratio <= spec.max_area_ratio:
                    continue

                candidates_by_label[spec.label].append(
                    {
                        'mask': mask,
                        'score': presence_score,
                        'prompt': spec.prompt,
                    }
                )

            if evaluation_data is not None:
                ground_truth_image_id = evaluation_image_id_by_path[source_path.resolve()]

                for label in labels:
                    ground_truth_mask = make_ground_truth_mask(
                        evaluation_data=evaluation_data,
                        image_id=ground_truth_image_id,
                        label=label,
                        height=height,
                        width=width,
                    )
                    evaluation_rows.append(
                        make_evaluation_row(
                            image_id=ground_truth_image_id,
                            file_name=str(
                                evaluation_data.image_by_id[ground_truth_image_id]['file_name']
                            ),
                            label=label,
                            predicted_mask=raw_masks_by_label[label],
                            ground_truth_mask=ground_truth_mask,
                            mean_probability=float(raw_probabilities_by_label[label].mean()),
                        )
                    )

            # Each text prompt produces one semantic mask. Multiple prompts for
            # the same label are combined by union.
            merged_candidates_by_label: dict[str, dict[str, Any]] = {}
            for label, candidates in candidates_by_label.items():
                merged_candidate = merge_masks(candidates)
                if merged_candidate is not None:
                    merged_candidates_by_label[label] = merged_candidate

            if make_exclusive:
                final_candidates = make_masks_exclusive(
                    merged_candidates_by_label,
                    image_shape=(height, width),
                )
            else:
                final_candidates = list(merged_candidates_by_label.items())

            preview_name = f'{Path(destination_name).stem}.png'

            for label, merged_candidate in final_candidates:
                mask = merged_candidate['mask']
                preview_masks_by_label[label] = [mask]

                save_binary_mask(
                    mask,
                    semantic_mask_output_dir / label / preview_name,
                )

                segmentation, area, bbox = encode_coco_rle(mask)
                coco_annotations.append(
                    {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id_by_label[label],
                        'segmentation': segmentation,
                        'area': area,
                        'bbox': bbox,
                        'iscrowd': 1,
                        'score': merged_candidate['score'],
                    }
                )
                prediction_rows.append(
                    {
                        'annotation_id': annotation_id,
                        'image_id': image_id,
                        'file_name': destination_name,
                        'label': label,
                        'prompt': merged_candidate['prompt'],
                        'score': merged_candidate['score'],
                        'area': area,
                        'bbox_x': bbox[0],
                        'bbox_y': bbox[1],
                        'bbox_width': bbox[2],
                        'bbox_height': bbox[3],
                    }
                )
                annotation_id += 1

            save_overlay_preview(
                image=image,
                destination_path=overlay_output_dir / preview_name,
                labels=labels,
                masks_by_label=preview_masks_by_label,
            )

            del vision_embeds, image_inputs

    coco_data = {
        'info': {
            'description': (
                'SAM 3 semantic segmentation predictions'
                if adapter_dir is None
                else 'LoRA fine-tuned SAM 3 semantic segmentation predictions'
            ),
            'version': '1.0',
            'date_created': datetime.now(timezone.utc).isoformat(),
        },
        'licenses': [],
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': categories,
    }

    annotation_path = annotation_output_dir / 'instances_default.json'
    with annotation_path.open('w', encoding='utf-8') as file:
        json.dump(coco_data, file, indent=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / 'selected_images.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['image_id', 'source_path', 'cvat_file_name'],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    with (output_dir / 'sam3_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'annotation_id',
            'image_id',
            'file_name',
            'label',
            'prompt',
            'score',
            'area',
            'bbox_x',
            'bbox_y',
            'bbox_width',
            'bbox_height',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    evaluation_metrics: dict[str, Any] | None = None
    if evaluation_data is not None:
        evaluation_metrics = save_evaluation_results(
            output_dir=output_dir,
            rows=evaluation_rows,
        )

    zip_path = output_dir / 'sam3_lora_semantic_cvat.zip'
    create_zip(package_dir, zip_path)

    print()
    print(f'Selected images: {len(coco_images)}')
    print(f'Generated masks: {len(coco_annotations)}')
    print(f'COCO JSON: {annotation_path}')
    print(f'CVAT ZIP: {zip_path}')
    print(f'Binary masks: {semantic_mask_output_dir}')
    print(f'Overlay previews: {overlay_output_dir}')

    if evaluation_metrics is not None:
        overall = evaluation_metrics['overall']
        print()
        print('Raw semantic-mask evaluation (before area/presence/exclusive filters):')
        print(f"  Positive mean IoU: {overall['positive_mean_iou']:.6f}")
        print(f"  Positive mean Dice: {overall['positive_mean_dice']:.6f}")
        print(f"  Negative FP pixel rate: {overall['negative_fp_pixel_rate']:.6f}")
        print(f"  Negative FP image rate: {overall['negative_fp_image_rate']:.6f}")
        print(f"  Balanced score: {overall['balanced_score']:.6f}")
        print(f"Evaluation metrics: {output_dir / 'evaluation_metrics.json'}")
        print(f"Per-sample metrics: {output_dir / 'evaluation_per_sample.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        '-i',
        '--input-dir',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='facebook/sam3',
    )
    parser.add_argument(
        '--adapter-dir',
        type=Path,
        default=None,
    )
    parser.add_argument(
        '--ground-truth-json',
        type=Path,
        default=None,
        help=(
            'Optional COCO ground-truth JSON. When specified, raw semantic masks '
            'are evaluated in addition to generating annotation outputs.'
        ),
    )
    parser.add_argument(
        '--split-json',
        type=Path,
        default=None,
        help='Optional split.json saved by finetune.py.',
    )
    parser.add_argument(
        '--eval-split',
        choices=['train', 'validation', 'all'],
        default='validation',
        help='Image split to evaluate when --split-json is specified.',
    )
    parser.add_argument(
        '-n',
        '--num-images',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--use-presence-filter',
        action='store_true',
        help=(
            'Filter prompts using confidence_threshold and presence_logits. '
            'Leave disabled when the presence head was not trained.'
        ),
    )
    parser.add_argument(
        '--make-exclusive',
        action='store_true',
        help='Remove overlaps between labels in descending presence-score order.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    generate_annotations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
        overwrite=args.overwrite,
        config_path=args.config,
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        use_presence_filter=args.use_presence_filter,
        make_exclusive=args.make_exclusive,
        ground_truth_json=args.ground_truth_json,
        split_json=args.split_json,
        eval_split=args.eval_split,
    )


if __name__ == '__main__':
    main()
