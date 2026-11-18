"""
Dataset loading, stratified splitting, and augmentation for Weather Image Classification.
Multi-class Weather Dataset: cloudy, rain, shine, sunrise.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ["cloudy", "rain", "shine", "sunrise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Standard ImageNet normalization for EfficientNet-B0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_weather_transforms(split: str = "train", img_size: int = 224):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def build_or_load_splits(
    data_dir: Path,
    manifest_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Build or load stratified train/val/test split to guarantee zero data leakage.
    """
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            print(f"Loaded existing split manifest from {manifest_path}:")
            print(f"  Train: {len(manifest['train'])}, Val: {len(manifest['val'])}, Test: {len(manifest['test'])}")
            return manifest

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_samples = []

    for cname in CLASS_NAMES:
        cdir = data_dir / cname
        if not cdir.exists():
            continue
        cidx = CLASS_TO_IDX[cname]
        for f in cdir.glob("*.*"):
            if f.suffix.lower() in valid_exts:
                all_samples.append((str(f.resolve()), cidx))

    if not all_samples:
        raise FileNotFoundError(f"No valid weather images found in {data_dir} across classes: {CLASS_NAMES}")

    paths, labels = zip(*all_samples)
    paths, labels = list(paths), list(labels)

    # First split train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, vt_paths, train_labels, vt_labels = train_test_split(
        paths, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )

    # Then split val vs test
    val_rel_ratio = val_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        vt_paths, vt_labels, test_size=(1.0 - val_rel_ratio), random_state=seed, stratify=vt_labels
    )

    manifest = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
        "class_names": CLASS_NAMES,
        "seed": seed
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created new stratified split manifest at {manifest_path}:")
    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)} (Total: {len(paths)})")

    return manifest


class WeatherDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def get_weather_dataloaders(
    data_dir: Path,
    manifest_path: Path,
    batch_size: int = 32,
    num_workers: int = 0
):
    splits = build_or_load_splits(data_dir, manifest_path)

    train_ds = WeatherDataset(splits["train"], transform=get_weather_transforms("train"))
    val_ds = WeatherDataset(splits["val"], transform=get_weather_transforms("val"))
    test_ds = WeatherDataset(splits["test"], transform=get_weather_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, splits
