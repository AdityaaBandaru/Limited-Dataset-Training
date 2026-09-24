"""Chest X-ray dataset with deterministic, class-balanced subsampling.

Expects the layout produced by scripts/download_data.py:

    <root>/<split>/NORMAL/*.jpeg
    <root>/<split>/PNEUMONIA/*.jpeg

Labels: NORMAL = 0, PNEUMONIA = 1 (the positive class).
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ("NORMAL", "PNEUMONIA")
IMG_EXTS = {".jpeg", ".jpg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_samples(split_dir: str | Path) -> list[tuple[Path, int]]:
    split_dir = Path(split_dir)
    samples = []
    for label, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}. Run scripts/download_data.py first.")
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
        )
        samples.extend((p, label) for p in files)
    return samples


def balanced_subsample(
    samples: list[tuple[Path, int]], max_per_class: int | None, seed: int
) -> list[tuple[Path, int]]:
    """Randomly keep at most `max_per_class` samples of each class (deterministic given seed)."""
    if max_per_class is None or max_per_class <= 0:
        return list(samples)
    rng = random.Random(seed)
    out = []
    for label in range(len(CLASSES)):
        cls_samples = [s for s in samples if s[1] == label]
        rng.shuffle(cls_samples)
        if len(cls_samples) < max_per_class:
            print(f"[warn] class {CLASSES[label]} has only {len(cls_samples)} samples (< {max_per_class})")
        out.extend(cls_samples[:max_per_class])
    return out


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        max_samples_per_class: int | None = None,
        seed: int = 42,
        transform=None,
    ):
        self.samples = balanced_subsample(list_samples(split_dir), max_samples_per_class, seed)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("L")
            if self.transform is not None:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts
