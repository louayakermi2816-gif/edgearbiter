"""Imagenette data and the image preprocessing shared by the benchmark, fine-tuning and the API.

Keeping one definition of the evaluation transform guarantees that a model is scored
the same way everywhere (resize to img/0.875, centre crop img, ImageNet normalisation).
"""
from __future__ import annotations

from typing import Optional

import torch

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Imagenette classes -> ImageNet-1k indices (tench, springer, cassette player, chain saw,
# church, French horn, garbage truck, gas pump, golf ball, parachute)
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
IMAGENETTE_CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn",
                      "garbage truck", "gas pump", "golf ball", "parachute"]


def class_names(imagenet_head: bool = True):
    """Label names of a model's output indices (ImageNet-1k, or the 10 Imagenette classes)."""
    if not imagenet_head:
        return IMAGENETTE_CLASSES
    from torchvision.models import ResNet18_Weights   # category list ships with torchvision, no download
    return ResNet18_Weights.DEFAULT.meta["categories"]


def eval_transform(img: int = 224):
    import torchvision.transforms as T
    return T.Compose([T.Resize(round(img / 0.875)), T.CenterCrop(img), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(img: int = 128):
    import torchvision.transforms as T
    return T.Compose([T.RandomResizedCrop(img, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])


def imagenette_size_for(img: int) -> str:
    """Smallest Imagenette release whose short side is at least the evaluation resize."""
    return "160px" if round(img / 0.875) <= 160 else "320px"


def imagenette(root: str = "data", split: str = "val", img: int = 224, size: Optional[str] = None,
               download: bool = True, transform=None):
    from torchvision.datasets import Imagenette
    size = size or imagenette_size_for(img)
    return Imagenette(root=root, split=split, size=size, download=download,
                      transform=transform or eval_transform(img))


def loader(dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       generator=g, persistent_workers=num_workers > 0)
