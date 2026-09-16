"""Fine-tune MobileNetV3-Small on Imagenette (10 classes) with a plain PyTorch training loop.

Data loaders with augmentation, AdamW + cosine schedule, per-epoch validation,
best-checkpoint saving and a metrics JSON. This is the supervised deep-learning
part of the project; the resulting model can be benchmarked like any other tier.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FinetuneConfig:
    data_root: str = "data"
    size: str = "160px"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    img: int = 128
    num_workers: int = 0
    freeze_features: bool = False
    seed: int = 0
    out_dir: str = "models"
    max_batches: Optional[int] = None      # for smoke tests


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_loaders(cfg: FinetuneConfig, download: bool = True):
    import torchvision.transforms as T
    from torchvision.datasets import Imagenette
    train_tf = T.Compose([T.RandomResizedCrop(cfg.img, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    val_tf = T.Compose([T.Resize(int(cfg.img * 1.15)), T.CenterCrop(cfg.img), T.ToTensor(), T.Normalize(MEAN, STD)])
    train = Imagenette(cfg.data_root, split="train", size=cfg.size, download=download, transform=train_tf)
    val = Imagenette(cfg.data_root, split="val", size=cfg.size, download=False, transform=val_tf)
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)
    return mk(train, True), mk(val, False)


def build_model(n_classes: int = 10, pretrained: bool = True, freeze_features: bool = False) -> nn.Module:
    import torchvision.models as tvm
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)
    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, max_batches: Optional[int] = None) -> float:
    model.eval(); correct = total = 0
    for b, (x, y) in enumerate(loader):
        correct += int((model(x).argmax(1) == y).sum()); total += len(y)
        if max_batches and b + 1 >= max_batches:
            break
    return correct / max(1, total)


def train(model: nn.Module, train_loader, val_loader, cfg: FinetuneConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = cfg.max_batches or len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * steps_per_epoch))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    best, history = 0.0, []
    for epoch in range(cfg.epochs):
        model.train(); t0 = time.time(); running = 0.0; n = 0
        for b, (x, y) in enumerate(train_loader):
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += float(loss) * len(y); n += len(y)
            if cfg.max_batches and b + 1 >= cfg.max_batches:
                break
        acc = evaluate(model, val_loader, cfg.max_batches)
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, n), "val_top1": acc, "seconds": round(time.time() - t0, 1)})
        if log:
            log(history[-1])
        if acc >= best:
            best = acc; torch.save(model.state_dict(), out / "mobilenetv3_small_imagenette.pt")
    metrics = {"best_val_top1": best, "history": history, "config": asdict(cfg)}
    (out / "mobilenetv3_small_imagenette.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics
