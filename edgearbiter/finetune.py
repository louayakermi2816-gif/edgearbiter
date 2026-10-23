"""Fine-tune MobileNetV3-Small on Imagenette (10 classes) with a plain PyTorch training loop.

Data loaders with augmentation, AdamW + one-cycle schedule, per-epoch validation,
best-checkpoint saving and a metrics JSON. This is the supervised deep-learning
part of the project; the resulting model can be benchmarked like any other tier
(`mobilenet_v3_small_imagenette` in `edgearbiter.benchmark.model_zoo`).

`pretrained=False` trains the same network from random initialisation. It is the
transfer-learning control (ImageNet init vs. scratch) and needs no weight download.
The two runs write separate checkpoints:
    models/mobilenetv3_small_imagenette.pt          ImageNet init (fine-tuning)
    models/mobilenetv3_small_imagenette_scratch.pt  random init
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .data import eval_transform, imagenette, loader, train_transform


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
    pretrained: bool = True                # False: random initialisation (train from scratch)
    threads: Optional[int] = None          # torch intra-op threads (None = torch default)
    seed: int = 0
    out_dir: str = "models"
    max_batches: Optional[int] = None      # for smoke tests

    @property
    def init(self) -> str:
        return "imagenet" if self.pretrained else "scratch"


def checkpoint_path(out_dir: str = "models", init: str = "imagenet") -> Path:
    return Path(out_dir) / ("mobilenetv3_small_imagenette" + ("" if init == "imagenet" else f"_{init}") + ".pt")


def read_metrics(ckpt: Path) -> Optional[dict]:
    path = Path(ckpt).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def report(metrics: dict) -> str:
    """Markdown summary of a training run (per-epoch loss / validation accuracy)."""
    cfg = metrics["config"]
    lines = [f"MobileNetV3-Small on Imagenette ({cfg['size']} release, {cfg['img']}px input), **{metrics['init']}** "
             f"initialisation · {cfg['epochs']} epochs · batch {cfg['batch_size']} · AdamW, one-cycle lr {cfg['lr']} · "
             f"best validation top-1 (10 classes, {cfg['img']}px centre crop): **{100 * metrics['best_val_top1']:.1f}%** · "
             f"{metrics['train_seconds'] / 60:.0f} min on CPU ({cfg['threads'] or 'default'} threads)", "",
             "| epoch | train loss | val top-1 | seconds |", "|---|---|---|---|"]
    lines += [f"| {h['epoch']} | {h['train_loss']:.3f} | {100 * h['val_top1']:.1f}% | {h['seconds']} |" for h in metrics["history"]]
    return "\n".join(lines) + "\n"


def build_loaders(cfg: FinetuneConfig, download: bool = True):
    train = imagenette(cfg.data_root, "train", size=cfg.size, download=download, transform=train_transform(cfg.img))
    val = imagenette(cfg.data_root, "val", size=cfg.size, download=False, transform=eval_transform(cfg.img))
    return (loader(train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, seed=cfg.seed),
            loader(val, cfg.batch_size, num_workers=cfg.num_workers))


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
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = min(cfg.max_batches or len(train_loader), len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * steps_per_epoch))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt = checkpoint_path(cfg.out_dir, cfg.init); ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, history, t_start = 0.0, [], time.time()
    for epoch in range(cfg.epochs):
        model.train(); t0 = time.time(); running = 0.0; n = 0
        for b, (x, y) in enumerate(train_loader):
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item() * len(y); n += len(y)
            if cfg.max_batches and b + 1 >= cfg.max_batches:
                break
        acc = evaluate(model, val_loader, cfg.max_batches)
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, n), "val_top1": acc, "seconds": round(time.time() - t0, 1)})
        if log:
            log(history[-1])
        if acc >= best:
            best = acc; torch.save(model.state_dict(), ckpt)
    metrics = {"init": cfg.init, "best_val_top1": best, "train_seconds": round(time.time() - t_start, 1),
               "history": history, "config": asdict(cfg)}
    ckpt.with_suffix(".json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics
