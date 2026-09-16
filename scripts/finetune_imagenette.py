"""Fine-tune MobileNetV3-Small on Imagenette.

    python scripts/finetune_imagenette.py --epochs 5 --batch-size 64
"""
from __future__ import annotations

import argparse
import sys

from edgearbiter.finetune import FinetuneConfig, build_loaders, build_model, train


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--freeze-features", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    cfg = FinetuneConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, img=args.img,
                         freeze_features=args.freeze_features, data_root=args.data_root, out_dir=args.out)
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(freeze_features=cfg.freeze_features)
    metrics = train(model, train_loader, val_loader, cfg)
    print(f"best val top-1: {100 * metrics['best_val_top1']:.2f}%  -> {args.out}/mobilenetv3_small_imagenette.pt")


if __name__ == "__main__":
    main()
