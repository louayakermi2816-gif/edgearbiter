"""Fine-tune MobileNetV3-Small on Imagenette.

    python scripts/finetune_imagenette.py --epochs 5 --batch-size 64
    python scripts/finetune_imagenette.py --from-scratch --epochs 20 --lr 3e-3   # random init, no weight download
    python scripts/finetune_imagenette.py --report-only models/mobilenetv3_small_imagenette_scratch.json

Checkpoints and metrics go to models/ (git-ignored); the summary to results/finetune_<init>.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from edgearbiter.finetune import FinetuneConfig, build_loaders, build_model, checkpoint_path, report, train


def write_results(metrics: dict, out: str) -> Path:
    path = Path(out) / f"finetune_{metrics['init']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report(metrics), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--freeze-features", action="store_true")
    ap.add_argument("--from-scratch", action="store_true", help="random initialisation instead of ImageNet weights")
    ap.add_argument("--workers", type=int, default=0, help="data-loading worker processes")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--report-only", metavar="METRICS_JSON", help="only (re)write the results summary of a finished run")
    args = ap.parse_args()
    if args.report_only:
        print(f"wrote {write_results(json.loads(Path(args.report_only).read_text(encoding='utf-8')), args.results)}")
        return
    cfg = FinetuneConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, img=args.img,
                         freeze_features=args.freeze_features, pretrained=not args.from_scratch,
                         num_workers=args.workers, threads=args.threads, seed=args.seed,
                         data_root=args.data_root, out_dir=args.out)
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(pretrained=cfg.pretrained, freeze_features=cfg.freeze_features)
    metrics = train(model, train_loader, val_loader, cfg)
    print(f"best val top-1 ({cfg.init} init): {100 * metrics['best_val_top1']:.2f}%  -> {checkpoint_path(cfg.out_dir, cfg.init)}")
    print(f"summary: {write_results(metrics, args.results)}")


if __name__ == "__main__":
    main()
