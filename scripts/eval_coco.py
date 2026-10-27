"""COCO val2017 box mAP of the SSDLite detector, FP32 and post-training INT8 (results/coco_map.{json,md}).

    python scripts/run_benchmark.py --weights pretrained --models ssdlite320_mobilenet_v3_large_int8 ...
    python scripts/eval_coco.py                      # COCO val2017 (~1 GB) downloads on first use
    python scripts/eval_coco.py --max-images 500     # quicker, noisier

The INT8 model is the one the benchmark calibrated and saved (models/ssdlite320_mobilenet_v3_large_int8.pt),
restored with `benchmark.load_model`; the FP32 model is torchvision's COCO checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgearbiter.benchmark import host_info, load_model  # noqa: E402
from edgearbiter.coco_eval import coco_map, coco_val  # noqa: E402

DETECTORS = ["ssdlite320_mobilenet_v3_large", "ssdlite320_mobilenet_v3_large_int8"]


def markdown(rows, host) -> str:
    lines = [f"COCO val2017 box mAP · {host['cpu']}, torch {host['torch']} · {host['date_utc']}", "",
             "| model | precision | mAP@[.5:.95] | mAP@.5 | mAP@.75 | images | evaluation time |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {100 * r['map']:.1f} | {100 * r['map50']:.1f} | "
                     f"{100 * r['map75']:.1f} | {r['images']} | {r['seconds'] / 60:.1f} min |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=DETECTORS)
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    ann, images = coco_val(args.coco_root)
    rows = []
    for name in args.models:
        print(f"[coco] {name}", flush=True)
        result = coco_map(load_model(name), ann, images, args.max_images)
        rows.append({"model": name, "precision": "INT8 (x86, PTQ)" if name.endswith("_int8") else "FP32", **result})
        print(f"    mAP {100 * result['map']:.1f}  mAP@.5 {100 * result['map50']:.1f}  ({result['images']} images, "
              f"{result['seconds']:.0f} s)", flush=True)
    host = host_info()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "coco_map.json").write_text(json.dumps({"host": host, "rows": rows}, indent=1), encoding="utf-8")
    (out / "coco_map.md").write_text(markdown(rows, host), encoding="utf-8")
    print(markdown(rows, host))


if __name__ == "__main__":
    main()
