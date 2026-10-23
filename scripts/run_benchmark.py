"""Benchmark the candidate models and (re)build results/cost_table.json.

    python scripts/run_benchmark.py                       # device role, with Imagenette accuracy if downloadable
    python scripts/run_benchmark.py --role edge           # run on the edge host, then merge with --merge
    python scripts/run_benchmark.py --no-data --iters 20  # latency/size only (quick)
    python scripts/run_benchmark.py --weights random      # no weight download: latency/size only
    python scripts/run_benchmark.py --threads 1 4         # one row per thread count
    python scripts/run_benchmark.py --models mobilenet_v3_small_imagenette_scratch --tag imagenette   # trained model
    python scripts/run_benchmark.py --merge results/benchmark_device.json results/benchmark_edge.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgearbiter.benchmark import TIERS, benchmark_models, build_cost_table, host_info, markdown_table

DEFAULT_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v3_large_int8", "resnet18", "resnet18_int8",
                  "efficientnet_b0", "ssdlite320_mobilenet_v3_large"]


def load_rows(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["rows"] if isinstance(data, dict) else data


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["device", "edge"], default="device")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--weights", choices=["pretrained", "random"], default="pretrained",
                    help="random: no download, latency/size only")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--threads", type=int, nargs="*", default=[None])
    ap.add_argument("--no-data", action="store_true", help="skip Imagenette accuracy")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--label", default=None, help="free-text host label stored with the results")
    ap.add_argument("--tag", default=None, help="suffix of the output files: benchmark_<role>_<tag>.*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--merge", nargs=2, metavar=("DEVICE_JSON", "EDGE_JSON"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        table = build_cost_table(load_rows(args.merge[0]), load_rows(args.merge[1]))
    else:
        host = host_info(args.label)
        rows = benchmark_models(args.models, role=args.role, iters=args.iters, threads=args.threads,
                                pretrained=args.weights == "pretrained", data_root=None if args.no_data else args.data_root,
                                max_batches=args.max_batches)
        stem = f"benchmark_{args.role}" + (f"_{args.tag}" if args.tag else "")
        (out / f"{stem}.json").write_text(json.dumps({"host": host, "rows": rows}, indent=1), encoding="utf-8")
        md = markdown_table(rows, host)
        (out / f"{stem}.md").write_text(md, encoding="utf-8")
        print(md)
        if args.role != "device" or args.tag:
            return
        missing = [m for m in TIERS.values() if m not in {r["model"] for r in rows}]
        if missing:
            print(f"cost table not built: tier models missing from the run: {missing}"); return
        table = build_cost_table(rows)
    (out / "cost_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\ncost table written to {out}/cost_table.json")


if __name__ == "__main__":
    main()
