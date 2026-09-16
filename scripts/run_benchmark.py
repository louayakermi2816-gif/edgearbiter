"""Benchmark the candidate models and (re)build results/cost_table.json.

    python scripts/run_benchmark.py                       # device role, with Imagenette accuracy if downloadable
    python scripts/run_benchmark.py --role edge           # run on the edge host, then merge with --merge
    python scripts/run_benchmark.py --no-data --iters 20  # latency/size only (quick)
    python scripts/run_benchmark.py --merge results/benchmark_device.json results/benchmark_edge.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgearbiter.benchmark import TIERS, benchmark_models, build_cost_table, imagenette_loader

DEFAULT_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v3_large_int8", "resnet18", "resnet18_int8",
                  "efficientnet_b0", "ssdlite320_mobilenet_v3_large"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["device", "edge"], default="device")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-data", action="store_true", help="skip Imagenette accuracy")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--merge", nargs=2, metavar=("DEVICE_JSON", "EDGE_JSON"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    if args.merge:
        dev = json.loads(Path(args.merge[0]).read_text()); edge = json.loads(Path(args.merge[1]).read_text())
        table = build_cost_table(dev, edge)
    else:
        loader = None if args.no_data else imagenette_loader()
        rows = benchmark_models(args.models, role=args.role, iters=args.iters, threads=args.threads, loader=loader, max_batches=args.max_batches)
        (out / f"benchmark_{args.role}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        lines = ["| model | int8 | p50 ms | p95 ms | size MB | top-1 (Imagenette) |", "|---|---|---|---|---|---|"]
        for r in rows:
            acc = "—" if r["accuracy"] is None else f"{100 * r['accuracy']:.1f}%"
            lines.append(f"| {r['model']} | {'yes' if r['int8'] else 'no'} | {r['latency_ms']['p50']:.1f} | {r['latency_ms']['p95']:.1f} | {r['size_mb']} | {acc} |")
        (out / f"benchmark_{args.role}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        if args.role != "device":
            return
        missing = [m for m in TIERS.values() if m not in {r["model"] for r in rows}]
        if missing:
            print(f"cost table not built: tier models missing from the run: {missing}"); return
        table = build_cost_table(rows)
    (out / "cost_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\ncost table written to {out}/cost_table.json")


if __name__ == "__main__":
    main()
