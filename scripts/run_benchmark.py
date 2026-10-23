"""Benchmark the candidate models and (re)build results/cost_table.json.

    python scripts/run_benchmark.py                       # device role, with Imagenette accuracy if downloadable
    python scripts/run_benchmark.py --role edge           # run on the edge host, then merge with --merge
    python scripts/run_benchmark.py --no-data --iters 20  # latency/size only (quick)
    python scripts/run_benchmark.py --weights random      # no weight download: latency/size only
    python scripts/run_benchmark.py --threads 1 4         # one row per thread count
    python scripts/run_benchmark.py --models mobilenet_v3_small_imagenette_scratch --tag imagenette   # trained model
    python scripts/run_benchmark.py --merge results/benchmark_device.json results/benchmark_edge.json

No device at hand: emulate it with one thread in a CPU-limited cgroup (Linux, root) and use this
machine as the edge (each model at its fastest thread count), then merge. The Imagenette models are
trained by finetune_imagenette.py; `--tiers deployment` picks FP32 or INT8 per architecture
(benchmark.deployment_tiers):
    M="mobilenet_v3_small_imagenette_scratch mobilenet_v3_small_imagenette_scratch_int8 ..."
    python scripts/run_benchmark.py --role device --threads 1 --cpu-quota 0.25 --iters 200 --models $M --tag imagenette
    python scripts/run_benchmark.py --role edge --threads 1 2 4 --iters 200 --models $M --tag imagenette
    python scripts/run_benchmark.py --tiers deployment --merge results/benchmark_device_imagenette.json \
        results/benchmark_edge_imagenette.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgearbiter.benchmark import (TIER_SETS, benchmark_models, build_cost_table, deployment_tiers, host_info,
                                   markdown_table)

DEFAULT_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v3_large_int8", "resnet18", "resnet18_int8",
                  "efficientnet_b0", "ssdlite320_mobilenet_v3_large"]


def load_run(path: str):
    """Rows and host of a saved benchmark run."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return (data["rows"], data.get("host")) if isinstance(data, dict) else (data, None)


def tier_models(name: str, device_rows):
    if name == "deployment":
        tiers, reasons = deployment_tiers(device_rows)
        print("tiers (deployment rule):\n  " + "\n  ".join(reasons))
        return tiers, reasons
    return TIER_SETS[name], f"fixed tier set {name!r}"


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
    ap.add_argument("--cpu-quota", type=float, default=None,
                    help="emulate a slower device: limit the inference process to this many CPUs (cgroup)")
    ap.add_argument("--tiers", choices=[*TIER_SETS, "deployment"], default="torchvision",
                    help="models that form the cost table (deployment: Imagenette models, FP32 or INT8 by rule)")
    ap.add_argument("--label", default=None, help="free-text host label stored with the results")
    ap.add_argument("--tag", default=None, help="suffix of the output files: benchmark_<role>_<tag>.*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--merge", nargs=2, metavar=("DEVICE_JSON", "EDGE_JSON"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        (dev_rows, dev_host), (edge_rows, edge_host) = load_run(args.merge[0]), load_run(args.merge[1])
        tiers, selection = tier_models(args.tiers, dev_rows)
        table = build_cost_table(dev_rows, edge_rows, tiers, meta={
            "device": {"benchmark": args.merge[0], "host": dev_host,
                       "cpu_quota": sorted({r.get("cpu_quota") for r in dev_rows if r.get("cpu_quota")})},
            "edge": {"benchmark": args.merge[1], "host": edge_host}, "tier_selection": selection})
    else:
        host = host_info(args.label)
        rows = benchmark_models(args.models, role=args.role, iters=args.iters, threads=args.threads,
                                pretrained=args.weights == "pretrained", data_root=None if args.no_data else args.data_root,
                                max_batches=args.max_batches, cpu_quota=args.cpu_quota)
        stem = f"benchmark_{args.role}" + (f"_{args.tag}" if args.tag else "")
        (out / f"{stem}.json").write_text(json.dumps({"host": host, "rows": rows}, indent=1), encoding="utf-8")
        md = markdown_table(rows, host)
        (out / f"{stem}.md").write_text(md, encoding="utf-8")
        print(md)
        if args.role != "device" or args.tag:
            return
        tiers, selection = tier_models(args.tiers, rows)
        missing = [m for m in tiers.values() if m not in {r["model"] for r in rows}]
        if missing:
            print(f"cost table not built: tier models missing from the run: {missing}"); return
        table = build_cost_table(rows, tiers=tiers, meta={"device": {"host": host}, "tier_selection": selection})
    (out / "cost_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\ncost table written to {out}/cost_table.json")


if __name__ == "__main__":
    main()
