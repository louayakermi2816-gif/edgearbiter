"""Benchmark torchvision classifiers / detectors (FP32 and INT8) and build the cost table.

For each model we measure:
    latency_ms   p50 / p95 over N single-image inferences (after warm-up), batch size 1
    size_mb      serialised state_dict size (a proxy for memory footprint)
    accuracy     top-1 on the Imagenette validation split when the dataset is available
                 (ImageNet-pretrained heads are evaluated through the Imagenette->ImageNet
                 class-index mapping, so no fine-tuning is needed for the benchmark)

The "device" latency is what we measure on this machine; the "edge" latency is
measured on the edge host when the script runs there (`--role edge`). Both are
merged into results/cost_table.json which drives the simulator.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

# Imagenette classes -> ImageNet-1k indices (tench, springer, cassette player, chain saw,
# church, French horn, garbage truck, gas pump, golf ball, parachute)
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

# tier name -> model key; edit to change which models play which role in the simulator
TIERS = {"tiny": "mobilenet_v3_small", "small": "mobilenet_v3_large_int8", "medium": "resnet18"}
ENERGY_MJ_PER_MS = 0.45     # crude device energy model: mJ per ms of CPU inference (calibrate with a power meter)


def model_zoo() -> Dict[str, Callable[[], torch.nn.Module]]:
    """Lazily constructed pretrained models (weights download on first use)."""
    import torchvision.models as tvm
    from torchvision.models import quantization as tvq

    zoo: Dict[str, Callable[[], torch.nn.Module]] = {
        "mobilenet_v3_small": lambda: tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT),
        "mobilenet_v3_large": lambda: tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT),
        "resnet18": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT),
        "efficientnet_b0": lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT),
        # INT8 (post-training quantised) variants shipped by torchvision
        "mobilenet_v3_large_int8": lambda: tvq.mobilenet_v3_large(weights=tvq.MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True),
        "resnet18_int8": lambda: tvq.resnet18(weights=tvq.ResNet18_QuantizedWeights.DEFAULT, quantize=True),
        # detector (latency only unless a COCO evaluation is wired in)
        "ssdlite320_mobilenet_v3_large": lambda: tvm.detection.ssdlite320_mobilenet_v3_large(weights=tvm.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    }
    return zoo


def set_quant_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    for e in ("fbgemm", "x86", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


@torch.no_grad()
def measure_latency(model: torch.nn.Module, example: torch.Tensor, iters: int = 50, warmup: int = 10, threads: Optional[int] = None) -> Dict[str, float]:
    if threads:
        torch.set_num_threads(threads)
    model.eval()
    for _ in range(warmup):
        model(example)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter(); model(example); times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean())}


def state_dict_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def imagenette_accuracy(model: torch.nn.Module, loader, max_batches: Optional[int] = None) -> float:
    """Top-1 accuracy of an ImageNet-1k classifier on Imagenette via class-index mapping."""
    model.eval(); correct = total = 0
    idx = torch.tensor(IMAGENETTE_TO_IMAGENET)
    for b, (x, y) in enumerate(loader):
        logits = model(x)[:, idx]                 # restrict to the 10 Imagenette classes
        correct += int((logits.argmax(1) == y).sum()); total += len(y)
        if max_batches and b + 1 >= max_batches:
            break
    return correct / max(1, total)


def imagenette_loader(root: str = "data", size: str = "160px", batch_size: int = 32, download: bool = True):
    """Validation loader for Imagenette; returns None if the dataset cannot be obtained."""
    try:
        import torchvision.transforms as T
        from torchvision.datasets import Imagenette
        tf = T.Compose([T.Resize(160), T.CenterCrop(128), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ds = Imagenette(root=root, split="val", size=size, download=download, transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as exc:
        print(f"[benchmark] Imagenette unavailable ({exc}); accuracy will be null")
        return None


def benchmark_models(names: List[str], role: str = "device", iters: int = 50, threads: Optional[int] = None,
                     loader=None, max_batches: Optional[int] = None) -> List[Dict]:
    zoo = model_zoo(); engine = set_quant_engine()
    rows = []
    for name in names:
        print(f"[benchmark] {name} ...")
        model = zoo[name]()
        example = [torch.rand(3, 320, 320)] if "ssdlite" in name else torch.rand(1, 3, 224, 224)
        lat = measure_latency(model, example, iters=iters, threads=threads)
        acc = None
        if loader is not None and "ssdlite" not in name:
            acc = imagenette_accuracy(model, loader, max_batches)
        rows.append({"model": name, "role": role, "int8": name.endswith("_int8"), "quant_engine": engine if name.endswith("_int8") else None,
                     "latency_ms": lat, "size_mb": round(state_dict_size_mb(model), 2), "accuracy": acc, "threads": threads or torch.get_num_threads()})
        print(f"    p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms  size {rows[-1]['size_mb']} MB  acc {acc}")
    return rows


def build_cost_table(device_rows: List[Dict], edge_rows: Optional[List[Dict]] = None, tiers: Dict[str, str] = TIERS) -> Dict:
    """Merge device (and optionally edge) measurements into the simulator's cost table."""
    by_name = {r["model"]: r for r in device_rows}
    edge_by_name = {r["model"]: r for r in (edge_rows or [])}
    out = []
    for tier, model in tiers.items():
        d = by_name[model]
        e = edge_by_name.get(model)
        dev_lat = d["latency_ms"]["p50"]
        edge_lat = e["latency_ms"]["p50"] if e else dev_lat / 8.0      # no edge run yet: assume an 8x faster server
        out.append({"name": tier, "model": model, "accuracy": d["accuracy"] if d["accuracy"] is not None else None,
                    "latency_ms": {"device": round(dev_lat, 2), "edge": round(edge_lat, 2)},
                    "energy_mj": round(dev_lat * ENERGY_MJ_PER_MS, 2), "size_mb": d["size_mb"],
                    "edge_measured": e is not None})
    return {"tiers": out, "frame_kb": 40.0, "tx_energy_mj_per_kb": 0.25, "source": "benchmark"}
