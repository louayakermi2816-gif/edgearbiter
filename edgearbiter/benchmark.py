"""Benchmark torchvision classifiers / detectors (FP32 and INT8) and build the cost table.

For each model we measure:
    latency_ms   p50 / p95 / mean over N single-image inferences (after warm-up), batch size 1,
                 at the model's input resolution, for each requested thread count
    size_mb      serialised state_dict size (a proxy for memory footprint)
    accuracy     10-way top-1 on the Imagenette validation split when pretrained weights and the
                 dataset are available (ImageNet-pretrained heads are restricted to the 10
                 Imagenette classes through the class-index mapping, so no fine-tuning is needed)

Weights
    pretrained   torchvision ImageNet weights (downloaded from download.pytorch.org on first use)
    random       the same architectures with random initialisation. Latency and size depend on the
                 architecture, not on the weight values (dense kernels do not exploit them), so a
                 random-weight run gives valid latency/size numbers when the weights cannot be
                 downloaded - but no accuracy. The detector's post-processing (NMS) is data
                 dependent, so its random-weight latency is only indicative.

INT8 models are torchvision's eager-mode quantised variants; each one runs on the quantisation
engine its weights were calibrated for (qnnpack for MobileNetV3, fbgemm for ResNet-18).

The "device" latency is what we measure on this machine; the "edge" latency is measured on the
edge host when the script runs there (`--role edge`). Both are merged into
results/cost_table.json which drives the simulator.
"""
from __future__ import annotations

import io
import os
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .data import IMAGENETTE_TO_IMAGENET, imagenette, loader as make_loader

# tier name -> model key; edit to change which models play which role in the simulator
TIERS = {"tiny": "mobilenet_v3_small", "small": "mobilenet_v3_large_int8", "medium": "resnet18"}
ENERGY_MJ_PER_MS = 0.45     # crude device energy model: mJ per ms of CPU inference (calibrate with a power meter)


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[bool], torch.nn.Module]   # build(pretrained) -> model
    img: int = 224                             # input resolution for latency and accuracy
    detector: bool = False                     # detectors are benchmarked for latency only
    backend: Optional[str] = None              # quantisation engine of an INT8 model
    imagenet_head: bool = True                 # False: native 10-class Imagenette head (no class mapping)


def _weights(enum, pretrained: bool):
    return enum.DEFAULT if pretrained else None


def _imagenette_model(models_dir: str, init: str) -> ModelSpec:
    """MobileNetV3-Small trained by `edgearbiter.finetune` (10-class head, its own input size)."""
    from .finetune import build_model, checkpoint_path, read_metrics

    ckpt = checkpoint_path(models_dir, init)
    img = int((read_metrics(ckpt) or {}).get("config", {}).get("img", 128))

    def build(pretrained: bool) -> torch.nn.Module:
        model = build_model(pretrained=False)
        if pretrained:
            if not ckpt.exists():
                raise FileNotFoundError(f"{ckpt} not found: run scripts/finetune_imagenette.py first")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return model
    return ModelSpec(build, img=img, imagenet_head=False)


def model_zoo(models_dir: str = "models") -> Dict[str, ModelSpec]:
    """Lazily constructed models (pretrained weights download on first use)."""
    import torchvision.models as tvm
    from torchvision.models import quantization as tvq

    qmnv3, qr18 = tvq.MobileNet_V3_Large_QuantizedWeights, tvq.ResNet18_QuantizedWeights
    ssd = tvm.detection.SSDLite320_MobileNet_V3_Large_Weights
    return {
        "mobilenet_v3_small": ModelSpec(lambda p: tvm.mobilenet_v3_small(weights=_weights(tvm.MobileNet_V3_Small_Weights, p))),
        "mobilenet_v3_large": ModelSpec(lambda p: tvm.mobilenet_v3_large(weights=_weights(tvm.MobileNet_V3_Large_Weights, p))),
        "resnet18": ModelSpec(lambda p: tvm.resnet18(weights=_weights(tvm.ResNet18_Weights, p))),
        "efficientnet_b0": ModelSpec(lambda p: tvm.efficientnet_b0(weights=_weights(tvm.EfficientNet_B0_Weights, p))),
        # INT8 (quantised) variants shipped by torchvision
        "mobilenet_v3_large_int8": ModelSpec(lambda p: tvq.mobilenet_v3_large(weights=_weights(qmnv3, p), quantize=True),
                                             backend=qmnv3.DEFAULT.meta["backend"]),
        "resnet18_int8": ModelSpec(lambda p: tvq.resnet18(weights=_weights(qr18, p), quantize=True),
                                   backend=qr18.DEFAULT.meta["backend"]),
        # detector (latency only unless a COCO evaluation is wired in)
        "ssdlite320_mobilenet_v3_large": ModelSpec(lambda p: tvm.detection.ssdlite320_mobilenet_v3_large(
            weights=_weights(ssd, p), weights_backbone=None), img=320, detector=True),
        # models trained on Imagenette by scripts/finetune_imagenette.py
        "mobilenet_v3_small_imagenette": _imagenette_model(models_dir, "imagenet"),
        "mobilenet_v3_small_imagenette_scratch": _imagenette_model(models_dir, "scratch"),
    }


def set_quant_engine(preferred: Optional[str] = None) -> str:
    engines = torch.backends.quantized.supported_engines
    for e in ((preferred,) if preferred else ()) + ("x86", "fbgemm", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def build(spec: ModelSpec, pretrained: bool) -> torch.nn.Module:
    """Build a model on the quantisation engine it was calibrated for, in eval mode."""
    if spec.backend:
        set_quant_engine(spec.backend)
    # eager-mode quantisation is deprecated upstream (still shipped in torch 2.14); random-init INT8
    # models also convert observers that saw no data. Neither affects the latency measurement.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*(torch\.ao\.quantization is deprecated|must run observer"
                                                  r"|quantized tensor creation functions)")
        return spec.build(pretrained).eval()


def example_input(spec: ModelSpec):
    return [torch.rand(3, spec.img, spec.img)] if spec.detector else torch.rand(1, 3, spec.img, spec.img)


@torch.no_grad()
def measure_latency(model: torch.nn.Module, example, iters: int = 50, warmup: int = 10, threads: Optional[int] = None) -> Dict[str, float]:
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
def imagenette_accuracy(model: torch.nn.Module, loader, max_batches: Optional[int] = None, imagenet_head: bool = True) -> float:
    """10-way top-1 on Imagenette. ImageNet-1k heads are restricted to the 10 Imagenette classes."""
    model.eval(); correct = total = 0
    idx = torch.tensor(IMAGENETTE_TO_IMAGENET)
    for b, (x, y) in enumerate(loader):
        logits = model(x)
        if imagenet_head:
            logits = logits[:, idx]
        correct += int((logits.argmax(1) == y).sum()); total += len(y)
        if max_batches and b + 1 >= max_batches:
            break
    return correct / max(1, total)


def imagenette_loader(root: str = "data", img: int = 224, batch_size: int = 32, download: bool = True):
    """Validation loader for Imagenette at the given resolution; None if the dataset cannot be obtained."""
    try:
        return make_loader(imagenette(root, "val", img=img, download=download), batch_size=batch_size)
    except Exception as exc:
        print(f"[benchmark] Imagenette unavailable ({exc}); accuracy will be null")
        return None


def host_info(label: Optional[str] = None) -> Dict:
    import torchvision
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"label": label, "cpu": cpu or platform.machine(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def benchmark_models(names: List[str], role: str = "device", iters: int = 50, threads: Sequence[Optional[int]] = (None,),
                     pretrained: bool = True, data_root: Optional[str] = None, max_batches: Optional[int] = None,
                     models_dir: str = "models") -> List[Dict]:
    """One row per (model, thread count). Accuracy needs pretrained weights and `data_root`."""
    zoo = model_zoo(models_dir); loaders: Dict[int, object] = {}
    rows = []
    for name in names:
        spec = zoo[name]
        print(f"[benchmark] {name} ({'pretrained' if pretrained else 'random-init'}, {spec.img}px) ...")
        model = build(spec, pretrained)
        engine = torch.backends.quantized.engine if spec.backend else None
        acc = None
        if pretrained and data_root and not spec.detector:
            if spec.img not in loaders:
                loaders[spec.img] = imagenette_loader(data_root, spec.img)
            if loaders[spec.img] is not None:
                acc = imagenette_accuracy(model, loaders[spec.img], max_batches, spec.imagenet_head)
        size = round(state_dict_size_mb(model), 2)
        for n_threads in threads:
            lat = measure_latency(model, example_input(spec), iters=iters, threads=n_threads)
            rows.append({"model": name, "role": role, "int8": spec.backend is not None, "quant_engine": engine,
                         "weights": "pretrained" if pretrained else "random", "img": spec.img,
                         "threads": n_threads or torch.get_num_threads(), "latency_ms": lat, "size_mb": size,
                         "accuracy": acc, "accuracy_metric": "imagenette_10way_top1" if acc is not None else None})
            print(f"    threads {rows[-1]['threads']}: p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms  size {size} MB  acc {acc}")
    return rows


def markdown_table(rows: List[Dict], host: Optional[Dict] = None) -> str:
    lines = []
    if host:
        weights = sorted({r["weights"] for r in rows})
        lines += [f"Host: {host.get('label') or host['cpu']} ({host['cpu']}, {host['logical_cpus']} logical CPUs) · "
                  f"torch {host['torch']} · weights: {', '.join(weights)} · {host['date_utc']}", ""]
        if "random" in weights:
            lines += ["Random-init weights: latency and size are valid, accuracy is not measured.", ""]
    lines += ["| model | int8 (engine) | input | threads | p50 ms | p95 ms | size MB | top-1 (Imagenette, 10-way) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = "—" if r["accuracy"] is None else f"{100 * r['accuracy']:.1f}%"
        q = f"yes ({r['quant_engine']})" if r["int8"] else "no"
        lines.append(f"| {r['model']} | {q} | {r['img']}px | {r['threads']} | {r['latency_ms']['p50']:.1f} | "
                     f"{r['latency_ms']['p95']:.1f} | {r['size_mb']} | {acc} |")
    return "\n".join(lines) + "\n"


def _pick(rows: List[Dict], model: str) -> Optional[Dict]:
    """The row of `model` with the most threads (a device uses all its cores)."""
    rows = [r for r in rows if r["model"] == model]
    return max(rows, key=lambda r: r.get("threads") or 0) if rows else None


def build_cost_table(device_rows: List[Dict], edge_rows: Optional[List[Dict]] = None, tiers: Dict[str, str] = TIERS) -> Dict:
    """Merge device (and optionally edge) measurements into the simulator's cost table."""
    out = []
    for tier, model in tiers.items():
        d = _pick(device_rows, model)
        if d is None:
            raise KeyError(f"tier model {model!r} missing from the device benchmark")
        e = _pick(edge_rows or [], model)
        dev_lat = d["latency_ms"]["p50"]
        edge_lat = e["latency_ms"]["p50"] if e else dev_lat / 8.0      # no edge run yet: assume an 8x faster server
        out.append({"name": tier, "model": model, "accuracy": d["accuracy"],
                    "latency_ms": {"device": round(dev_lat, 2), "edge": round(edge_lat, 2)},
                    "energy_mj": round(dev_lat * ENERGY_MJ_PER_MS, 2), "size_mb": d["size_mb"],
                    "edge_measured": e is not None})
    return {"tiers": out, "frame_kb": 40.0, "tx_energy_mj_per_kb": 0.25, "source": "benchmark"}
