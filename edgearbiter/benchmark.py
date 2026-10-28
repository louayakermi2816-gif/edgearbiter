"""Benchmark FP32 and INT8 classifiers / detectors and build the simulator's cost table.

For each model we measure:
    latency_ms   p50 / p95 / mean over N single-image inferences (after warm-up), batch size 1,
                 at the model's input resolution, for each requested thread count
    memory       what the deployed model needs, in a fresh process (`edgearbiter.bench_worker`):
                 its weights plus the resident memory that running it adds (activations,
                 workspace, kernel caches), peak over the timed inferences
    size_mb      serialised state_dict size (weights on disk)
    accuracy     top-1 on the Imagenette validation split: native 10-class heads for the models
                 trained by `edgearbiter.finetune`; ImageNet-pretrained heads restricted to the 10
                 Imagenette classes through the class-index mapping

Models
    torchvision zoo    MobileNetV3-Small/Large, ResNet-18, EfficientNet-B0, SSDLite; INT8 variants
                       shipped by torchvision (eager mode, on the engine their weights were calibrated
                       for: qnnpack for MobileNetV3, fbgemm for ResNet-18), and SSDLite with its backbone
                       and heads quantised after training (FX graph mode, `quantize.QuantizedSSD`).
                       `pretrained` downloads the ImageNet / COCO weights; `random` builds the same
                       architectures without download (latency / size / memory are valid, accuracy is not).
    Imagenette models  <arch>_imagenette[_scratch] trained by `edgearbiter.finetune`, and their
                       `_int8` variants: post-training static quantisation (FX graph mode, x86
                       engine) calibrated on Imagenette training images (`edgearbiter.quantize`);
                       `_int8qat`: quantisation-aware training instead (2 epochs, same INT8 kernels).

Hardware profiles (no device needed): the "edge" is this machine (each model at its fastest
measured thread count); the "device" is emulated by one thread in a CPU-bandwidth-limited cgroup
(`--cpu-quota 0.25`, `edgearbiter.cpulimit`). Both are merged into results/cost_table.json, which
drives the simulator: accuracy, p50 latency (device / edge), p95 latency (the simulator's latency
tail), energy per inference, size and memory per tier. `deployment_tiers` chooses the tiers among the
models trained on Imagenette: one per architecture, FP32 or INT8 by a fixed rule.
"""
from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import IMAGENETTE_TO_IMAGENET, eval_transform, imagenette, loader as make_loader

# tier name -> model key; edit to change which models play which role in the simulator
TIERS = {"tiny": "mobilenet_v3_small", "small": "mobilenet_v3_large_int8", "medium": "resnet18"}
TIER_SETS = {"torchvision": TIERS}
ENERGY_MJ_PER_MS = 0.45     # crude device energy model: mJ per ms of CPU inference (calibrate with a power meter)


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[bool], torch.nn.Module]   # build(pretrained) -> model (FP32 for `ptq` models)
    img: int = 224                             # input resolution for latency and accuracy
    detector: bool = False                     # detectors: latency and memory only (no COCO evaluation)
    backend: Optional[str] = None              # quantisation engine of an INT8 model
    imagenet_head: bool = True                 # False: native 10-class Imagenette head (no class mapping)
    ptq: bool = False                          # INT8 by post-training quantisation of `build(...)`'s model
    qat: bool = False                          # ... by quantisation-aware training instead (with `ptq`)


def _weights(enum, pretrained: bool):
    return enum.DEFAULT if pretrained else None


def _torchvision_int8(builder, weights, pretrained: bool) -> torch.nn.Module:
    """A torchvision INT8 model. Pretrained weights carry their engine. Random-init ones are built for the engine
    `build` selected on this CPU (torchvision's default, fbgemm, is x86-only: Apple silicon has qnnpack), and
    quantised here with that engine's default qconfig when torchvision leaves them in FP32: it sets a qconfig for
    fbgemm and qnnpack only, and PyTorch on Windows has neither (onednn)."""
    if pretrained:
        return builder(weights=weights.DEFAULT, quantize=True)
    engine = torch.backends.quantized.engine
    model = builder(weights=None, quantize=True, backend=engine)
    if any(isinstance(m, torch.ao.nn.quantized.Conv2d) for m in model.modules()):
        return model
    model = builder(weights=None, quantize=False).eval()
    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qconfig(engine)
    torch.ao.quantization.prepare(model, inplace=True)
    model(torch.rand(1, 3, 224, 224))
    return torch.ao.quantization.convert(model)


def _imagenette_model(models_dir: str, init: str, arch: str = "mobilenet_v3_small", int8: bool = False,
                      qat: bool = False) -> ModelSpec:
    """A model trained by `edgearbiter.finetune` (10-class head, its own input size), FP32 or INT8 (PTQ, or QAT)."""
    from .finetune import build_model, checkpoint_path, read_metrics

    ckpt = checkpoint_path(models_dir, init, arch)
    img = int((read_metrics(ckpt) or {}).get("config", {}).get("img", 128))

    def build(pretrained: bool) -> torch.nn.Module:
        model = build_model(pretrained=False, arch=arch)
        if pretrained:
            if not ckpt.exists():
                raise FileNotFoundError(f"{ckpt} not found: run scripts/finetune_imagenette.py --arch {arch} first")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return model
    return ModelSpec(build, img=img, imagenet_head=False, ptq=int8, qat=qat, backend="x86" if int8 else None)


def model_zoo(models_dir: str = "models") -> Dict[str, ModelSpec]:
    """Lazily constructed models (pretrained weights download on first use)."""
    import torchvision.models as tvm
    from torchvision.models import quantization as tvq
    from .finetune import ARCHS

    qmnv3, qr18 = tvq.MobileNet_V3_Large_QuantizedWeights, tvq.ResNet18_QuantizedWeights
    ssd = tvm.detection.SSDLite320_MobileNet_V3_Large_Weights

    def ssdlite(pretrained: bool) -> torch.nn.Module:
        return tvm.detection.ssdlite320_mobilenet_v3_large(weights=_weights(ssd, pretrained), weights_backbone=None)
    zoo = {
        "mobilenet_v3_small": ModelSpec(lambda p: tvm.mobilenet_v3_small(weights=_weights(tvm.MobileNet_V3_Small_Weights, p))),
        "mobilenet_v3_large": ModelSpec(lambda p: tvm.mobilenet_v3_large(weights=_weights(tvm.MobileNet_V3_Large_Weights, p))),
        "resnet18": ModelSpec(lambda p: tvm.resnet18(weights=_weights(tvm.ResNet18_Weights, p))),
        "efficientnet_b0": ModelSpec(lambda p: tvm.efficientnet_b0(weights=_weights(tvm.EfficientNet_B0_Weights, p))),
        # INT8 (quantised) variants shipped by torchvision
        "mobilenet_v3_large_int8": ModelSpec(lambda p: _torchvision_int8(tvq.mobilenet_v3_large, qmnv3, p),
                                             backend=qmnv3.DEFAULT.meta["backend"]),
        "resnet18_int8": ModelSpec(lambda p: _torchvision_int8(tvq.resnet18, qr18, p),
                                   backend=qr18.DEFAULT.meta["backend"]),
        # detector (latency only unless a COCO evaluation is wired in), FP32 and post-training INT8
        "ssdlite320_mobilenet_v3_large": ModelSpec(ssdlite, img=320, detector=True),
        "ssdlite320_mobilenet_v3_large_int8": ModelSpec(ssdlite, img=320, detector=True, ptq=True, backend="x86"),
    }
    # models trained on Imagenette by scripts/finetune_imagenette.py, FP32 and post-training INT8
    for arch in ARCHS:
        for init, suffix in (("imagenet", ""), ("scratch", "_scratch")):
            zoo[f"{arch}_imagenette{suffix}"] = _imagenette_model(models_dir, init, arch)
            zoo[f"{arch}_imagenette{suffix}_int8"] = _imagenette_model(models_dir, init, arch, int8=True)
            zoo[f"{arch}_imagenette{suffix}_int8qat"] = _imagenette_model(models_dir, init, arch, int8=True, qat=True)
    return zoo


# architectures of the Imagenette tiers, smallest first (see deployment_tiers)
LADDER = (("tiny", "mobilenet_v3_small"), ("small", "mobilenet_v3_large"), ("medium", "resnet18"))


def int8_state_path(models_dir: str, name: str) -> Path:
    """Where the benchmark saves the calibrated state of a post-training INT8 model (`ModelSpec.ptq`)."""
    return Path(models_dir) / f"{name}.pt"


def load_model(name: str, models_dir: str = "models") -> torch.nn.Module:
    """A model of the zoo with its trained / pretrained weights, as deployed: post-training INT8 models are
    restored from the state the benchmark saved (`int8_state_path`)."""
    spec = model_zoo(models_dir)[name]
    model = build(spec, pretrained=True)
    if spec.ptq:
        from .quantize import load_quantized
        path = int8_state_path(models_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run scripts/run_benchmark.py on {name} first (it calibrates it)")
        model = load_quantized(model, path, spec.backend, spec.img)
    return model


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
    # models also convert observers that saw no data. Neither affects the latency measurement. A torchvision
    # model left without a qconfig is quantised by `_torchvision_int8`.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*(torch\.ao\.quantization is deprecated|must run observer"
                                                  r"|quantized tensor creation functions|None of the submodule)")
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


def imagenette_loader(root: str = "data", img: int = 224, batch_size: int = 32, download: bool = True, split: str = "val",
                      shuffle: bool = False, augment: bool = False):
    """Imagenette loader at the given resolution (evaluation transform, or the training augmentation); None if
    the dataset is unavailable."""
    from .data import train_transform
    try:
        return make_loader(imagenette(root, split, img=img, download=download,
                                      transform=train_transform(img) if augment else eval_transform(img)),
                           batch_size=batch_size, shuffle=shuffle)
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


def run_worker(job: dict, timeout: float = 1800.0) -> dict:
    """Measure one model in a fresh process (see `edgearbiter.bench_worker`)."""
    proc = subprocess.run([sys.executable, "-m", "edgearbiter.bench_worker", json.dumps(job)], capture_output=True,
                          text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT "):])
    raise RuntimeError(f"benchmark worker failed for {job['model']}:\n{proc.stderr[-2000:]}")


def benchmark_models(names: List[str], role: str = "device", iters: int = 50, threads: Sequence[Optional[int]] = (None,),
                     pretrained: bool = True, data_root: Optional[str] = None, max_batches: Optional[int] = None,
                     models_dir: str = "models", cpu_quota: Optional[float] = None, calib_batches: int = 16) -> List[Dict]:
    """One row per (model, thread count). Accuracy needs trained/pretrained weights and `data_root`;
    INT8 models quantised here (`*_int8`, `*_int8qat`) need `data_root` for their calibration / training
    images, except with random weights (latency and memory only), which calibrate on random images."""
    from contextlib import nullcontext
    from .cpulimit import CpuLimit
    zoo = model_zoo(models_dir); loaders: Dict[tuple, object] = {}
    rows = []
    with tempfile.TemporaryDirectory() as tmp, (CpuLimit(cpu_quota) if cpu_quota else nullcontext()) as limit:
        for name in names:
            spec = zoo[name]
            label = "random-init" if not pretrained else ("trained" if not spec.imagenet_head else "pretrained")
            print(f"[benchmark] {name} ({label}, {spec.img}px) ...", flush=True)
            model = build(spec, pretrained)
            int8_state = None
            if spec.ptq:
                from .quantize import quantize_ptq, quantize_qat
                key = ("train", spec.img, spec.qat)
                if data_root:
                    loaders.setdefault(key, imagenette_loader(data_root, spec.img, split="train", shuffle=True,
                                                              batch_size=64 if spec.qat else 32, augment=spec.qat))
                images, fast = loaders.get(key), False
                if images is None:
                    if pretrained:
                        raise ValueError(f"{name}: INT8 quantisation of trained weights needs calibration / training "
                                         f"images (Imagenette, data_root)")
                    # random weights measure latency and memory only: random images and min/max observers calibrate
                    # them as well (same INT8 graph), in seconds
                    images, fast = [(torch.rand(8, 3, spec.img, spec.img), torch.randint(0, 10, (8,))) for _ in range(2)], True
                model = (quantize_qat(model, images, backend=spec.backend, img=spec.img) if spec.qat else
                         quantize_ptq(model, images, backend=spec.backend, batches=calib_batches, img=spec.img, fast=fast))
                # trained weights: keep the calibrated model for deployment (API /infer); random ones: temporary
                int8_state = str(int8_state_path(models_dir, name) if pretrained else Path(tmp) / f"{name}.pt")
                torch.save(model.state_dict(), int8_state)
            acc = None
            if pretrained and data_root and not spec.detector:
                key = ("val", spec.img)
                loaders.setdefault(key, imagenette_loader(data_root, spec.img))
                if loaders[key] is not None:
                    acc = imagenette_accuracy(model, loaders[key], max_batches, spec.imagenet_head)
            size = round(state_dict_size_mb(model), 2)
            for n_threads in threads:
                job = {"model": name, "pretrained": pretrained, "threads": n_threads or torch.get_num_threads(),
                       "iters": iters, "models_dir": models_dir, "int8_state": int8_state, "engine": spec.backend,
                       "cgroup_procs": limit.procs if limit else None}
                m = run_worker(job)
                rows.append({"model": name, "role": role, "int8": spec.backend is not None, "quant_engine": m["quant_engine"],
                             "quantisation": (("QAT (FX)" if spec.qat else "PTQ (FX)") if spec.ptq else "torchvision")
                             if spec.backend else None,
                             "weights": "random" if not pretrained else label, "img": spec.img, "threads": m["threads"],
                             "cpu_quota": cpu_quota, "latency_ms": m["latency_ms"],
                             "latency_net_ms": m.get("latency_net_ms"), "size_mb": size,
                             "memory_mb": m["model_mem_mb"], "inference_mem_mb": m["inference_mem_mb"],
                             "rss_peak_mb": m["rss_peak_mb"], "accuracy": acc,
                             "accuracy_metric": ("imagenette_10way_top1" if acc is not None else None)})
                lat = m["latency_ms"]
                print(f"    threads {m['threads']}{f' @ {cpu_quota:g} CPU' if cpu_quota else ''}: p50 {lat['p50']:.1f} ms  "
                      f"p95 {lat['p95']:.1f} ms  memory {m['model_mem_mb']} MB  size {size} MB  acc {acc}", flush=True)
    return rows


def markdown_table(rows: List[Dict], host: Optional[Dict] = None) -> str:
    lines = []
    if host:
        weights = sorted({r["weights"] for r in rows})
        lines += [f"Host: {host.get('label') or host['cpu']} ({host['cpu']}, {host['logical_cpus']} logical CPUs) · "
                  f"torch {host['torch']} · weights: {', '.join(weights)} · {host['date_utc']}", ""]
        quotas = sorted({r.get("cpu_quota") for r in rows if r.get("cpu_quota")})
        if quotas:
            lines += [f"Emulated device: inference thread limited to {', '.join(f'{q:g}' for q in quotas)} CPU "
                      f"(cgroup CFS quota).", ""]
        if "random" in weights:
            lines += ["Random-init weights: latency, size and memory are valid, accuracy is not measured.", ""]
    lines += ["| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = "—" if r["accuracy"] is None else f"{100 * r['accuracy']:.1f}%"
        method = {"PTQ (FX)": ", PTQ", "QAT (FX)": ", QAT"}.get(r.get("quantisation"), "")
        q = f"INT8 ({r['quant_engine']}{method})" if r["int8"] else "FP32"
        mem = "—" if r.get("memory_mb") is None else f"{r['memory_mb']:.1f}"
        lines.append(f"| {r['model']} | {q} | {r['img']}px | {r['threads']} | {r['latency_ms']['p50']:.1f} | "
                     f"{r['latency_ms']['p95']:.1f} | {mem} | {r['size_mb']} | {acc} |")
    nets = [r for r in rows if r.get("latency_net_ms")]
    if nets:
        lines += ["", "Detectors, backbone + heads only (the part that runs in INT8; no box decoding or NMS):", "",
                  "| model | precision | threads | p50 ms | p95 ms |", "|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {'INT8' if r['int8'] else 'FP32'} | {r['threads']} | {r['latency_net_ms']['p50']:.1f} | "
                  f"{r['latency_net_ms']['p95']:.1f} |" for r in nets]
    return "\n".join(lines) + "\n"


def _pick(rows: List[Dict], model: str, fastest: bool = False) -> Optional[Dict]:
    """The row of `model` with the most threads (a device uses all its cores), or with the lowest p50
    (`fastest`: a server runs each model at its best thread count)."""
    rows = [r for r in rows if r["model"] == model]
    if not rows:
        return None
    return min(rows, key=lambda r: r["latency_ms"]["p50"]) if fastest else max(rows, key=lambda r: r.get("threads") or 0)


def deployment_tiers(device_rows: List[Dict], init: str = "scratch", max_int8_loss: float = 0.01) -> Tuple[Dict[str, str], List[str]]:
    """Tiers from the models trained on Imagenette, one per architecture of LADDER: the post-training INT8
    model when it is faster on the device and loses at most `max_int8_loss` top-1, else the FP32 model.
    Returns the tier -> model mapping and the reason for each choice."""
    suffix = "" if init == "imagenet" else f"_{init}"
    tiers, reasons = {}, []
    for tier, arch in LADDER:
        fp, q = _pick(device_rows, f"{arch}_imagenette{suffix}"), _pick(device_rows, f"{arch}_imagenette{suffix}_int8")
        if fp is None:
            raise KeyError(f"{arch}_imagenette{suffix} missing from the device benchmark")
        use = fp
        if q is not None and fp["accuracy"] is not None and q["accuracy"] is not None:
            if q["latency_ms"]["p50"] < fp["latency_ms"]["p50"] and fp["accuracy"] - q["accuracy"] <= max_int8_loss:
                use = q
            reasons.append(f"{tier}: {arch} {'INT8' if use is q else 'FP32'} (device p50 INT8 {q['latency_ms']['p50']:.1f} "
                           f"vs FP32 {fp['latency_ms']['p50']:.1f} ms, top-1 {100 * q['accuracy']:.1f} vs "
                           f"{100 * fp['accuracy']:.1f} %)")
        tiers[tier] = use["model"]
    return tiers, reasons


def _p95(row: Dict) -> Optional[float]:
    p95 = row["latency_ms"].get("p95")
    return round(p95, 2) if p95 is not None else None


def build_cost_table(device_rows: List[Dict], edge_rows: Optional[List[Dict]] = None, tiers: Dict[str, str] = TIERS,
                     meta: Optional[Dict] = None) -> Dict:
    """Merge device (and optionally edge) measurements into the simulator's cost table."""
    out = []
    for tier, model in tiers.items():
        d = _pick(device_rows, model)
        if d is None:
            raise KeyError(f"tier model {model!r} missing from the device benchmark")
        e = _pick(edge_rows or [], model, fastest=True)
        dev_lat = d["latency_ms"]["p50"]
        edge_lat = e["latency_ms"]["p50"] if e else dev_lat / 8.0      # no edge run yet: assume an 8x faster server
        out.append({"name": tier, "model": model, "accuracy": d["accuracy"],
                    "latency_ms": {"device": round(dev_lat, 2), "edge": round(edge_lat, 2)},
                    "latency_p95_ms": {"device": _p95(d), "edge": _p95(e) if e else None},
                    "energy_mj": round(dev_lat * ENERGY_MJ_PER_MS, 2), "size_mb": d["size_mb"],
                    "memory_mb": d.get("memory_mb"), "edge_measured": e is not None,
                    "edge_threads": e.get("threads") if e else None})
    table = {"tiers": out, "frame_kb": 40.0, "tx_energy_mj_per_kb": 0.25, "source": "benchmark"}
    if meta:
        table["meta"] = meta
    return table
