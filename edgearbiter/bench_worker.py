"""Measure one model in a fresh process: latency (p50 / p95 / mean) and peak resident memory.

Run by `edgearbiter.benchmark` (one process per model and hardware profile) so that memory is
not polluted by other models and the CPU limit applies to this measurement only:

    python -m edgearbiter.bench_worker '{"model": "resnet18", "pretrained": false, "threads": 1, ...}'

Memory: `model_mem_mb` = `weights_mb` + `inference_mem_mb`, what the deployed model needs:
its weights (serialised state) plus the resident memory that running it adds (activations,
workspace, kernel caches): the peak during the timed inferences minus the resident memory of
the process with the model loaded. Building the model is excluded on purpose: the INT8 models
are rebuilt through FX quantisation, whose toolchain costs ~180 MB on first use and keeps
~15 MB of each graph it builds, none of which a deployed model carries. `rss_baseline_mb`
(torch imported) and `rss_loaded_mb` (model loaded, garbage collected and returned to the OS)
are reported for reference. The peak is the kernel's high-water mark on Linux (reset through
/proc/self/clear_refs); elsewhere a thread samples the resident memory every millisecond.
"""
from __future__ import annotations

import ctypes
import gc
import json
import sys
import threading
import time
from pathlib import Path

STATUS = Path("/proc/self/status")


def _status_mb(field: str) -> float:
    for line in STATUS.read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024 / 1e6                    # kB (KiB) -> MB, the unit of the model sizes
    raise KeyError(field)


class PeakMemory:
    """Resident memory now (`baseline`) and the peak reached from now until `peak()`."""

    def __init__(self):
        self.proc, self.hwm = STATUS.exists(), False
        self.restart()
        self.baseline = self.current()
        self._stop = threading.Event()
        if not self.hwm:
            threading.Thread(target=self._sample, daemon=True).start()

    def restart(self) -> None:
        """Forget the peak so far: from now on the peak starts at the current resident memory."""
        if self.proc:
            try:
                Path("/proc/self/clear_refs").write_text("5")            # reset VmHWM to the current RSS
                self.hwm = True
            except OSError:
                pass
        self._max = self.current()

    def current(self) -> float:
        if self.proc:
            return _status_mb("VmRSS")
        import psutil
        return psutil.Process().memory_info().rss / 1e6

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._max = max(self._max, self.current()); time.sleep(0.001)

    def peak(self) -> float:
        if self.hwm:
            return _status_mb("VmHWM")
        self._stop.set()
        return max(self._max, self.current())


def release_garbage() -> None:
    """Collect garbage and hand freed heap memory back to the OS, so it does not count as resident."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):                                   # not glibc
        pass


def load_quantization_toolchain() -> None:
    """Quantise a two-layer model (FX and eager mode): imports what INT8 models need on first use."""
    import torch
    from .quantize import quantize_ptq
    tiny = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        quantize_ptq(tiny, [(x, None)], batches=1, img=16)(x)
        eager = torch.ao.quantization.QuantWrapper(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU())).eval()
        eager.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        prepared = torch.ao.quantization.prepare(eager); prepared(x)
        torch.ao.quantization.convert(prepared)(x)


def run(job: dict) -> dict:
    import torch
    from .benchmark import build, example_input, measure_latency, model_zoo
    torch.set_num_threads(int(job.get("threads") or torch.get_num_threads()))
    with torch.no_grad():
        torch.ones(8, 8) @ torch.ones(8, 8)                      # initialise the runtime before the baseline
    spec = model_zoo(job.get("models_dir", "models"))[job["model"]]
    load_quantization_toolchain()                                # part of the baseline, not of the model
    release_garbage()
    mem = PeakMemory()
    baseline = mem.baseline
    model = build(spec, job.get("pretrained", True))
    if job.get("int8_state"):                                    # FX post-training quantised weights + scales
        from .quantize import load_quantized
        model = load_quantized(model, job["int8_state"], job.get("engine"), spec.img)
    release_garbage(); mem.restart()                             # memory of the loaded model, not of building it
    loaded = mem.current()
    if job.get("cgroup_procs"):                                  # emulated device: CPU limit for the timed part
        from .cpulimit import join
        join(job["cgroup_procs"])
    lat = measure_latency(model, example_input(spec), iters=int(job.get("iters", 50)), warmup=int(job.get("warmup", 10)))
    peak = mem.peak()
    from .benchmark import state_dict_size_mb
    weights = state_dict_size_mb(model)
    return {"latency_ms": lat, "rss_baseline_mb": round(baseline, 1), "rss_loaded_mb": round(loaded, 1),
            "rss_peak_mb": round(peak, 1), "weights_mb": round(weights, 2), "inference_mem_mb": round(peak - loaded, 1),
            "model_mem_mb": round(weights + peak - loaded, 1), "threads": torch.get_num_threads(),
            "quant_engine": torch.backends.quantized.engine if (spec.backend or job.get("int8_state")) else None}


if __name__ == "__main__":
    t0 = time.time()
    out = run(json.loads(sys.argv[1]))
    out["worker_seconds"] = round(time.time() - t0, 1)
    print("BENCH_RESULT " + json.dumps(out))
