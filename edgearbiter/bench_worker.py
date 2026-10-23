"""Measure one model in a fresh process: latency (p50 / p95 / mean) and peak resident memory.

Run by `edgearbiter.benchmark` (one process per model and hardware profile) so that memory is
not polluted by other models and the CPU limit applies to this measurement only:

    python -m edgearbiter.bench_worker '{"model": "resnet18", "pretrained": false, "threads": 1, ...}'

Memory: `rss_baseline_mb` is the resident memory of the process after importing torch,
`rss_peak_mb` its peak while building the model and running the timed inferences;
`model_mem_mb` = peak - baseline is the memory the model needs (weights, activations,
workspace). On Linux the kernel's high-water mark is reset after the baseline
(/proc/self/clear_refs), so import-time peaks do not hide the model; elsewhere a thread
samples the resident memory every millisecond.
"""
from __future__ import annotations

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
        if self.proc:
            try:
                Path("/proc/self/clear_refs").write_text("5")            # reset VmHWM to the current RSS
                self.hwm = True
            except OSError:
                pass
        self.baseline = self.current()
        self._max, self._stop = self.baseline, threading.Event()
        if not self.hwm:
            threading.Thread(target=self._sample, daemon=True).start()

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


def run(job: dict) -> dict:
    import torch
    from .benchmark import build, example_input, measure_latency, model_zoo
    torch.set_num_threads(int(job.get("threads") or torch.get_num_threads()))
    with torch.no_grad():
        torch.ones(8, 8) @ torch.ones(8, 8)                      # initialise the runtime before the baseline
    spec = model_zoo(job.get("models_dir", "models"))[job["model"]]
    mem = PeakMemory()
    model = build(spec, job.get("pretrained", True))
    if job.get("int8_state"):                                    # FX post-training quantised weights + scales
        from .quantize import quantize_ptq
        model = quantize_ptq(model, [(torch.rand(1, 3, spec.img, spec.img), None)], backend=job.get("engine"), batches=1,
                             img=spec.img)
        model.load_state_dict(torch.load(job["int8_state"], weights_only=False))
    if job.get("cgroup_procs"):                                  # emulated device: CPU limit for the timed part
        from .cpulimit import join
        join(job["cgroup_procs"])
    lat = measure_latency(model, example_input(spec), iters=int(job.get("iters", 50)), warmup=int(job.get("warmup", 10)))
    peak = mem.peak()
    return {"latency_ms": lat, "rss_baseline_mb": round(mem.baseline, 1), "rss_peak_mb": round(peak, 1),
            "model_mem_mb": round(peak - mem.baseline, 1), "threads": torch.get_num_threads(),
            "quant_engine": torch.backends.quantized.engine if (spec.backend or job.get("int8_state")) else None}


if __name__ == "__main__":
    t0 = time.time()
    out = run(json.loads(sys.argv[1]))
    out["worker_seconds"] = round(time.time() - t0, 1)
    print("BENCH_RESULT " + json.dumps(out))
