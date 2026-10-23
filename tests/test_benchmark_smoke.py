"""Benchmark helpers on tiny / random-init models and synthetic data (no downloads)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from edgearbiter.benchmark import (IMAGENETTE_TO_IMAGENET, benchmark_models, build, build_cost_table, example_input,
                                   imagenette_accuracy, markdown_table, measure_latency, model_zoo, state_dict_size_mb)


def test_measure_latency_and_size():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1000))
    lat = measure_latency(model, torch.rand(1, 3, 32, 32), iters=5, warmup=2)
    assert lat["p50"] > 0 and lat["p95"] >= lat["p50"]
    assert state_dict_size_mb(model) > 0


def test_imagenette_accuracy_mapping():
    class Oracle(nn.Module):  # always predicts the ImageNet index of class 3 ("chain saw")
        def forward(self, x):
            out = torch.zeros(x.shape[0], 1000); out[:, IMAGENETTE_TO_IMAGENET[3]] = 1.0; return out
    loader = [(torch.rand(4, 3, 8, 8), torch.tensor([3, 3, 0, 3]))]
    assert imagenette_accuracy(Oracle(), loader) == pytest.approx(0.75)


def test_imagenette_accuracy_native_head():
    class Native(nn.Module):  # 10-class head, always class 1
        def forward(self, x):
            out = torch.zeros(x.shape[0], 10); out[:, 1] = 1.0; return out
    loader = [(torch.rand(4, 3, 8, 8), torch.tensor([1, 1, 2, 1]))]
    assert imagenette_accuracy(Native(), loader, imagenet_head=False) == pytest.approx(0.75)


def test_random_init_int8_runs_on_its_calibration_engine():
    spec = model_zoo()["mobilenet_v3_large_int8"]
    if spec.backend not in torch.backends.quantized.supported_engines:
        pytest.skip(f"{spec.backend} not available")
    model = build(spec, pretrained=False)
    assert torch.backends.quantized.engine == spec.backend == "qnnpack"
    with torch.no_grad():
        assert model(example_input(spec)).shape == (1, 1000)


def test_benchmark_rows_and_table():
    rows = benchmark_models(["mobilenet_v3_small"], iters=2, threads=(1, 2), pretrained=False)
    assert [r["threads"] for r in rows] == [1, 2] and all(r["accuracy"] is None and r["weights"] == "random" for r in rows)
    assert all(r["memory_mb"] > 5 for r in rows)                       # ~10 MB of weights + activations
    md = markdown_table(rows, {"label": None, "cpu": "test-cpu", "logical_cpus": 2, "torch": torch.__version__, "date_utc": "now"})
    assert "Random-init weights" in md and md.count("| mobilenet_v3_small |") == 2


def test_post_training_int8_keeps_predictions_and_restores_from_state():
    import io
    from edgearbiter.finetune import build_model
    from edgearbiter.quantize import quantize_ptq
    model = build_model(pretrained=False, arch="resnet18").eval()
    calib = [(torch.rand(4, 3, 64, 64), None) for _ in range(3)]
    q = quantize_ptq(model, calib, batches=3, img=64)
    x = torch.rand(2, 3, 64, 64)
    assert q(x).shape == (2, 10) and state_dict_size_mb(q) < state_dict_size_mb(model) / 3
    buf = io.BytesIO(); torch.save(q.state_dict(), buf); buf.seek(0)
    fresh = quantize_ptq(build_model(pretrained=False, arch="resnet18").eval(), [(torch.rand(1, 3, 64, 64), None)], batches=1, img=64)
    fresh.load_state_dict(torch.load(buf, weights_only=False))
    assert torch.equal(fresh(x), q(x))


def test_cpu_limit_slows_the_measurement_down():
    from edgearbiter.cpulimit import CpuLimit
    try:
        limit = CpuLimit(0.25).__enter__()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    limit.__exit__(None, None, None)
    free = benchmark_models(["mobilenet_v3_small"], iters=5, threads=(1,), pretrained=False)[0]["latency_ms"]["p50"]
    slow = benchmark_models(["mobilenet_v3_small"], iters=5, threads=(1,), pretrained=False, cpu_quota=0.25)[0]
    assert slow["cpu_quota"] == 0.25 and slow["latency_ms"]["p50"] > 2 * free


def test_cost_table_uses_the_row_with_most_threads():
    rows = [{"model": m, "threads": th, "latency_ms": {"p50": lat / th}, "size_mb": 1.0, "accuracy": None}
            for m, lat in (("mobilenet_v3_small", 20.0), ("mobilenet_v3_large_int8", 40.0), ("resnet18", 100.0)) for th in (1, 4)]
    table = build_cost_table(rows)
    assert table["tiers"][2]["latency_ms"]["device"] == 25.0
