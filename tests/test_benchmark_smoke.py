"""Benchmark helpers on a tiny random model and synthetic data (no downloads)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from edgearbiter.benchmark import IMAGENETTE_TO_IMAGENET, imagenette_accuracy, measure_latency, state_dict_size_mb


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
