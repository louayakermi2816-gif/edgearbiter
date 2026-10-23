import json

from edgearbiter.benchmark import build_cost_table
from edgearbiter.costs import CostTable


def test_synthetic_default_when_missing(tmp_path):
    t = CostTable.load(tmp_path / "missing.json")
    assert t.source == "synthetic" and t.n_tiers == 3
    assert t.latency(0, "device") < t.latency(2, "device")
    assert t.accuracy(0) < t.accuracy(2)


def test_save_and_reload(tmp_path):
    t = CostTable.synthetic(); p = tmp_path / "cost_table.json"
    t.save(p)
    t2 = CostTable.load(p)
    assert t2.tiers == t.tiers and t2.source == str(p)


def test_build_cost_table_from_benchmark_rows():
    rows = [{"model": m, "latency_ms": {"p50": lat, "p95": lat * 1.2}, "size_mb": 10.0, "accuracy": acc}
            for m, lat, acc in (("mobilenet_v3_small", 20.0, 0.7), ("mobilenet_v3_large_int8", 40.0, 0.75), ("resnet18", 100.0, 0.8))]
    table = build_cost_table(rows)
    assert [t["name"] for t in table["tiers"]] == ["tiny", "small", "medium"]
    assert table["tiers"][2]["latency_ms"]["device"] == 100.0 and table["tiers"][2]["edge_measured"] is False
    t = CostTable(tiers=table["tiers"])
    assert t.energy(2) > t.energy(0)


def test_synthetic_keyword_and_fingerprint(tmp_path):
    t = CostTable.synthetic(); p = tmp_path / "copy.json"
    t.save(p)
    assert CostTable.load("synthetic").source == "synthetic"
    assert CostTable.load(p).fingerprint == t.fingerprint                      # same numbers, other path
    changed = CostTable(tiers=[dict(x) for x in t.tiers]); changed.tiers[0]["accuracy"] = 0.5
    assert changed.fingerprint != t.fingerprint


def _row(model, p50, acc, threads=1):
    return {"model": model, "threads": threads, "latency_ms": {"p50": p50, "p95": 1.1 * p50}, "size_mb": 1.0,
            "accuracy": acc, "memory_mb": 10.0}


def test_deployment_tiers_take_int8_only_when_faster_and_as_accurate():
    from edgearbiter.benchmark import deployment_tiers
    rows = [_row("mobilenet_v3_small_imagenette_scratch", 36.0, 0.826),
            _row("mobilenet_v3_small_imagenette_scratch_int8", 37.0, 0.825),       # slower: FP32
            _row("mobilenet_v3_large_imagenette_scratch", 58.0, 0.850),
            _row("mobilenet_v3_large_imagenette_scratch_int8", 48.0, 0.835),       # loses 1.5 pt: FP32
            _row("resnet18_imagenette_scratch", 112.0, 0.870),
            _row("resnet18_imagenette_scratch_int8", 33.0, 0.866)]                 # faster, -0.4 pt: INT8
    tiers, reasons = deployment_tiers(rows)
    assert tiers == {"tiny": "mobilenet_v3_small_imagenette_scratch", "small": "mobilenet_v3_large_imagenette_scratch",
                     "medium": "resnet18_imagenette_scratch_int8"} and len(reasons) == 3


def test_edge_latency_is_the_fastest_thread_count():
    device = [_row("m", 40.0, 0.8)]
    edge = [_row("m", 9.0, 0.8, threads=1), _row("m", 12.0, 0.8, threads=4)]   # small models do not scale with threads
    tier = build_cost_table(device, edge, tiers={"tiny": "m"}, meta={"note": "x"})
    assert tier["tiers"][0]["latency_ms"] == {"device": 40.0, "edge": 9.0} and tier["tiers"][0]["edge_threads"] == 1
    assert tier["tiers"][0]["latency_p95_ms"]["edge"] == 9.9 and tier["meta"] == {"note": "x"}
