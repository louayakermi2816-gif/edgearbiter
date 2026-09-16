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
