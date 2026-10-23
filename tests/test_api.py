import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from edgearbiter import api


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_and_cost_table(client):
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/cost_table").json()["tiers"]) == 3


def test_decide_and_sim(client):
    d = client.post("/policy/decide", json={"battery": 0.9, "queue": 12, "rtt_ms": 12, "bandwidth_mbps": 30, "arrivals": 6, "edge_load": 0.2}).json()
    assert d["placement"] in ("edge", "local") and d["rate_fps"] in (1, 5, 10)
    assert d["edge_accept"] is True and d["policy"] == {"device": "rule", "edge": "rule"}
    assert client.post("/sim/reset", params={"seed": 1}).status_code == 200
    s = client.post("/sim/step").json()
    assert "kpis" in s and "last" in s


def test_busy_edge_stops_admitting_and_device_adapts(client):
    busy = {"battery": 0.9, "queue": 12, "rtt_ms": 12, "bandwidth_mbps": 30, "arrivals": 6, "edge_load": 0.9}
    first = client.post("/policy/decide", json=busy).json()
    assert first["edge_accept"] is False                     # edge rule rejects above 70 % load
    second = client.post("/policy/decide", json=busy).json()
    assert second["placement"] == "local" and "rejected" not in second   # device saw the published rejection
