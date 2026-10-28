"""The software IoT loop: virtual ESP32 node <-> MQTT broker <-> policy service, no hardware."""
import json
import socket
import threading
import time

import pytest

from edgearbiter.costs import CostTable
from edgearbiter.sim.env import SimConfig
from edgearbiter.virtual_node import VirtualNode


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_actions_from_api_and_firmware_style_decisions():
    node = VirtualNode(broker=None, timeout=0.0, costs=CostTable.synthetic(), log=None)
    env = node.env
    api_style = {"action": env.encode_device(2, 2, 1), "edge_accept": True, "edge_share": 0.5}
    assert node.actions(api_style) == {"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(1, 0)}
    fw_style = {"tier": 1, "rate_fps": 5, "placement": "local"}
    assert node.actions(fw_style)["device"] == env.encode_device(1, 1, 0)
    assert node.actions(None)["device"] == env.encode_device(0, 1, 0)        # nothing received yet: tiny local


def test_offline_node_keeps_running_on_timeouts():
    node = VirtualNode(broker=None, timeout=0.0, interval=0.0, costs=CostTable.synthetic(), cfg=SimConfig(horizon=20), log=None)
    summary = node.run(steps=45)
    assert summary["timeouts"] == 45 and node.episode == 2                 # episodes roll over


def test_mqtt_loop_with_a_policy_service():
    pytest.importorskip("amqtt")
    pytest.importorskip("fastapi")
    import paho.mqtt.client as mqtt
    from fastapi.testclient import TestClient

    from edgearbiter import api
    from edgearbiter.broker import start_in_thread

    port = free_port()
    start_in_thread("127.0.0.1", port)
    service = TestClient(api.app)
    edge_load = {"value": 0.3}

    def on_message(client, userdata, msg):          # what the Node-RED flow does
        payload = json.loads(msg.payload)
        if msg.topic.endswith("/edge"):
            edge_load["value"] = payload["edge_load"]
            return
        decision = service.post("/policy/decide", json=dict(payload, edge_load=edge_load["value"])).json()
        client.publish("ea/decision", json.dumps(dict(decision, seq=payload["seq"])))

    bridge = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    bridge.on_message = on_message
    bridge.connect("127.0.0.1", port)
    bridge.subscribe([("ea/telemetry", 0), ("ea/edge", 0)])
    bridge.loop_start()
    seen = []
    watcher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    watcher.on_message = lambda c, u, m: seen.append(json.loads(m.payload))
    watcher.connect("127.0.0.1", port); watcher.subscribe("ea/node"); watcher.loop_start()
    time.sleep(0.5)
    node = VirtualNode("127.0.0.1", port, topic="ea", interval=0.05, timeout=2.0, costs=CostTable.synthetic(), log=None)
    try:
        summary = node.run(steps=30)
    finally:
        node.close(); bridge.loop_stop(); watcher.loop_stop()
    time.sleep(0.2)
    assert summary["decisions"] == 30 and summary["timeouts"] == 0, summary
    assert summary["kpis"]["processed"] > 0 and summary["mean_loop_rtt_ms"] > 0, summary
    assert len(seen) >= 25 and {"tier", "where", "battery", "kpis", "loop_rtt_ms"} <= set(seen[-1]), (len(seen), summary)
