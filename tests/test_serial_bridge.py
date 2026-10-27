"""Serial <-> MQTT bridge of the ESP32 node's serial build (the firmware itself runs in QEMU: scripts/run_qemu_node.py)."""
import json
import socket
import threading
import time

import pytest

from edgearbiter.serial_bridge import SerialBridge, TcpLink

TELEMETRY = {"node": "esp32-1", "seq": 3, "battery": 0.98, "queue": 2, "rtt_ms": 20, "arrivals": 4, "uptime_s": 3,
             "tier": 0, "placement": "local"}


class FakeLink:
    def __init__(self, lines):
        self.lines, self.written = [line.encode() for line in lines], []

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def write(self, data):
        self.written.append(data.decode())


def test_node_lines_are_published_and_decisions_go_back_as_one_line():
    link = FakeLink(["ets Jul 29 2019 12:21:46\n", f"telemetry {json.dumps(TELEMETRY)}\n",
                     'node {"node": "esp32-1", "seq": 3, "kpis": {"processed": 7}}\r\n', "telemetry {broken\n", "\n"])
    published, logged = [], []
    bridge = SerialBridge(link, lambda topic, payload: published.append((topic, json.loads(payload))), topic="ea",
                          log=logged.append)
    counts = bridge.run()
    assert [t for t, _ in published] == ["ea/telemetry", "ea/node"] and published[0][1] == TELEMETRY
    assert counts == {"telemetry": 1, "node": 1, "decision": 0}
    assert any("ets Jul 29" in m for m in logged) and any("malformed" in m for m in logged)
    bridge.send_decision(b'{"seq": 3, "tier": 2, "rate_fps": 10, "placement": "edge", "policy": {"device": "actor"}}')
    assert link.written == ['decision {"seq":3,"tier":2,"rate_fps":10,"placement":"edge","policy":{"device":"actor"}}\n']


def test_bridge_relays_between_a_tcp_serial_port_and_the_broker():
    pytest.importorskip("amqtt")
    import paho.mqtt.client as mqtt

    from edgearbiter.broker import start_in_thread

    def free_port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    broker_port, serial_port = free_port(), free_port()
    start_in_thread("127.0.0.1", broker_port)
    server = socket.create_server(("127.0.0.1", serial_port))      # stands in for QEMU's serial socket
    received = []

    def node():
        conn, _ = server.accept()
        with conn, conn.makefile("rwb") as f:
            for seq in range(1, 4):
                f.write(f"telemetry {json.dumps(dict(TELEMETRY, seq=seq))}\n".encode()); f.flush()
                received.append(f.readline().decode())                   # wait for the decision of this interval

    threading.Thread(target=node, daemon=True).start()
    policy = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)           # what Node-RED + the policy service do
    policy.on_message = lambda c, u, m: c.publish("ea/decision", json.dumps(
        {"seq": json.loads(m.payload)["seq"], "tier": 2, "rate_fps": 10, "placement": "edge"}))
    policy.connect("127.0.0.1", broker_port); policy.subscribe("ea/telemetry"); policy.loop_start()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    link = TcpLink("127.0.0.1", serial_port)
    bridge = SerialBridge(link, lambda topic, payload: client.publish(topic, payload), topic="ea", log=None)
    client.on_message = lambda c, u, m: bridge.send_decision(m.payload)
    client.connect("127.0.0.1", broker_port); client.subscribe("ea/decision"); client.loop_start()
    time.sleep(0.5)
    try:
        counts = bridge.run(max_lines=3)
        for _ in range(50):
            if len(received) == 3:
                break
            time.sleep(0.1)
    finally:
        client.loop_stop(); policy.loop_stop(); link.close(); server.close()
    assert counts["telemetry"] == 3
    assert [json.loads(line.split(" ", 1)[1])["seq"] for line in received] == [1, 2, 3]
    assert all(line.startswith("decision ") for line in received)
