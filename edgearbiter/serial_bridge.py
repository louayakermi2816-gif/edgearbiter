"""Serial <-> MQTT bridge for the ESP32 node built with the serial transport (firmware/esp32/build.sh serial).

That build writes "<subtopic> <json>" lines on its serial port (telemetry, node) and reads "decision <json>"
lines. The bridge publishes the first on <topic>/<subtopic> and forwards every <topic>/decision message to
the node as a line, so Node-RED and the policy service see the same messages as from the Wi-Fi build or the
virtual node. The serial port is a TCP socket (Espressif's QEMU, `-serial tcp::5555,server`) or a device
(a board on USB; needs pyserial):

    python -m edgearbiter.serial_bridge --tcp localhost:5555 --broker localhost
    python -m edgearbiter.serial_bridge --device /dev/ttyUSB0 --broker localhost

Other lines (boot messages, logs) are printed with a "[node]" prefix.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Callable, Optional

UPLINK = ("telemetry", "node")          # subtopics the node publishes through the bridge


class TcpLink:
    """The node's serial port exposed on a TCP socket (QEMU's chardev socket); retries while it starts."""

    def __init__(self, host: str, port: int, attempts: int = 60):
        for attempt in range(attempts):
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.5)
        self.sock.settimeout(None)
        self.reader = self.sock.makefile("rb")

    def readline(self) -> bytes:
        return self.reader.readline()

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)

    def close(self) -> None:
        self.sock.close()


class DeviceLink:
    """A serial device (pyserial), e.g. a board on USB."""

    def __init__(self, device: str, baud: int = 115200):
        import serial
        self.port = serial.Serial(device, baud, timeout=None)

    def readline(self) -> bytes:
        return self.port.readline()

    def write(self, data: bytes) -> None:
        self.port.write(data)

    def close(self) -> None:
        self.port.close()


class SerialBridge:
    def __init__(self, link, publish: Callable[[str, str], None], topic: str = "edgearbiter",
                 log: Optional[Callable] = print):
        self.link, self.publish, self.topic, self.log = link, publish, topic, log or (lambda *a: None)
        self.counts = {sub: 0 for sub in UPLINK} | {"decision": 0}

    def handle_line(self, line: str) -> Optional[str]:
        """Publish an uplink line from the node; the subtopic it went to, or None for any other line."""
        sub, _, payload = line.strip().partition(" ")
        if sub in UPLINK and payload.startswith("{"):
            try:
                json.loads(payload)
            except ValueError:
                self.log(f"[node] malformed {sub} line: {payload[:80]}")
                return None
            self.publish(f"{self.topic}/{sub}", payload)
            self.counts[sub] += 1
            return sub
        if line.strip():
            self.log(f"[node] {line.rstrip()}")
        return None

    def send_decision(self, payload) -> None:
        """Forward one decision message to the node as a single compact line."""
        decision = json.loads(payload)
        self.link.write(f"decision {json.dumps(decision, separators=(',', ':'))}\n".encode())
        self.counts["decision"] += 1

    def run(self, max_lines: Optional[int] = None) -> dict:
        """Relay node lines until the link closes (or `max_lines` uplink messages); returns the counts."""
        while max_lines is None or sum(self.counts[s] for s in UPLINK) < max_lines:
            raw = self.link.readline()
            if not raw:
                break
            self.handle_line(raw.decode("utf-8", errors="replace"))
        return dict(self.counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--tcp", metavar="HOST:PORT", help="the node's serial port on a TCP socket (QEMU)")
    where.add_argument("--device", help="a serial device, e.g. /dev/ttyUSB0 (needs pyserial)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="edgearbiter")
    args = ap.parse_args()

    import paho.mqtt.client as mqtt
    if args.tcp:
        host, port = args.tcp.rsplit(":", 1)
        link = TcpLink(host, int(port))
    else:
        link = DeviceLink(args.device, args.baud)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"serial-bridge-{int(time.time())}")
    bridge = SerialBridge(link, lambda topic, payload: client.publish(topic, payload, qos=0), args.topic)
    client.on_connect = lambda c, u, f, rc, p: c.subscribe(f"{args.topic}/decision")
    client.on_message = lambda c, u, msg: bridge.send_decision(msg.payload)
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()
    try:
        print(f"bridge: node <-> mqtt://{args.broker}:{args.port}/{args.topic}", flush=True)
        counts = bridge.run()
    except KeyboardInterrupt:
        counts = bridge.counts
    finally:
        client.loop_stop(); client.disconnect(); link.close()
    print(f"bridge: {counts}")


if __name__ == "__main__":
    main()
