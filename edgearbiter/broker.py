"""A local MQTT broker in pure Python (amqtt), for running the software IoT loop without
Mosquitto or Docker. `docker compose up` uses Mosquitto instead.

    pip install -e ".[iot]"        # amqtt
    python -m edgearbiter.broker --port 1883
"""
from __future__ import annotations

import argparse
import asyncio
import threading


def config(host: str = "0.0.0.0", port: int = 1883) -> dict:
    # anonymous clients only: no $SYS topics or packet logging, which amqtt loads by default
    return {"listeners": {"default": {"type": "tcp", "bind": f"{host}:{port}"}},
            "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}}


async def serve(host: str = "0.0.0.0", port: int = 1883, ready: threading.Event = None) -> None:
    from amqtt.broker import Broker
    broker = Broker(config(host, port))
    await broker.start()
    if ready is not None:
        ready.set()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await broker.shutdown()


def start_in_thread(host: str = "127.0.0.1", port: int = 1883, timeout: float = 10.0) -> threading.Thread:
    """Run the broker in a daemon thread (tests, demos); returns once it accepts connections."""
    ready = threading.Event()
    thread = threading.Thread(target=lambda: asyncio.run(serve(host, port, ready)), daemon=True, name="mqtt-broker")
    thread.start()
    if not ready.wait(timeout):
        raise RuntimeError(f"MQTT broker did not start on {host}:{port}")
    return thread


def main():
    ap = argparse.ArgumentParser(description="Local MQTT broker (amqtt)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()
    print(f"[broker] MQTT on {args.host}:{args.port}")
    try:
        asyncio.run(serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
