"""The control loop in one command, without Docker or Node-RED: the policy service with the trained MAPPO
agents (models/demo), an MQTT broker and the virtual ESP32 node, which prints what the agents decide.

    python scripts/demo.py                  # one 300 s episode of `energy_constrained`, two intervals a second
    python scripts/demo.py --interval 1     # real time
    python scripts/demo.py --rules          # the rule-based policies instead, for comparison

Each simulated second the node sends its telemetry (battery, queue, link), the device agent picks the model,
the sensing rate and where to run it, the edge agent admits the offload or not and sets its CPU share, and the
node runs that second in the simulator (same cost table, link, workload and energy model as in training).
The same loop with the Node-RED board: `docker compose up --build`, then http://localhost:1880/dashboard.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTORS = {"MODEL_DEVICE": ROOT / "models" / "demo" / "mappo_energy_constrained_device.pt",
          "MODEL_EDGE": ROOT / "models" / "demo" / "mappo_energy_constrained_edge.pt"}
ARCH = {"mobilenet_v3_small": "MobileNetV3-Small", "mobilenet_v3_large": "MobileNetV3-Large", "resnet18": "ResNet-18"}


def model_name(model: str) -> str:
    arch = next((v for k, v in sorted(ARCH.items(), key=lambda kv: -len(kv[0])) if model.startswith(k)), model)
    return f"{arch} {'INT8' if model.endswith('int8') else 'FP32'}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_service(port: int) -> dict:
    """The policy service (FastAPI) in a thread; returns its /health once it answers."""
    import uvicorn
    server = uvicorn.Server(uvicorn.Config("edgearbiter.api:app", host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="policy-service").start()
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                return json.loads(resp.read())
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("the policy service did not start")


def main():
    ap = argparse.ArgumentParser(description="EdgeArbiter demo: the control loop in one command")
    ap.add_argument("--interval", type=float, default=0.5, help="wall-clock seconds per simulated second")
    ap.add_argument("--steps", type=int, default=300, help="simulated seconds to run (an episode is 300)")
    ap.add_argument("--seed", type=int, default=7, help="seed of the link, workload and edge load")
    ap.add_argument("--rules", action="store_true", help="rule-based policies instead of the trained agents")
    ap.add_argument("--no-mqtt", action="store_true", help="no broker (the node asks the service over HTTP anyway)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)                                       # the service and the node read results/cost_table.json
    sys.path.insert(0, str(ROOT))
    for var, path in ACTORS.items():
        if args.rules:
            os.environ.pop(var, None)
        else:
            os.environ[var] = str(path)

    broker = None
    if not args.no_mqtt:
        try:
            from edgearbiter.broker import start_in_thread
            broker = free_port()
            start_in_thread(port=broker)
        except ImportError:
            broker = None
            print('(no MQTT broker: pip install -e ".[iot]" to add one)')
    port = free_port()
    health = start_service(port)

    from edgearbiter.experiments import SCENARIOS
    from edgearbiter.sim.link import LINK_STATES
    from edgearbiter.virtual_node import VirtualNode

    cfg = replace(SCENARIOS["energy_constrained"], battery_mj=22_800.0)

    def episode_summary(k: dict) -> str:
        why = "the battery is empty" if k["t"] < cfg.horizon else "end of the episode"
        return (f"\nEpisode over at t = {k['t']} s ({why}): mean accuracy {100 * k['mean_accuracy']:.1f} %, "
                f"deadline misses {100 * k['violation_rate']:.1f} %, energy {k['energy_j']:.1f} of 22.8 J, "
                f"{k['processed']} frames processed ({k['offloaded']} offloaded, {k['dropped']} dropped), "
                f"return {k['return']:.1f}\n")

    ended = []                                          # the node reports each episode's KPIs when it ends
    node = VirtualNode("127.0.0.1" if broker else None, broker or 1883, interval=args.interval, timeout=5.0,
                       seed=args.seed, cfg=cfg, api=f"http://127.0.0.1:{port}", log=ended.append)
    agents = (f"trained MAPPO agents ({', '.join(health['actors'])})" if health["actors"] else "rule-based policies")
    print(f"EdgeArbiter demo: scenario energy_constrained, 22.8 J of battery for a 300 s episode, {agents}")
    print(f"policy service http://127.0.0.1:{port}/docs" + (f", MQTT broker 127.0.0.1:{broker} (edgearbiter/#)"
                                                          if broker else ""))
    print("A line when the decision changes, and every 30 simulated seconds. Ctrl+C stops.\n")
    print(f"{'t':>5}  {'battery':>7}  {'link':<22}  {'edge load':>9}  {'decision':<56}  latency")
    last = None
    try:
        for _ in range(args.steps):
            t0 = time.monotonic()
            env = node.env
            link = f"{LINK_STATES[env.link.state]:<4} {env.link.bandwidth_mbps:4.0f} Mbps {env.link.rtt_ms:4.0f} ms"
            battery, edge_load = env.battery, env.edge_load.value
            out = node.step()
            if out["rejected"]:
                where = "offload rejected: tiny model on the device"
            else:
                where = "offloaded, edge admits" if out["where"] == "edge" else "on the device"
            decision = f"{model_name(out['model'])} @ {out['rate_fps']} fps, {where}"
            key = (decision, out["edge_accepting"])
            if key != last or out["t"] % 30 == 0 or ended:
                print(f"{out['t']:>4}s  {100 * battery:5.0f} %  {link:<22}  {100 * edge_load:7.0f} %  {decision:<56}  "
                      f"{out['latency_ms']:4.0f} ms")
                last = key
            if ended:
                print(episode_summary(ended.pop()))
                last = None
            time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
    s = node.summary()
    print(f"{s['decisions']} decisions from the policy service ({s['timeouts']} timeouts), "
          f"mean round trip {s['mean_loop_rtt_ms']} ms")


if __name__ == "__main__":
    main()
