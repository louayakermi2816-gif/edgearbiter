"""Virtual ESP32 node: the IoT device of the loop, in software (no hardware needed).

It speaks the MQTT protocol of `firmware/esp32/edge_node/edge_node.ino` and simulates
the device and its surroundings with the simulator's own models and cost table
(Markov link, bursty workload, energy budget, edge server load):

    publishes  <topic>/telemetry  node, seq, battery, queue, rtt_ms, bandwidth_mbps, arrivals,
                                  violation_rate, t_fraction, uptime_s, tier, placement
               <topic>/edge       {"edge_load"}: what the edge host reports about its own load
               <topic>/node       outcome of each interval + running KPIs (for the Node-RED board)
    receives   <topic>/decision   {"seq", "action", "tier", "rate_fps", "placement", "edge_accept",
                                   "edge_share"} from Node-RED, which calls /policy/decide

Each interval: publish telemetry -> wait for the decision carrying the same `seq` -> run
the interval in the simulator with that decision -> publish the outcome. The time from
telemetry to decision is the control-loop round trip (MQTT + Node-RED + API). With
`--api` the node calls the policy service over HTTP instead of waiting for Node-RED.

    python -m edgearbiter.virtual_node --broker localhost --interval 1.0
    python -m edgearbiter.virtual_node --api http://localhost:8000 --interval 0.2 --steps 300
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import replace
from typing import Callable, Optional

from .costs import CostTable
from .sim.env import RATES, SHARES, EdgeArbiterEnv, SimConfig, edge_rule


class VirtualNode:
    def __init__(self, broker: Optional[str] = "localhost", port: int = 1883, topic: str = "edgearbiter",
                 node_id: str = "virtual-esp32-1", interval: float = 1.0, timeout: Optional[float] = None,
                 seed: int = 0, costs: Optional[CostTable] = None, cfg: SimConfig = SimConfig(),
                 api: Optional[str] = None, log: Optional[Callable] = print):
        self.topic, self.node_id, self.interval, self.api, self.log = topic, node_id, interval, api, log
        self.timeout = interval if timeout is None else timeout
        self.env = EdgeArbiterEnv(costs or CostTable.load(), cfg)
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.seq, self.episode, self.started = 0, 0, time.monotonic()      # durations: immune to clock changes
        self.stats = {"decisions": 0, "timeouts": 0, "rtt_ms_sum": 0.0}
        self._decision: Optional[dict] = None
        self._last_decision: Optional[dict] = None
        self._cv = threading.Condition()
        self.client = None
        if broker:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{node_id}-{int(time.time())}")
            if not api:                                   # decisions arrive over MQTT (from Node-RED)
                self.client.on_connect = lambda c, u, f, rc, p: c.subscribe(f"{topic}/decision")
            self.client.on_message = self._on_message
            for attempt in range(30):                     # the broker may still be starting (docker compose)
                try:
                    self.client.connect(broker, port, keepalive=30)
                    break
                except OSError:
                    if attempt == 29:
                        raise
                    time.sleep(1.0)
            self.client.loop_start()

    # ------------------------------------------------------------- MQTT side
    def _on_message(self, client, userdata, msg):
        try:
            decision = json.loads(msg.payload)
        except ValueError:
            return
        with self._cv:
            self._decision = decision
            self._cv.notify_all()

    def _publish(self, sub: str, payload: dict) -> None:
        if self.client is not None:
            self.client.publish(f"{self.topic}/{sub}", json.dumps(payload), qos=0)

    def telemetry(self) -> dict:
        env, last = self.env, self.env.last
        return {"node": self.node_id, "seq": self.seq, "battery": round(env.battery, 4), "queue": env.queue,
                "rtt_ms": round(env.link.rtt_ms, 1), "bandwidth_mbps": round(env.link.bandwidth_mbps, 2),
                "arrivals": last["arrivals"], "violation_rate": round(last["violations"] / max(1, last["processed"]), 3),
                "t_fraction": round(env.t / env.cfg.horizon, 4), "uptime_s": round(time.monotonic() - self.started, 1),
                "tier": last["tier"], "placement": "edge" if last["where"] == "edge" else "local"}

    def _wait_decision(self, seq: int) -> Optional[dict]:
        deadline = time.monotonic() + self.timeout
        with self._cv:
            while True:
                d = self._decision
                if d is not None and d.get("seq", seq) == seq:
                    self._decision = None
                    return d
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def _ask_api(self, telemetry: dict) -> Optional[dict]:
        """Direct mode: what Node-RED does, as one HTTP call (None if the service does not answer in time)."""
        import urllib.error
        import urllib.request
        body = dict(telemetry, edge_load=self.env.edge_load.value)
        req = urllib.request.Request(f"{self.api.rstrip('/')}/policy/decide", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout or 5.0) as resp:
                decision = dict(json.loads(resp.read()), seq=telemetry["seq"])
        except (urllib.error.URLError, OSError, ValueError):
            return None
        self._publish("decision", decision)            # same stream the hardware node would receive
        return decision

    # ------------------------------------------------------- device actions
    def actions(self, decision: Optional[dict]) -> dict:
        """Joint action for the simulator from a decision (the firmware keeps the last one on timeout)."""
        env = self.env
        if decision is None:
            decision = self._last_decision
        if decision is None:                                  # nothing received yet: tiny model locally
            return {"device": env.encode_device(0, RATES.index(5), 0), "edge": edge_rule(self.obs, env)}
        if "action" in decision:
            device = int(decision["action"])
        else:                                                 # firmware-style decision
            device = env.encode_device(int(decision["tier"]), RATES.index(int(decision["rate_fps"])),
                                       int(decision.get("placement") == "edge"))
        if "edge_accept" in decision:
            edge = env.encode_edge(int(bool(decision["edge_accept"])), SHARES.index(float(decision["edge_share"])))
        else:
            edge = edge_rule(self.obs, env)
        return {"device": device, "edge": edge}

    # ------------------------------------------------------------ main loop
    def step(self) -> dict:
        """One control interval: telemetry -> decision -> simulated execution -> outcome."""
        t0 = time.monotonic()
        self.seq += 1
        tel = self.telemetry()
        self._publish("edge", {"edge_load": round(self.env.edge_load.value, 4)})
        self._publish("telemetry", tel)
        decision = self._ask_api(tel) if self.api else self._wait_decision(tel["seq"])
        rtt_ms = (time.monotonic() - t0) * 1000.0
        if decision is None:
            self.stats["timeouts"] += 1
        else:
            self.stats["decisions"] += 1; self.stats["rtt_ms_sum"] += rtt_ms
            self._last_decision = decision
        self.obs, reward, term, trunc, info = self.env.step(self.actions(decision))
        last = info["last"]
        out = {"node": self.node_id, "seq": tel["seq"], "episode": self.episode, "t": self.env.t, "reward": round(reward, 3),
               "decision": "timeout" if decision is None else "ok", "loop_rtt_ms": round(rtt_ms, 1),
               "tier": last["tier"], "model": self.env.costs.tiers[last["tier"]]["model"], "where": last["where"],
               "rate_fps": last["rate_fps"], "processed": last["processed"], "latency_ms": round(last["latency_ms"], 1),
               "miss_rate": round(last["violations"] / max(1, last["processed"]), 3), "rejected": bool(last["rejected"]),
               "edge_accepting": last["edge_accepting"], "edge_load": round(last["edge_load"], 3),
               "battery": round(self.env.battery, 4), "kpis": info["kpis"]}
        self._publish("node", out)
        if term or trunc:
            self._publish("node/episode", {"node": self.node_id, "episode": self.episode, "kpis": info["kpis"]})
            if self.log:
                self.log({"episode": self.episode, **info["kpis"]})
            self.episode += 1
            self.obs, _ = self.env.reset(seed=self.seed + self.episode)
        return out

    def run(self, steps: Optional[int] = None) -> dict:
        n = 0
        try:
            while steps is None or n < steps:
                t0 = time.monotonic()
                self.step(); n += 1
                time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        return self.summary()

    def summary(self) -> dict:
        s = self.stats
        return {"intervals": self.seq, "decisions": s["decisions"], "timeouts": s["timeouts"],
                "mean_loop_rtt_ms": round(s["rtt_ms_sum"] / max(1, s["decisions"]), 1), "kpis": self.env.kpis()}

    def close(self) -> None:
        if self.client is not None:
            self.client.loop_stop(); self.client.disconnect()


def main():
    from .experiments import SCENARIOS
    ap = argparse.ArgumentParser(description="Virtual ESP32 node (software stand-in for the hardware node)")
    ap.add_argument("--broker", default="localhost", help="MQTT broker host ('' to disable MQTT)")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="edgearbiter")
    ap.add_argument("--node-id", default="virtual-esp32-1")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds per control interval (simulated second)")
    ap.add_argument("--timeout", type=float, default=None, help="max wait for a decision (default: one interval)")
    ap.add_argument("--steps", type=int, default=None, help="stop after this many intervals (default: run forever)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="default")
    ap.add_argument("--battery-j", type=float, default=None,
                    help="energy budget per episode instead of the scenario's (e.g. the one a study was trained with)")
    ap.add_argument("--api", default=None, help="call the policy service directly, e.g. http://localhost:8000")
    ap.add_argument("--cost-table", default=None)
    args = ap.parse_args()
    cfg = SCENARIOS[args.scenario]
    if args.battery_j:
        cfg = replace(cfg, battery_mj=1000.0 * args.battery_j)
    node = VirtualNode(args.broker or None, args.port, args.topic, args.node_id, args.interval, args.timeout, args.seed,
                       CostTable.load(args.cost_table), cfg, api=args.api)
    print(f"[virtual-node] {args.node_id}: broker={args.broker or '-'} api={args.api or '-'} interval={args.interval}s")
    try:
        print(json.dumps(node.run(args.steps), indent=1))
    finally:
        node.close()


if __name__ == "__main__":
    main()
