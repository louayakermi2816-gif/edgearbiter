"""Two-agent IoT-edge simulator (parallel, dict-based API in the style of PettingZoo).

Agents
    device : chooses model tier, sensing rate and placement (local / offload)
    edge   : accepts or rejects offloads and reserves a CPU share for them
Both act simultaneously once per control interval (one step = one second): the
edge decision is an admission/allocation policy for the coming interval.

Observations are local (decentralised execution). The device cannot read the
server's utilisation; it sees the admission decision the edge published in the
previous interval (accepting, CPU share). The edge sees its own load and the
request stream (offload request, offloaded frames, requested tier, the device's
queue and link state carried in the request header). MAPPO's centralised critic
sees both observations during training only.

Frames arrive from a bursty workload, are queued on the device (bounded queue,
drops beyond capacity) and processed either locally or on the edge server.
Latency, energy and accuracy come from the cost table (`edgearbiter.costs`). When the
table has measured p95 latencies, the inference time of a frame is log-normal through
the measured median and p95, and deadline misses are counted in expectation (each frame
misses with the probability that its latency exceeds the deadline); without a p95 the
latency is deterministic and a configuration either meets the deadline or does not.
offloading goes through the stochastic link model (RTT + size / bandwidth) and is
throughput-limited by the uplink bandwidth. The edge also serves background
tenants: reserving a CPU share for the device while the server is busy slows them
down, which is charged as a congestion cost share * rho / (1 - rho) (the M/M/1
backlog at background utilisation rho). This makes the edge decision a real
trade-off: a full share meets deadlines on slow links, half a share is cheaper,
rejecting protects the other tenants when the server is congested.

Shared reward per step
    + sum(accuracy of processed frames)
    - a * deadline violations - b * energy (J) - c * dropped frames - d * rejected offload
    - e * edge congestion cost
Episode ends after `horizon` steps or when the battery is empty (penalised).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from ..costs import CostTable
from .link import LINK_STATES, EdgeLoad, Link, Workload, congestion

RATES = (1, 5, 10)          # sensing rate options (frames / s)
SHARES = (0.5, 1.0)         # edge CPU share options
AGENTS = ("device", "edge")


@dataclass(frozen=True)
class SimConfig:
    horizon: int = 300
    deadline_ms: float = 100.0
    max_queue: int = 30
    battery_mj: float = 60_000.0
    w_violation: float = 0.5
    w_energy_j: float = 0.5
    w_drop: float = 0.3
    w_reject: float = 0.2
    w_edge: float = 0.5               # edge congestion cost of the reserved CPU share
    battery_empty_penalty: float = 20.0


DEVICE_OBS = ("battery", "queue", "link_good", "link_fair", "link_poor", "rtt", "bandwidth", "arrivals",
              "violation_rate", "edge_accepting", "edge_share", "time")
EDGE_OBS = ("edge_load", "offload_request", "offload_frames", "link_good", "link_fair", "link_poor", "device_queue",
            "requested_tier", "time")


def miss_probability(latency_ms: float, deadline_ms: float, sigma: float = 0.0, fixed_ms: float = 0.0) -> float:
    """Probability that a frame misses the deadline. `latency_ms` is the median end-to-end latency, of
    which `fixed_ms` (network) is deterministic and the rest (inference) log-normal with log-spread
    `sigma`; sigma = 0 is a deterministic latency."""
    if sigma <= 0.0:
        return 1.0 if latency_ms > deadline_ms else 0.0
    budget, compute = deadline_ms - fixed_ms, latency_ms - fixed_ms
    if budget <= 0.0:
        return 1.0
    return 0.5 * math.erfc((math.log(budget) - math.log(compute)) / (sigma * math.sqrt(2.0)))


def device_observation(battery: float, queue_frac: float, link_state: int, rtt_ms: float, bandwidth_mbps: float,
                       arrivals: float, violation_rate: float, edge_accepting: float, edge_share: float,
                       t_frac: float) -> np.ndarray:
    """Device observation vector (shared by the simulator and the API's real-device telemetry).

    `edge_accepting` / `edge_share` are the admission decision the edge published last interval
    (share is 0 while the edge is not accepting)."""
    onehot = [1.0 if i == link_state else 0.0 for i in range(len(LINK_STATES))]
    return np.asarray([battery, min(1.0, queue_frac), *onehot, min(1.0, rtt_ms / 200.0), min(1.0, bandwidth_mbps / 50.0),
                       min(1.0, arrivals / 12.0), violation_rate, edge_accepting, edge_share if edge_accepting else 0.0,
                       t_frac], dtype=np.float32)


def edge_observation(edge_load: float, offload_request: float, offload_frames: float, link_state: int, queue_frac: float,
                     requested_tier_frac: float, t_frac: float) -> np.ndarray:
    """Edge observation vector: own load plus what the last request header carried."""
    onehot = [1.0 if i == link_state else 0.0 for i in range(len(LINK_STATES))]
    return np.asarray([edge_load, offload_request, min(1.0, offload_frames / 10.0), *onehot, min(1.0, queue_frac),
                       requested_tier_frac, t_frac], dtype=np.float32)


class EdgeArbiterEnv:
    def __init__(self, costs: Optional[CostTable] = None, cfg: SimConfig = SimConfig()):
        self.costs = costs or CostTable.load()
        self.cfg = cfg
        self.n_tiers = self.costs.n_tiers
        self.action_spaces = {"device": spaces.Discrete(self.n_tiers * len(RATES) * 2),
                              "edge": spaces.Discrete(2 * len(SHARES))}
        # mixed-radix factors of the joint action indices, most significant first (see encode_device / encode_edge)
        self.action_factors = {"device": (self.n_tiers, len(RATES), 2), "edge": (2, len(SHARES))}
        self.observation_spaces = {"device": spaces.Box(0.0, 1.0, shape=(len(DEVICE_OBS),), dtype=np.float32),
                                   "edge": spaces.Box(0.0, 1.0, shape=(len(EDGE_OBS),), dtype=np.float32)}
        self.sigma = {(t, w): self.costs.latency_sigma(t, w) for t in range(self.n_tiers) for w in ("device", "edge")}
        self.rng = np.random.default_rng()
        self._reset_state()

    # ----------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {"kpis": self.kpis()}

    def step(self, actions: Dict[str, int]):
        cfg, costs = self.cfg, self.costs
        tier, rate_idx, place = self.decode_device(actions["device"])
        accept, share_idx = self.decode_edge(actions["edge"])
        rate, share = RATES[rate_idx], SHARES[share_idx]

        # environment dynamics
        self.link.step(self.rng)
        arrivals = self.workload.step(self.rng)
        edge_load = self.edge_load.step(self.rng)
        self.queue += arrivals
        dropped = max(0, self.queue - cfg.max_queue)
        self.queue -= dropped

        # execution of the chosen configuration
        requested = min(self.queue, rate)
        rejected = 0
        if place == 1 and accept:
            service_ms = costs.latency(tier, "edge") / share * (1.0 + edge_load)
            network_ms = self.link.transfer_ms(costs.frame_kb)
            latency = network_ms + service_ms
            energy_per_frame = costs.tx_energy()
            capacity = max(1, min(int(1000.0 / max(service_ms, 1.0)), self.link.uplink_fps(costs.frame_kb)))
            used_tier, where = tier, "edge"
        elif place == 1 and not accept:                       # rejected -> fall back to the tiny model locally
            rejected = 1
            used_tier, where = 0, "device"
            network_ms = self.link.rtt_ms
            latency = costs.latency(0, "device") + network_ms
            energy_per_frame = costs.energy(0) + 0.1 * costs.tx_energy()
            capacity = max(1, int(1000.0 / max(costs.latency(0, "device"), 1.0)))
        else:
            used_tier, where = tier, "device"
            network_ms = 0.0
            latency = costs.latency(tier, "device")
            energy_per_frame = costs.energy(tier)
            capacity = max(1, int(1000.0 / max(latency, 1.0)))

        processed = min(requested, capacity)
        self.queue -= processed
        violations = processed * miss_probability(latency, cfg.deadline_ms, self.sigma[used_tier, where], network_ms)
        energy_mj = processed * energy_per_frame + 2.0            # 2 mJ idle draw
        accuracy_sum = processed * costs.accuracy(used_tier)
        offload_frames = processed if where == "edge" else 0
        edge_cost = share * congestion(edge_load) if offload_frames else 0.0   # reserved share used this interval
        self.battery = max(0.0, self.battery - energy_mj / cfg.battery_mj)

        reward = (accuracy_sum - cfg.w_violation * violations - cfg.w_energy_j * energy_mj / 1000.0
                  - cfg.w_drop * dropped - cfg.w_reject * rejected - cfg.w_edge * edge_cost)
        self.t += 1
        terminated = self.battery <= 0.0
        truncated = self.t >= cfg.horizon and not terminated
        if terminated:
            reward -= cfg.battery_empty_penalty

        # bookkeeping
        self.last = {"arrivals": arrivals, "processed": processed, "violations": violations, "dropped": dropped,
                     "latency_ms": latency, "energy_mj": energy_mj, "offload_request": int(place == 1),
                     "offload_frames": offload_frames, "tier": used_tier, "requested_tier": tier if place == 1 else 0, "where": where,
                     "rate_fps": rate, "rejected": rejected, "share": share if offload_frames else 0.0,
                     "edge_cost": edge_cost, "edge_load": edge_load, "edge_accepting": accept, "edge_share": share}
        self.tot["processed"] += processed; self.tot["violations"] += violations; self.tot["dropped"] += dropped
        self.tot["energy_mj"] += energy_mj; self.tot["accuracy_sum"] += accuracy_sum; self.tot["arrivals"] += arrivals
        self.tot["offloaded"] += offload_frames; self.tot["rejected"] += rejected; self.tot["reward"] += reward
        self.tot["edge_cost"] += edge_cost
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"kpis": self.kpis(), "last": self.last}

    # ------------------------------------------------------------- helpers
    def decode_device(self, a: int) -> Tuple[int, int, int]:
        a = int(a)
        place = a % 2; a //= 2
        rate_idx = a % len(RATES); a //= len(RATES)
        tier = a % self.n_tiers
        return tier, rate_idx, place

    def encode_device(self, tier: int, rate_idx: int, place: int) -> int:
        return (int(tier) * len(RATES) + int(rate_idx)) * 2 + int(place)

    def decode_edge(self, a: int) -> Tuple[int, int]:
        a = int(a)
        return a // len(SHARES), a % len(SHARES)

    def encode_edge(self, accept: int, share_idx: int) -> int:
        return int(accept) * len(SHARES) + int(share_idx)

    def _reset_state(self):
        self.t = 0
        self.battery = 1.0
        self.queue = 0
        self.link, self.workload, self.edge_load = Link(), Workload(), EdgeLoad()
        self.last = {"arrivals": 0, "processed": 0, "violations": 0, "dropped": 0, "latency_ms": 0.0, "energy_mj": 0.0,
                     "offload_request": 0, "offload_frames": 0, "tier": 0, "requested_tier": 0, "where": "device",
                     "rate_fps": 0, "rejected": 0, "share": 0.0, "edge_cost": 0.0, "edge_load": self.edge_load.value,
                     "edge_accepting": 1, "edge_share": SHARES[-1]}      # the edge starts out accepting
        self.tot = {k: 0.0 for k in ("processed", "violations", "dropped", "energy_mj", "accuracy_sum", "arrivals",
                                     "offloaded", "rejected", "reward", "edge_cost")}

    def _obs(self) -> Dict[str, np.ndarray]:
        cfg, last = self.cfg, self.last
        viol_rate = last["violations"] / max(1, last["processed"])
        device = device_observation(self.battery, self.queue / cfg.max_queue, self.link.state, self.link.rtt_ms,
                                    self.link.bandwidth_mbps, last["arrivals"], viol_rate,
                                    last["edge_accepting"], last["edge_share"], self.t / cfg.horizon)
        edge = edge_observation(self.edge_load.value, float(last["offload_request"]), last["offload_frames"],
                                self.link.state, self.queue / cfg.max_queue,
                                last["requested_tier"] / max(1, self.n_tiers - 1), self.t / cfg.horizon)
        return {"device": device, "edge": edge}

    def joint_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs["device"], obs["edge"]]).astype(np.float32)

    def kpis(self) -> dict:
        p = max(1.0, self.tot["processed"])
        return {"t": self.t, "battery": round(self.battery, 3), "frames_in": int(self.tot["arrivals"]),
                "processed": int(self.tot["processed"]), "dropped": int(self.tot["dropped"]),
                "offloaded": int(self.tot["offloaded"]), "rejected": int(self.tot["rejected"]),
                "mean_accuracy": round(self.tot["accuracy_sum"] / p, 4),
                "violation_rate": round(self.tot["violations"] / p, 4),
                "energy_j": round(self.tot["energy_mj"] / 1000.0, 2), "edge_cost": round(self.tot["edge_cost"], 2),
                "return": round(self.tot["reward"], 2)}


# ----------------------------------------------------------- fixed edge rule
def edge_rule(obs: Dict[str, np.ndarray], env: EdgeArbiterEnv) -> int:
    """Accept offloads unless the server is busy; give full share when lightly loaded."""
    load = float(obs["edge"][0])
    accept = 1 if load < 0.7 else 0
    share_idx = 1 if load < 0.5 else 0
    return env.encode_edge(accept, share_idx)


DevicePolicy = Callable[[Dict[str, np.ndarray], EdgeArbiterEnv], int]
