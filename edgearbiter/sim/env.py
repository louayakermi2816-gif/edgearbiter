"""Two-agent IoT-edge simulator (parallel, dict-based API in the style of PettingZoo).

Agents
    device : chooses model tier, sensing rate and placement (local / offload)
    edge   : accepts or rejects the offload request and allocates a CPU share

One step = one second of operation. Frames arrive from a bursty workload, are
queued on the device (bounded queue, drops beyond capacity) and processed either
locally or on the edge server; latency, energy and accuracy come from the cost
table (`edgearbiter.costs`) and from the stochastic link model.

Shared reward per step
    + sum(accuracy of processed frames)
    - a * deadline violations - b * energy (J) - c * dropped frames - d * rejected offload
Episode ends after `horizon` steps or when the battery is empty (penalised).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from ..costs import CostTable
from .link import LINK_STATES, EdgeLoad, Link, Workload

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
    battery_empty_penalty: float = 20.0


class EdgeArbiterEnv:
    def __init__(self, costs: Optional[CostTable] = None, cfg: SimConfig = SimConfig()):
        self.costs = costs or CostTable.load()
        self.cfg = cfg
        self.n_tiers = self.costs.n_tiers
        self.action_spaces = {"device": spaces.Discrete(self.n_tiers * len(RATES) * 2),
                              "edge": spaces.Discrete(2 * len(SHARES))}
        self.observation_spaces = {"device": spaces.Box(0.0, 1.0, shape=(11,), dtype=np.float32),
                                   "edge": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)}
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
            latency = self.link.transfer_ms(costs.frame_kb) + service_ms
            energy_per_frame = costs.tx_energy()
            capacity = max(1, int(1000.0 / max(service_ms, 1.0)))
            used_tier, where = tier, "edge"
        elif place == 1 and not accept:                       # rejected -> fall back to the tiny model locally
            rejected = 1
            used_tier, where = 0, "device"
            latency = costs.latency(0, "device") + self.link.rtt_ms
            energy_per_frame = costs.energy(0) + 0.1 * costs.tx_energy()
            capacity = max(1, int(1000.0 / max(costs.latency(0, "device"), 1.0)))
        else:
            used_tier, where = tier, "device"
            latency = costs.latency(tier, "device")
            energy_per_frame = costs.energy(tier)
            capacity = max(1, int(1000.0 / max(latency, 1.0)))

        processed = min(requested, capacity)
        self.queue -= processed
        violations = processed if latency > cfg.deadline_ms else 0
        energy_mj = processed * energy_per_frame + 2.0            # 2 mJ idle draw
        accuracy_sum = processed * costs.accuracy(used_tier)
        self.battery = max(0.0, self.battery - energy_mj / cfg.battery_mj)

        reward = (accuracy_sum - cfg.w_violation * violations - cfg.w_energy_j * energy_mj / 1000.0
                  - cfg.w_drop * dropped - cfg.w_reject * rejected)
        self.t += 1
        terminated = self.battery <= 0.0
        truncated = self.t >= cfg.horizon and not terminated
        if terminated:
            reward -= cfg.battery_empty_penalty

        # bookkeeping
        self.last = {"arrivals": arrivals, "processed": processed, "violations": violations, "dropped": dropped,
                     "latency_ms": latency, "energy_mj": energy_mj, "offload_request": int(place == 1),
                     "offload_frames": processed if (place == 1 and accept) else 0, "tier": used_tier, "where": where,
                     "rejected": rejected, "edge_load": edge_load}
        self.tot["processed"] += processed; self.tot["violations"] += violations; self.tot["dropped"] += dropped
        self.tot["energy_mj"] += energy_mj; self.tot["accuracy_sum"] += accuracy_sum; self.tot["arrivals"] += arrivals
        self.tot["offloaded"] += self.last["offload_frames"]; self.tot["rejected"] += rejected; self.tot["reward"] += reward
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
                     "offload_request": 0, "offload_frames": 0, "tier": 0, "where": "device", "rejected": 0, "edge_load": 0.3}
        self.tot = {k: 0.0 for k in ("processed", "violations", "dropped", "energy_mj", "accuracy_sum", "arrivals", "offloaded", "rejected", "reward")}

    def _obs(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        onehot = [1.0 if i == self.link.state else 0.0 for i in range(len(LINK_STATES))]
        viol_rate = self.last["violations"] / max(1, self.last["processed"])
        device = [self.battery, self.queue / cfg.max_queue, *onehot, min(1.0, self.link.rtt_ms / 200.0),
                  min(1.0, self.link.bandwidth_mbps / 50.0), min(1.0, self.last["arrivals"] / 12.0), viol_rate,
                  self.edge_load.value, self.t / cfg.horizon]
        edge = [self.edge_load.value, float(self.last["offload_request"]), min(1.0, self.last["offload_frames"] / 10.0),
                *onehot, self.queue / cfg.max_queue, self.t / cfg.horizon]
        return {"device": np.asarray(device, dtype=np.float32), "edge": np.asarray(edge, dtype=np.float32)}

    def joint_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs["device"], obs["edge"]]).astype(np.float32)

    def kpis(self) -> dict:
        p = max(1.0, self.tot["processed"])
        return {"t": self.t, "battery": round(self.battery, 3), "frames_in": int(self.tot["arrivals"]),
                "processed": int(self.tot["processed"]), "dropped": int(self.tot["dropped"]),
                "offloaded": int(self.tot["offloaded"]), "rejected": int(self.tot["rejected"]),
                "mean_accuracy": round(self.tot["accuracy_sum"] / p, 4),
                "violation_rate": round(self.tot["violations"] / p, 4),
                "energy_j": round(self.tot["energy_mj"] / 1000.0, 2), "return": round(self.tot["reward"], 2)}


# ----------------------------------------------------------- fixed edge rule
def edge_rule(obs: Dict[str, np.ndarray], env: EdgeArbiterEnv) -> int:
    """Accept offloads unless the server is busy; give full share when lightly loaded."""
    load = float(obs["edge"][0])
    accept = 1 if load < 0.7 else 0
    share_idx = 1 if load < 0.5 else 0
    return env.encode_edge(accept, share_idx)


DevicePolicy = Callable[[Dict[str, np.ndarray], EdgeArbiterEnv], int]
