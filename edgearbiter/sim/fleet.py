"""Several devices sharing one edge server: the multi-device version of `EdgeArbiterEnv`.

Agents: device0 .. device{N-1} and edge, acting simultaneously once per second.
Each device is the device agent of `sim/env.py` (same 18 actions, same 12-feature
observation) with its own workload, queue, battery and radio link; the devices differ by
their distance to the access point (`LINK_PROFILES`: near / mid / far). They share the edge,
which sits behind their access point: the k devices offloading in the same interval split
its airtime and the CPU share reserved for them, so each sees k times the transfer and the
service time. The devices cannot see each other; only the edge sees every request. It
admits or rejects each device separately (2^N admission masks) and sets the CPU share.

Team reward: the devices' rewards (as in `EdgeArbiterEnv`) summed, minus the edge congestion
cost. A device whose battery is empty is out for the rest of the episode (penalty once);
the episode ends at the horizon or when every device is out. The random draws do not depend
on the actions, so every policy sees the same links and workloads for a given seed.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from gymnasium import spaces

from ..costs import CostTable
from .env import DEVICE_OBS, RATES, SHARES, EdgeArbiterEnv, SimConfig, device_observation, miss_probability
from .link import LINK_PROFILES, LINK_STATES, EdgeLoad, Link, Workload, congestion, link_cdf

PROFILES = ("near", "mid", "far")
EDGE_FIELDS = ("offload_request", "offload_frames", "link_good", "link_fair", "link_poor", "queue", "requested_tier")


class FleetEnv:
    decode_device = EdgeArbiterEnv.decode_device       # same device action encoding as the single-device env
    encode_device = EdgeArbiterEnv.encode_device

    def __init__(self, costs: Optional[CostTable] = None, cfg: SimConfig = SimConfig(), profiles: Sequence[str] = PROFILES):
        self.costs = costs or CostTable.load()
        self.cfg, self.profiles = cfg, tuple(profiles)
        self.n_devices, self.n_tiers = len(self.profiles), self.costs.n_tiers
        self.devices = [f"device{i}" for i in range(self.n_devices)]
        self._cdfs = [link_cdf(LINK_PROFILES[p]) for p in self.profiles]
        self.action_spaces = {**{d: spaces.Discrete(self.n_tiers * len(RATES) * 2) for d in self.devices},
                              "edge": spaces.Discrete(2 ** self.n_devices * len(SHARES))}
        self.observation_spaces = {
            **{d: spaces.Box(0.0, 1.0, shape=(len(DEVICE_OBS),), dtype=np.float32) for d in self.devices},
            "edge": spaces.Box(0.0, 1.0, shape=(2 + len(EDGE_FIELDS) * self.n_devices,), dtype=np.float32)}
        self.sigma = {(t, w): self.costs.latency_sigma(t, w) for t in range(self.n_tiers) for w in ("device", "edge")}
        self.rng = np.random.default_rng()
        self._reset_state()

    # ----------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {"kpis": self.kpis()}

    def encode_edge(self, mask: int, share_idx: int) -> int:
        """`mask` bit i admits device i's offloads for the coming interval."""
        return int(mask) * len(SHARES) + int(share_idx)

    def decode_edge(self, a: int):
        return int(a) // len(SHARES), int(a) % len(SHARES)

    def step(self, actions: Dict[str, int]):
        cfg, costs, n = self.cfg, self.costs, self.n_devices
        mask, share_idx = self.decode_edge(actions["edge"])
        share = SHARES[share_idx]
        plan = [self.decode_device(actions[d]) for d in self.devices]

        arrivals = []                                  # exogenous dynamics: same draws whatever the actions
        for link, workload in zip(self.links, self.workloads):
            link.step(self.rng)
            arrivals.append(workload.step(self.rng))
        edge_load = self.edge_load.step(self.rng)

        admitted = [self.alive[i] and plan[i][2] == 1 and bool(mask >> i & 1) for i in range(n)]
        k = max(1, sum(admitted))                      # devices sharing the airtime and the reserved CPU
        reward = 0.0
        for i in range(n):
            tier, rate_idx, place = plan[i]
            if not self.alive[i]:
                self.last[i] = self._idle(arrivals[i])
                continue
            link = self.links[i]
            self.queue[i] += arrivals[i]
            dropped = max(0, self.queue[i] - cfg.max_queue)
            self.queue[i] -= dropped
            requested, rejected = min(self.queue[i], RATES[rate_idx]), 0
            if admitted[i]:
                service_ms = costs.latency(tier, "edge") / (share / k) * (1.0 + edge_load)
                network_ms = link.rtt_ms + costs.frame_kb * 8.0 * k / link.bandwidth_mbps
                latency, energy_per_frame = network_ms + service_ms, costs.tx_energy()
                capacity = max(1, min(int(1000.0 / max(service_ms, 1.0)), link.uplink_fps(costs.frame_kb) // k))
                used_tier, where = tier, "edge"
            elif place == 1:                           # rejected -> the tiny model locally, after a wasted round trip
                rejected, used_tier, where = 1, 0, "device"
                network_ms = link.rtt_ms
                latency = costs.latency(0, "device") + network_ms
                energy_per_frame = costs.energy(0) + 0.1 * costs.tx_energy()
                capacity = max(1, int(1000.0 / max(costs.latency(0, "device"), 1.0)))
            else:
                used_tier, where, network_ms = tier, "device", 0.0
                latency, energy_per_frame = costs.latency(tier, "device"), costs.energy(tier)
                capacity = max(1, int(1000.0 / max(latency, 1.0)))
            processed = min(requested, capacity)
            self.queue[i] -= processed
            violations = processed * miss_probability(latency, cfg.deadline_ms, self.sigma[used_tier, where], network_ms)
            energy_mj = processed * energy_per_frame + 2.0
            accuracy_sum = processed * costs.accuracy(used_tier)
            self.battery[i] = max(0.0, self.battery[i] - energy_mj / cfg.battery_mj)
            r = (accuracy_sum - cfg.w_violation * violations - cfg.w_energy_j * energy_mj / 1000.0
                 - cfg.w_drop * dropped - cfg.w_reject * rejected)
            if self.battery[i] <= 0.0:
                self.alive[i] = False
                r -= cfg.battery_empty_penalty
            reward += r
            offload_frames = processed if where == "edge" else 0
            self.last[i] = {"arrivals": arrivals[i], "processed": processed, "violations": violations, "dropped": dropped,
                            "latency_ms": latency, "energy_mj": energy_mj, "offload_request": int(place == 1),
                            "offload_frames": offload_frames, "tier": used_tier, "requested_tier": tier if place == 1 else 0,
                            "where": where, "rate_fps": RATES[rate_idx], "rejected": rejected}
            for key, v in (("processed", processed), ("violations", violations), ("dropped", dropped),
                           ("energy_mj", energy_mj), ("accuracy_sum", accuracy_sum), ("offloaded", offload_frames),
                           ("rejected", rejected), ("arrivals", arrivals[i])):
                self.tot[key] += v
        offloading = sum(1 for o in self.last if o["offload_frames"])
        edge_cost = share * congestion(edge_load) if offloading else 0.0
        reward -= cfg.w_edge * edge_cost
        self.edge_last = {"mask": mask, "share": share, "edge_load": edge_load, "edge_cost": edge_cost,
                          "sharing": offloading}
        self.tot["edge_cost"] += edge_cost; self.tot["reward"] += reward
        self.t += 1
        terminated = not any(self.alive)
        truncated = self.t >= cfg.horizon and not terminated
        return self._obs(), float(reward), terminated, truncated, {"kpis": self.kpis(),
                                                                   "last": {"devices": self.last, "edge": self.edge_last}}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _idle(arrivals: int = 0) -> dict:
        return {"arrivals": arrivals, "processed": 0, "violations": 0.0, "dropped": 0, "latency_ms": 0.0, "energy_mj": 0.0,
                "offload_request": 0, "offload_frames": 0, "tier": 0, "requested_tier": 0, "where": "device",
                "rate_fps": 0, "rejected": 0}

    def _reset_state(self):
        n = self.n_devices
        self.t = 0
        self.links = [Link(cdf=c) for c in self._cdfs]
        self.workloads = [Workload() for _ in range(n)]
        self.edge_load = EdgeLoad()
        self.queue, self.battery, self.alive = [0] * n, [1.0] * n, [True] * n
        self.last = [self._idle() for _ in range(n)]
        self.edge_last = {"mask": 2 ** n - 1, "share": SHARES[-1], "edge_load": self.edge_load.value, "edge_cost": 0.0,
                          "sharing": 0}                 # the edge starts out admitting everyone
        self.tot = {k: 0.0 for k in ("processed", "violations", "dropped", "energy_mj", "accuracy_sum", "arrivals",
                                     "offloaded", "rejected", "reward", "edge_cost")}

    def _obs(self) -> Dict[str, np.ndarray]:
        cfg, mask, t_frac = self.cfg, self.edge_last["mask"], self.t / self.cfg.horizon
        obs = {}
        edge = [self.edge_load.value]
        for i, d in enumerate(self.devices):
            last, link = self.last[i], self.links[i]
            accepting = float(mask >> i & 1)
            obs[d] = device_observation(self.battery[i], self.queue[i] / cfg.max_queue, link.state, link.rtt_ms,
                                        link.bandwidth_mbps, last["arrivals"], last["violations"] / max(1, last["processed"]),
                                        accepting, self.edge_last["share"], t_frac)
            onehot = [1.0 if s == link.state else 0.0 for s in range(len(LINK_STATES))]
            edge += [float(last["offload_request"]), min(1.0, last["offload_frames"] / 10.0), *onehot,
                     min(1.0, self.queue[i] / cfg.max_queue), last["requested_tier"] / max(1, self.n_tiers - 1)]
        obs["edge"] = np.asarray(edge + [t_frac], dtype=np.float32)
        return obs

    def kpis(self) -> dict:
        p = max(1.0, self.tot["processed"])
        return {"t": self.t, "battery": round(float(np.mean(self.battery)), 3), "alive": int(sum(self.alive)),
                "frames_in": int(self.tot["arrivals"]), "processed": int(self.tot["processed"]),
                "dropped": int(self.tot["dropped"]), "offloaded": int(self.tot["offloaded"]),
                "rejected": int(self.tot["rejected"]), "mean_accuracy": round(self.tot["accuracy_sum"] / p, 4),
                "violation_rate": round(self.tot["violations"] / p, 4), "energy_j": round(self.tot["energy_mj"] / 1000.0, 2),
                "edge_cost": round(self.tot["edge_cost"], 2), "return": round(self.tot["reward"], 2)}


# ------------------------------------------------------------ fixed policies
def fleet_edge_rule(obs: Dict[str, np.ndarray], env: FleetEnv) -> int:
    """The single-device edge rule for everyone: admit all below 70 % background load, full share below 50 %."""
    load = float(obs["edge"][0])
    return env.encode_edge(2 ** env.n_devices - 1 if load < 0.7 else 0, 1 if load < 0.5 else 0)


def link_aware_edge_rule(obs: Dict[str, np.ndarray], env: FleetEnv) -> int:
    """Admit only the devices that report a good link (below 70 % background load), full share below 50 %."""
    o, load = obs["edge"], float(obs["edge"][0])
    good = EDGE_FIELDS.index("link_good")
    mask = 0 if load >= 0.7 else sum(1 << i for i in range(env.n_devices) if o[1 + len(EDGE_FIELDS) * i + good] > 0.5)
    return env.encode_edge(mask, 1 if load < 0.5 else 0)


def on_every_device(policy: Callable, env: FleetEnv) -> Dict[str, Callable]:
    """Run a single-device policy (it reads obs["device"]) on each device of the fleet."""
    return {d: (lambda obs, e, _d=d: policy({"device": obs[_d], "edge": obs["edge"]}, e)) for d in env.devices}
