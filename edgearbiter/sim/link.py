"""Stochastic models of the wireless link, the workload and the edge background load."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LINK_STATES = ("good", "fair", "poor")
# (rtt_ms range, bandwidth_mbps range) per link state
LINK_PARAMS = {"good": ((8.0, 20.0), (20.0, 50.0)), "fair": ((25.0, 60.0), (5.0, 15.0)), "poor": ((80.0, 200.0), (0.5, 3.0))}
# Markov transition matrix over LINK_STATES (rows sum to 1)
LINK_TRANSITIONS = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.03, 0.17, 0.80]])
_LINK_CDF = LINK_TRANSITIONS.cumsum(axis=1) / LINK_TRANSITIONS.sum(axis=1, keepdims=True)


def classify_link(rtt_ms: float, bandwidth_mbps: float) -> int:
    """Link state index from a measured RTT and bandwidth (used for real-device telemetry)."""
    if rtt_ms <= 20 and bandwidth_mbps >= 20:
        return 0
    if rtt_ms <= 60 and bandwidth_mbps >= 5:
        return 1
    return 2


@dataclass
class Link:
    state: int = 0
    rtt_ms: float = 12.0
    bandwidth_mbps: float = 30.0

    def step(self, rng: np.random.Generator) -> None:
        # same draw as rng.choice(3, p=LINK_TRANSITIONS[state]) without its per-call validation
        self.state = int(_LINK_CDF[self.state].searchsorted(rng.random(), side="right"))
        (rlo, rhi), (blo, bhi) = LINK_PARAMS[LINK_STATES[self.state]]
        self.rtt_ms = float(rng.uniform(rlo, rhi))
        self.bandwidth_mbps = float(rng.uniform(blo, bhi))

    def transfer_ms(self, frame_kb: float) -> float:
        return self.rtt_ms + frame_kb * 8.0 / self.bandwidth_mbps  # kb*8/Mbps = ms

    def uplink_fps(self, frame_kb: float) -> int:
        """Frames per second the uplink can carry (bandwidth-limited offload throughput)."""
        return int(self.bandwidth_mbps * 1000.0 / (frame_kb * 8.0))


@dataclass
class Workload:
    """Frames arriving per second; alternates between a calm and a busy regime."""
    busy: bool = False
    calm_rate: float = 2.0
    busy_rate: float = 9.0
    p_switch: float = 0.05

    def step(self, rng: np.random.Generator) -> int:
        if rng.random() < self.p_switch:
            self.busy = not self.busy
        return int(rng.poisson(self.busy_rate if self.busy else self.calm_rate))


@dataclass
class EdgeLoad:
    """Background utilisation of the edge server in [0, 1] (bounded random walk)."""
    value: float = 0.3

    def step(self, rng: np.random.Generator) -> float:
        self.value = min(0.95, max(0.0, self.value + float(rng.normal(0.0, 0.05))))
        return self.value


def congestion(load: float) -> float:
    """M/M/1 mean number of background jobs at utilisation `load` (rho / (1 - rho))."""
    return load / (1.0 - load)
