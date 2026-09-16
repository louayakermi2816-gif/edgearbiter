"""Stochastic models of the wireless link, the workload and the edge background load."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

LINK_STATES = ("good", "fair", "poor")
# (rtt_ms range, bandwidth_mbps range) per link state
LINK_PARAMS = {"good": ((8.0, 20.0), (20.0, 50.0)), "fair": ((25.0, 60.0), (5.0, 15.0)), "poor": ((80.0, 200.0), (0.5, 3.0))}
# Markov transition matrix over LINK_STATES (rows sum to 1)
LINK_TRANSITIONS = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.03, 0.17, 0.80]])


@dataclass
class Link:
    state: int = 0
    rtt_ms: float = 12.0
    bandwidth_mbps: float = 30.0

    def step(self, rng: np.random.Generator) -> None:
        self.state = int(rng.choice(3, p=LINK_TRANSITIONS[self.state]))
        (rlo, rhi), (blo, bhi) = LINK_PARAMS[LINK_STATES[self.state]]
        self.rtt_ms = float(rng.uniform(rlo, rhi))
        self.bandwidth_mbps = float(rng.uniform(blo, bhi))

    def transfer_ms(self, frame_kb: float) -> float:
        return self.rtt_ms + frame_kb * 8.0 / self.bandwidth_mbps  # kb*8/Mbps = ms


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
        self.value = float(np.clip(self.value + rng.normal(0.0, 0.05), 0.0, 0.95))
        return self.value
