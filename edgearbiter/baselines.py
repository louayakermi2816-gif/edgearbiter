"""Static and rule-based device policies used as baselines for the learned agents."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .sim.env import RATES, EdgeArbiterEnv

Policy = Callable[[Dict[str, np.ndarray], EdgeArbiterEnv], int]


def _static(tier: int, rate_idx: int, place: int) -> Policy:
    def _p(obs, env):
        return env.encode_device(min(tier, env.n_tiers - 1), rate_idx, place)
    return _p


def rule_based(obs: Dict[str, np.ndarray], env: EdgeArbiterEnv) -> int:
    """Offload the best model when the link is good and the edge is free; otherwise
    run the tiny model locally. Sense at 10 fps when frames pile up, 5 fps otherwise."""
    o = obs["device"]
    queue, link_good, edge_load = o[1], o[2] > 0.5, o[9]
    rate_idx = 2 if queue > 0.3 else 1
    if link_good and edge_load < 0.7:
        return env.encode_device(env.n_tiers - 1, rate_idx, 1)
    return env.encode_device(0, rate_idx, 0)


BASELINES: Dict[str, Policy] = {
    "always_tiny_local": _static(0, RATES.index(5), 0),
    "always_medium_local": _static(2, RATES.index(5), 0),
    "always_offload_medium": _static(2, RATES.index(10), 1),
    "rule_based": rule_based,
}
