"""Static and rule-based device policies used as baselines for the learned agents."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .sim.env import DEVICE_OBS, RATES, SHARES, EdgeArbiterEnv, edge_rule

Policy = Callable[[Dict[str, np.ndarray], EdgeArbiterEnv], int]
I = {name: i for i, name in enumerate(DEVICE_OBS)}


def _static(tier: int, rate_idx: int, place: int) -> Policy:
    def _p(obs, env):
        return env.encode_device(min(tier, env.n_tiers - 1), rate_idx, place)
    return _p


def rule_based(obs: Dict[str, np.ndarray], env: EdgeArbiterEnv) -> int:
    """Offload the best model when the link is good and the edge is accepting; otherwise
    run the tiny model locally. Sense at 10 fps when frames pile up, 5 fps otherwise."""
    o = obs["device"]
    queue, link_good, edge_accepting = o[I["queue"]], o[I["link_good"]] > 0.5, o[I["edge_accepting"]] > 0.5
    rate_idx = 2 if queue > 0.3 else 1
    if link_good and edge_accepting:
        return env.encode_device(env.n_tiers - 1, rate_idx, 1)
    return env.encode_device(0, rate_idx, 0)


def deadline_greedy(obs: Dict[str, np.ndarray], env: EdgeArbiterEnv, assumed_edge_load: float = 0.5) -> int:
    """Model-based heuristic: the most accurate (tier, placement) whose latency predicted from the
    cost table and the last link measurement meets the deadline (edge only while it is accepting,
    at the advertised share and an assumed background load). Sensing rate follows the battery:
    10 fps while the charge is ahead of the elapsed time, 5 fps otherwise."""
    o, costs, cfg = obs["device"], env.costs, env.cfg
    rtt, bw = o[I["rtt"]] * 200.0, max(o[I["bandwidth"]] * 50.0, 0.1)
    share = o[I["edge_share"]] if o[I["edge_accepting"]] > 0.5 else 0.0
    best, best_key = env.encode_device(0, 1, 0), None
    for tier in range(env.n_tiers):
        options = [(0, costs.latency(tier, "device"), costs.energy(tier))]
        if share > 0:
            edge_ms = rtt + costs.frame_kb * 8.0 / bw + costs.latency(tier, "edge") / share * (1.0 + assumed_edge_load)
            options.append((1, edge_ms, costs.tx_energy()))
        for place, latency, energy in options:
            if latency <= cfg.deadline_ms:
                key = (costs.accuracy(tier), -energy)
                if best_key is None or key > best_key:
                    best_key, best = key, (tier, place)
    if best_key is None:
        return env.encode_device(0, 1, 0)
    rate_idx = 2 if o[I["battery"]] >= 1.0 - o[I["time"]] else 1
    return env.encode_device(best[0], rate_idx, best[1])


BASELINES: Dict[str, Policy] = {
    "always_tiny_local": _static(0, RATES.index(5), 0),
    "always_medium_local": _static(2, RATES.index(5), 0),
    "always_offload_medium": _static(2, RATES.index(10), 1),
    "rule_based": rule_based,
    "deadline_greedy": deadline_greedy,
}

# edge-side baselines (the device policy is evaluated against one of these unless the edge is learned)
EDGE_BASELINES: Dict[str, Policy] = {
    "edge_rule": edge_rule,
    "always_accept_full": lambda obs, env: env.encode_edge(1, len(SHARES) - 1),
    "always_accept_half": lambda obs, env: env.encode_edge(1, 0),
}

