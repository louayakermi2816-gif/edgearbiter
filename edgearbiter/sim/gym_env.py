"""Gymnasium (single-agent) view of the simulator: the device agent, with the edge on a fixed policy.

`DeviceEnv` is a standard `gymnasium.Env` (Box observation, Discrete(18) action), so it works
with any Gymnasium tooling, e.g. Stable-Baselines3, and passes `gymnasium.utils.env_checker`.
The two-agent `EdgeArbiterEnv` stays the reference for IPPO / MAPPO.

    env = DeviceEnv()                       # edge follows the threshold rule
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from ..costs import CostTable
from .env import EdgeArbiterEnv, SimConfig, edge_rule


class DeviceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, costs: Optional[CostTable] = None, cfg: SimConfig = SimConfig(),
                 edge_policy: Callable = edge_rule):
        self.sim = EdgeArbiterEnv(costs, cfg)
        self.edge_policy = edge_policy
        self.observation_space = self.sim.observation_spaces["device"]
        self.action_space = self.sim.action_spaces["device"]
        self._obs = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is None:                                # derive the simulator's stream from Gymnasium's RNG
            seed = int(self.np_random.integers(2 ** 31))
        self._obs, info = self.sim.reset(seed=seed)
        return self._obs["device"], info

    def step(self, action):
        actions = {"device": int(action), "edge": self.edge_policy(self._obs, self.sim)}
        self._obs, reward, terminated, truncated, info = self.sim.step(actions)
        return self._obs["device"], float(reward), terminated, truncated, info
