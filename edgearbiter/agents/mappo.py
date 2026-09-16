"""Multi-agent PPO for the dict-based EdgeArbiter environment.

Two variants, selected by `centralized_critic`:
    IPPO  - each agent has its own actor and its own critic on its own observation
    MAPPO - each agent has its own actor; critics see the joint observation
Both are trained with a shared team reward. Any agent not listed in `controlled`
follows a fixed policy (e.g. the rule-based edge), which is how single-agent
training of the device is done.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(inp: int, out: int, hidden: int, act=nn.Tanh) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), act(), nn.Linear(hidden, hidden), act(), nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01); nn.init.zeros_(self.net[-1].bias)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        d = self.dist(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int((d.probs.argmax(-1) if deterministic else d.sample()).item())


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class MAPPOConfig:
    total_steps: int = 60_000
    rollout_len: int = 600
    epochs: int = 4
    minibatch: int = 100
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    centralized_critic: bool = True     # True = MAPPO, False = IPPO


def gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards); adv = np.zeros(T, np.float32); last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


FixedPolicy = Callable[[Dict[str, np.ndarray], object], int]


def train_mappo(env, cfg: MAPPOConfig = MAPPOConfig(), controlled: Sequence[str] = ("device", "edge"),
                fixed_policies: Optional[Dict[str, FixedPolicy]] = None, log: Optional[Callable] = print):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    fixed_policies = fixed_policies or {}
    obs_dims = {a: env.observation_spaces[a].shape[0] for a in env.observation_spaces}
    joint_dim = sum(obs_dims.values())
    actors = {a: Actor(obs_dims[a], env.action_spaces[a].n, cfg.hidden) for a in controlled}
    critics = {a: Critic(joint_dim if cfg.centralized_critic else obs_dims[a], cfg.hidden) for a in controlled}
    params = [p for a in controlled for p in list(actors[a].parameters()) + list(critics[a].parameters())]
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    def critic_input(a, obs):
        return env.joint_obs(obs) if cfg.centralized_critic else obs[a]

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, history, steps = 0.0, [], 0
    while steps < cfg.total_steps:
        B = {a: {"obs": [], "cobs": [], "act": [], "logp": [], "val": []} for a in controlled}
        rew, done = [], []
        for _ in range(cfg.rollout_len):
            actions = {}
            for a in controlled:
                t_obs = torch.as_tensor(obs[a], dtype=torch.float32)
                with torch.no_grad():
                    d = actors[a].dist(t_obs.unsqueeze(0)); act = d.sample(); logp = d.log_prob(act)
                    val = critics[a](torch.as_tensor(critic_input(a, obs), dtype=torch.float32).unsqueeze(0))
                actions[a] = int(act.item())
                B[a]["obs"].append(obs[a]); B[a]["cobs"].append(critic_input(a, obs))
                B[a]["act"].append(int(act.item())); B[a]["logp"].append(float(logp)); B[a]["val"].append(float(val))
            for a, pol in fixed_policies.items():
                if a not in actions:
                    actions[a] = pol(obs, env)
            next_obs, r, term, trunc, info = env.step(actions)
            rew.append(float(r)); done.append(float(term or trunc)); ep_ret += r; steps += 1
            if term or trunc:
                history.append({"steps": steps, "return": ep_ret, "kpis": info["kpis"]})
                if log:
                    log({"steps": steps, "return": round(ep_ret, 1), **{k: info["kpis"][k] for k in ("mean_accuracy", "violation_rate", "energy_j")}})
                ep_ret = 0.0
                next_obs, _ = env.reset()
            obs = next_obs

        for a in controlled:
            with torch.no_grad():
                last_v = float(critics[a](torch.as_tensor(critic_input(a, obs), dtype=torch.float32).unsqueeze(0)))
            adv, ret = gae(np.array(rew), np.array(B[a]["val"]), np.array(done), last_v, cfg.gamma, cfg.gae_lambda)
            t_obs = torch.as_tensor(np.array(B[a]["obs"]), dtype=torch.float32)
            t_cobs = torch.as_tensor(np.array(B[a]["cobs"]), dtype=torch.float32)
            t_act = torch.as_tensor(B[a]["act"]); t_logp = torch.as_tensor(B[a]["logp"])
            t_adv = torch.as_tensor(adv); t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
            t_ret = torch.as_tensor(ret)
            n = len(t_obs)
            for _ in range(cfg.epochs):
                idx = torch.randperm(n)
                for s in range(0, n, cfg.minibatch):
                    mb = idx[s:s + cfg.minibatch]
                    d = actors[a].dist(t_obs[mb]); logp = d.log_prob(t_act[mb])
                    ratio = torch.exp(logp - t_logp[mb])
                    surr = torch.min(ratio * t_adv[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * t_adv[mb])
                    v = critics[a](t_cobs[mb])
                    loss = -surr.mean() + cfg.vf_coef * (v - t_ret[mb]).pow(2).mean() - cfg.ent_coef * d.entropy().mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(params, cfg.max_grad_norm); opt.step()
    return actors, critics, history


def policies_from_actors(actors: Dict[str, Actor]) -> Dict[str, FixedPolicy]:
    return {a: (lambda obs, env, _m=m, _a=a: _m.act(obs[_a])) for a, m in actors.items()}
