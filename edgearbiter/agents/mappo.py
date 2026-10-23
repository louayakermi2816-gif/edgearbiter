"""Multi-agent PPO for the dict-based EdgeArbiter environment.

Two variants, selected by `centralized_critic`:
    IPPO  - each agent has its own actor and its own critic on its own observation
    MAPPO - each agent has its own actor; one shared critic sees the joint observation
            (centralised training, decentralised execution)
Both are trained with the shared team reward. Any agent not listed in `controlled`
follows a fixed policy (e.g. the rule-based edge), which is how single-agent
training of the device is done.

Implementation choices follow Yu et al., "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games" (NeurIPS 2022): value targets are normalised with
running statistics, actors and critics have separate optimisers and gradient
clipping, rollouts come from `n_envs` simulator copies stepped in lock-step, and
the learning rate is annealed linearly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(inp: int, out: int, hidden: int, act=nn.Tanh) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), act(), nn.Linear(hidden, hidden), act(), nn.Linear(hidden, out))


class FactoredCategorical:
    """Independent categoricals over the digits of a mixed-radix joint action index.

    factors=(n_tiers, n_rates, 2) turns the device's joint index into separate tier, rate and
    placement heads; factors=(n,) is a plain categorical over n actions."""

    def __init__(self, logits: torch.Tensor, factors: Sequence[int]):
        self.factors = tuple(factors)
        self.parts = [Categorical(logits=l, validate_args=False) for l in torch.split(logits, self.factors, dim=-1)]
        self.place = [int(np.prod(self.factors[i + 1:])) for i in range(len(self.factors))]

    def _digits(self, a: torch.Tensor):
        return [torch.div(a, w, rounding_mode="floor") % f for w, f in zip(self.place, self.factors)]

    def sample(self) -> torch.Tensor:
        return sum(p.sample() * w for p, w in zip(self.parts, self.place))

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return sum(p.log_prob(d) for p, d in zip(self.parts, self._digits(a)))

    def entropy(self) -> torch.Tensor:
        return sum(p.entropy() for p in self.parts)

    def mode(self) -> torch.Tensor:
        return sum(p.logits.argmax(-1) * w for p, w in zip(self.parts, self.place))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, factors: Optional[Sequence[int]] = None):
        super().__init__()
        self.factors = tuple(int(f) for f in (factors or (n_actions,)))     # plain ints: checkpoints load weights_only
        if int(np.prod(self.factors)) != n_actions:
            raise ValueError(f"factors {self.factors} do not multiply to {n_actions} actions")
        self.net = mlp(obs_dim, sum(self.factors), hidden)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01); nn.init.zeros_(self.net[-1].bias)

    def dist(self, obs: torch.Tensor) -> FactoredCategorical:
        return FactoredCategorical(self.net(obs), self.factors)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        d = self.dist(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int((d.mode() if deterministic else d.sample()).item())


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class RunningNorm:
    """Running mean / variance of the value targets (parallel Welford update)."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        bm, bv, n = float(x.mean()), float(x.var()), x.size
        d, tot = bm - self.mean, self.count + n
        self.mean += d * n / tot
        self.var = (self.var * self.count + bv * n + d * d * self.count * n / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


@dataclass
class MAPPOConfig:
    total_steps: int = 60_000           # environment steps, summed over the parallel envs
    n_envs: int = 8
    rollout_len: int = 128              # steps per env per update (batch = n_envs * rollout_len)
    epochs: int = 5
    minibatch: int = 256
    lr: float = 3e-4                    # actor learning rate
    critic_lr: float = 1e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 128
    factorized: bool = False            # one policy head per action factor (env.action_factors) instead of one softmax
    value_norm: bool = True
    seed: int = 0
    centralized_critic: bool = True     # True = MAPPO, False = IPPO
    callback_every: int = 0             # env steps between `callback(steps, actors)` calls (0 = never)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over arrays of shape (T,) or (T, n_envs)."""
    T = len(rewards); adv = np.zeros_like(np.asarray(rewards), dtype=np.float32); last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


FixedPolicy = Callable[[Dict[str, np.ndarray], object], int]


def train_mappo(env, cfg: MAPPOConfig = MAPPOConfig(), controlled: Sequence[str] = ("device", "edge"),
                fixed_policies: Optional[Dict[str, FixedPolicy]] = None, log: Optional[Callable] = print,
                callback: Optional[Callable[[int, Dict[str, Actor]], None]] = None):
    """Train the `controlled` agents; returns (actors, critics, history of training episodes).

    With a centralised critic, `critics` maps every controlled agent to the same shared critic."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    fixed_policies = fixed_policies or {}
    agents = list(env.observation_spaces)
    envs = [env] + [copy.deepcopy(env) for _ in range(cfg.n_envs - 1)]
    obs_dims = {a: env.observation_spaces[a].shape[0] for a in agents}
    factors = getattr(env, "action_factors", {}) if cfg.factorized else {}
    actors = {a: Actor(obs_dims[a], env.action_spaces[a].n, cfg.hidden, factors.get(a)) for a in controlled}
    if cfg.centralized_critic:
        critics = {"team": Critic(sum(obs_dims.values()), cfg.hidden)}
        value_of = {a: "team" for a in controlled}
    else:
        critics = {a: Critic(obs_dims[a], cfg.hidden) for a in controlled}
        value_of = {a: a for a in controlled}
    norms = {k: RunningNorm() for k in critics}
    a_opt = {a: torch.optim.Adam(actors[a].parameters(), lr=cfg.lr, eps=1e-5) for a in controlled}
    c_opt = {k: torch.optim.Adam(c.parameters(), lr=cfg.critic_lr, eps=1e-5) for k, c in critics.items()}

    def critic_inputs(ob: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        joint = np.concatenate([ob[a] for a in agents], axis=-1)
        return {k: joint if k == "team" else ob[k] for k in critics}

    def values(cin: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            out = {k: critics[k](torch.as_tensor(cin[k], dtype=torch.float32)).numpy() for k in critics}
        return {k: norms[k].denormalize(v) if cfg.value_norm else v for k, v in out.items()}

    obs_list = [e.reset(seed=cfg.seed * 10_000 + i)[0] for i, e in enumerate(envs)]
    ep_ret = np.zeros(cfg.n_envs)
    history: List[dict] = []
    steps, update, next_callback = 0, 0, cfg.callback_every
    n_updates = max(1, -(-cfg.total_steps // (cfg.n_envs * cfg.rollout_len)))
    T, N = cfg.rollout_len, cfg.n_envs
    while steps < cfg.total_steps:
        update += 1
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            for o in a_opt.values():
                o.param_groups[0]["lr"] = frac * cfg.lr
            for o in c_opt.values():
                o.param_groups[0]["lr"] = frac * cfg.critic_lr
        buf_obs = {a: np.zeros((T, N, obs_dims[a]), np.float32) for a in controlled}
        buf_cin = {k: np.zeros((T, N, critics[k].net[0].in_features), np.float32) for k in critics}
        buf_act = {a: np.zeros((T, N), np.int64) for a in controlled}
        buf_logp = {a: np.zeros((T, N), np.float32) for a in controlled}
        buf_val = {k: np.zeros((T, N), np.float32) for k in critics}
        rew, done = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        finished = []
        for t in range(T):
            ob = {a: np.stack([o[a] for o in obs_list]) for a in agents}
            cin = critic_inputs(ob)
            for k, v in values(cin).items():
                buf_cin[k][t], buf_val[k][t] = cin[k], v
            actions = [dict() for _ in range(N)]
            for a in controlled:
                with torch.no_grad():
                    d = actors[a].dist(torch.as_tensor(ob[a])); act = d.sample(); logp = d.log_prob(act)
                act = act.numpy()
                buf_obs[a][t], buf_act[a][t], buf_logp[a][t] = ob[a], act, logp.numpy()
                for i in range(N):
                    actions[i][a] = int(act[i])
            for i, e in enumerate(envs):
                for a, pol in fixed_policies.items():
                    if a not in actions[i]:
                        actions[i][a] = pol(obs_list[i], e)
                next_obs, r, term, trunc, info = e.step(actions[i])
                rew[t, i], done[t, i] = r, float(term or trunc); ep_ret[i] += r
                if term or trunc:
                    finished.append({"steps": steps + (t + 1) * N, "return": float(ep_ret[i]), "kpis": info["kpis"]})
                    ep_ret[i] = 0.0
                    next_obs, _ = e.reset()
                obs_list[i] = next_obs
        steps += T * N
        history += finished
        if log and finished:
            mean = lambda key: round(float(np.mean([f["kpis"][key] for f in finished])), 3)
            log({"steps": steps, "episodes": len(finished), "return": round(float(np.mean([f["return"] for f in finished])), 1),
                 "mean_accuracy": mean("mean_accuracy"), "violation_rate": mean("violation_rate"), "energy_j": mean("energy_j")})

        # advantages and value targets (raw reward scale), then normalised targets for the critics
        last_v = values(critic_inputs({a: np.stack([o[a] for o in obs_list]) for a in agents}))
        adv, ret_n = {}, {}
        for k in critics:
            adv_k, ret_k = gae(rew, buf_val[k], done, last_v[k], cfg.gamma, cfg.gae_lambda)
            if cfg.value_norm:
                norms[k].update(ret_k)
                ret_k = norms[k].normalize(ret_k)
            adv_k = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)
            adv[k], ret_n[k] = torch.as_tensor(adv_k.reshape(-1)), torch.as_tensor(ret_k.reshape(-1), dtype=torch.float32)
        t_cin = {k: torch.as_tensor(buf_cin[k].reshape(T * N, -1)) for k in critics}
        t_obs = {a: torch.as_tensor(buf_obs[a].reshape(T * N, -1)) for a in controlled}
        t_act = {a: torch.as_tensor(buf_act[a].reshape(-1)) for a in controlled}
        t_logp = {a: torch.as_tensor(buf_logp[a].reshape(-1)) for a in controlled}
        B = T * N
        for _ in range(cfg.epochs):
            perm = torch.randperm(B)
            for s in range(0, B, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                for k, critic in critics.items():
                    loss_v = (critic(t_cin[k][mb]) - ret_n[k][mb]).pow(2).mean()
                    c_opt[k].zero_grad(); loss_v.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm); c_opt[k].step()
                for a in controlled:
                    d = actors[a].dist(t_obs[a][mb])
                    ratio = torch.exp(d.log_prob(t_act[a][mb]) - t_logp[a][mb])
                    adv_mb = adv[value_of[a]][mb]
                    surr = torch.min(ratio * adv_mb, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_mb)
                    loss_pi = -surr.mean() - cfg.ent_coef * d.entropy().mean()
                    a_opt[a].zero_grad(); loss_pi.backward()
                    nn.utils.clip_grad_norm_(actors[a].parameters(), cfg.max_grad_norm); a_opt[a].step()
        if callback and cfg.callback_every and steps >= next_callback:
            callback(steps, actors)
            next_callback += cfg.callback_every
    return actors, {a: critics[value_of[a]] for a in controlled}, history


def policies_from_actors(actors: Dict[str, Actor]) -> Dict[str, FixedPolicy]:
    return {a: (lambda obs, env, _m=m, _a=a: _m.act(obs[_a])) for a, m in actors.items()}


def save_actor(actor: Actor, path, **meta) -> None:
    """Checkpoint with the shapes needed to rebuild the actor (plus free-form metadata)."""
    net = actor.net
    torch.save({"state_dict": actor.state_dict(), "obs_dim": net[0].in_features, "n_actions": int(np.prod(actor.factors)),
                "factors": list(actor.factors), "hidden": net[0].out_features, **meta}, path)


def load_actor(path) -> Actor:
    """Load a checkpoint written by `save_actor` (or a bare state_dict of a single-softmax actor)."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    w0, wl = sd["net.0.weight"], sd["net.4.weight"]
    factors = ckpt.get("factors") or [wl.shape[0]]
    actor = Actor(w0.shape[1], int(np.prod(factors)), w0.shape[0], factors)
    actor.load_state_dict(sd); actor.eval()
    return actor
