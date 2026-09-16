"""Seeded evaluation of (device, edge) policy pairs and Markdown result tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .costs import CostTable
from .sim.env import EdgeArbiterEnv, SimConfig, edge_rule

KPIS = ["return", "mean_accuracy", "violation_rate", "energy_j", "processed", "dropped", "offloaded", "battery"]


def run_episode(env: EdgeArbiterEnv, device_policy: Callable, edge_policy: Callable, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step({"device": device_policy(obs, env), "edge": edge_policy(obs, env)})
        done = term or trunc
    return {k: float(info["kpis"][k]) for k in KPIS}


def evaluate(device_policy: Callable, edge_policy: Callable = edge_rule, costs: Optional[CostTable] = None,
             cfg: SimConfig = SimConfig(), n_episodes: int = 20, seed0: int = 1000) -> Dict[str, np.ndarray]:
    env = EdgeArbiterEnv(costs, cfg)
    rows = [run_episode(env, device_policy, edge_policy, seed0 + i) for i in range(n_episodes)]
    return {k: np.array([r[k] for r in rows]) for k in KPIS}


def summarize(results: Dict[str, Dict[str, np.ndarray]]) -> str:
    cols = ["return", "mean_accuracy", "violation_rate", "energy_j", "processed", "dropped", "offloaded"]
    lines = ["| policy | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, res in results.items():
        fmt = lambda c: f"{res[c].mean():.3f} ± {res[c].std():.3f}" if c in ("mean_accuracy", "violation_rate") else f"{res[c].mean():.1f} ± {res[c].std():.1f}"
        lines.append(f"| {name} | " + " | ".join(fmt(c) for c in cols) + " |")
    return "\n".join(lines)


def save_results(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(summarize(results) + "\n", encoding="utf-8")
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "episode"] + KPIS)
        for p, res in results.items():
            for i in range(len(res["return"])):
                w.writerow([p, i] + [float(res[k][i]) for k in KPIS])
    (out_dir / f"{name}.json").write_text(json.dumps({p: {k: v.tolist() for k, v in r.items()} for p, r in results.items()}, indent=1), encoding="utf-8")
