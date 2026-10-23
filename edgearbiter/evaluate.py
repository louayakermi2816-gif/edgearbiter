"""Seeded evaluation of (device, edge) policy pairs and Markdown result tables.

Every policy is evaluated on the same seeded episodes, so differences in return
between two policies can be compared episode by episode (paired), which removes
most of the workload variance. Learned policies trained with several seeds are
stored as their per-seed evaluations concatenated in seed order.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .costs import CostTable
from .sim.env import EdgeArbiterEnv, SimConfig, edge_rule

KPIS = ["return", "mean_accuracy", "violation_rate", "energy_j", "processed", "dropped", "offloaded", "rejected",
        "edge_cost", "battery"]
TABLE_COLS = ["return", "mean_accuracy", "violation_rate", "energy_j", "dropped", "offloaded", "rejected", "edge_cost"]
TEST_SEED0 = 1000          # reported results
VAL_SEED0 = 50_000         # model selection / learning curves (disjoint from the test seeds)


def run_episode(env: EdgeArbiterEnv, device_policy: Callable, edge_policy: Callable, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step({"device": device_policy(obs, env), "edge": edge_policy(obs, env)})
        done = term or trunc
    return {k: float(info["kpis"][k]) for k in KPIS}


def evaluate(device_policy: Callable, edge_policy: Callable = edge_rule, costs: Optional[CostTable] = None,
             cfg: SimConfig = SimConfig(), n_episodes: int = 20, seed0: int = TEST_SEED0) -> Dict[str, np.ndarray]:
    env = EdgeArbiterEnv(costs, cfg)
    rows = [run_episode(env, device_policy, edge_policy, seed0 + i) for i in range(n_episodes)]
    return {k: np.array([r[k] for r in rows]) for k in KPIS}


def concat(results: list) -> Dict[str, np.ndarray]:
    """Concatenate per-seed evaluations of one method (same episodes for every seed)."""
    return {k: np.concatenate([r[k] for r in results]) for k in KPIS}


def paired_delta(res: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], key: str = "return"):
    """Mean and 95 % CI half-width of res - ref, episode by episode (ref tiled over seeds)."""
    a, b = res[key], ref[key]
    d = a - np.tile(b, len(a) // len(b))
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def _fmt(col: str, x: np.ndarray) -> str:
    return f"{x.mean():.3f} ± {x.std():.3f}" if col in ("mean_accuracy", "violation_rate") else f"{x.mean():.1f} ± {x.std():.1f}"


def summarize(results: Dict[str, Dict[str, np.ndarray]], ref: Optional[str] = None) -> str:
    cols = TABLE_COLS
    head = ["policy"] + cols + ([f"Δ return vs {ref}"] if ref else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, res in results.items():
        cells = [_fmt(c, res[c]) for c in cols]
        if ref:
            m, h = paired_delta(res, results[ref])
            cells.append("—" if name == ref else f"{m:+.1f} ± {h:.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(results: Dict[str, Dict[str, np.ndarray]], out_dir: Path, name: str, ref: Optional[str] = None,
                 header: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text((header + "\n\n" if header else "") + summarize(results, ref) + "\n", encoding="utf-8")
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "episode"] + KPIS)
        for p, res in results.items():
            for i in range(len(res["return"])):
                w.writerow([p, i] + [float(res[k][i]) for k in KPIS])
    (out_dir / f"{name}.json").write_text(json.dumps({p: {k: v.tolist() for k, v in r.items()} for p, r in results.items()}, indent=1), encoding="utf-8")
