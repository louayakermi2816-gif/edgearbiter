"""Train device + edge agents jointly: IPPO (independent critics) and MAPPO (centralised critic).

    python scripts/train_marl.py --steps 120000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from edgearbiter.agents.mappo import MAPPOConfig, policies_from_actors, train_mappo
from edgearbiter.baselines import BASELINES
from edgearbiter.costs import CostTable
from edgearbiter.evaluate import evaluate, save_results, summarize
from edgearbiter.sim.env import EdgeArbiterEnv


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    costs = CostTable.load(args.cost_table)
    Path("models").mkdir(exist_ok=True); out = Path(args.out); out.mkdir(exist_ok=True)
    results = {n: evaluate(p, costs=costs, n_episodes=args.episodes) for n, p in BASELINES.items()}
    for name, central in (("ippo", False), ("mappo", True)):
        env = EdgeArbiterEnv(costs)
        actors, _, history = train_mappo(env, MAPPOConfig(total_steps=args.steps, seed=args.seed, centralized_critic=central))
        for a, m in actors.items():
            torch.save(m.state_dict(), f"models/{name}_{a}.pt")
        (out / f"{name}_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        pol = policies_from_actors(actors)
        results[f"{name} (device+edge)"] = evaluate(pol["device"], pol["edge"], costs=costs, n_episodes=args.episodes)
    print(summarize(results)); save_results(results, out, "marl_vs_baselines")


if __name__ == "__main__":
    main()
