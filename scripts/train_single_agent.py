"""Train the device agent alone with PPO (edge follows the fixed rule) and compare with baselines.

    python scripts/train_single_agent.py --steps 60000
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
from edgearbiter.sim.env import EdgeArbiterEnv, edge_rule


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    costs = CostTable.load(args.cost_table)
    env = EdgeArbiterEnv(costs)
    actors, critics, history = train_mappo(env, MAPPOConfig(total_steps=args.steps, seed=args.seed, centralized_critic=False),
                                           controlled=("device",), fixed_policies={"edge": edge_rule})
    Path("models").mkdir(exist_ok=True); torch.save(actors["device"].state_dict(), "models/device_ppo.pt")
    out = Path(args.out); out.mkdir(exist_ok=True)
    (out / "single_agent_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    results = {n: evaluate(p, costs=costs, n_episodes=args.episodes) for n, p in BASELINES.items()}
    results["ppo device (edge rule)"] = evaluate(policies_from_actors(actors)["device"], costs=costs, n_episodes=args.episodes)
    print(summarize(results)); save_results(results, out, "single_agent_vs_baselines")


if __name__ == "__main__":
    main()
