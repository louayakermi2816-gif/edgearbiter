"""What did a trained device policy learn? Action frequencies on the test episodes, split by situation.

    python scripts/inspect_policy.py --scenario energy_constrained --algo ppo --seed 0
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

from edgearbiter.agents.mappo import load_actor
from edgearbiter.baselines import I
from edgearbiter.costs import CostTable
from edgearbiter.evaluate import TEST_SEED0
from edgearbiter.experiments import ALGOS, SCENARIOS, run_paths
from edgearbiter.sim.env import RATES, EdgeArbiterEnv, edge_rule


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="energy_constrained")
    ap.add_argument("--algo", choices=list(ALGOS), default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--name", default="experiments", help="study name given to run_experiments.py")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--cost-table", default=None)
    args = ap.parse_args()
    _, paths = run_paths(Path(args.out), Path(args.models), args.scenario, args.algo, args.seed, args.name)
    actors = {a: load_actor(p) for a, p in paths.items()}
    env = EdgeArbiterEnv(CostTable.load(args.cost_table), SCENARIOS[args.scenario])
    tiers = [t["name"] for t in env.costs.tiers]
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=TEST_SEED0 + ep)
        done = False
        while not done:
            a = actors["device"].act(obs["device"])
            ea = actors["edge"].act(obs["edge"]) if "edge" in actors else edge_rule(obs, env)
            o = obs["device"]
            rows.append((*env.decode_device(a), o[I["link_good"]] > 0.5 and o[I["edge_accepting"]] > 0.5))
            obs, _, term, trunc, _ = env.step({"device": a, "edge": ea})
            done = term or trunc
    print(f"{args.algo} seed {args.seed}, scenario {args.scenario}, {args.episodes} test episodes, {len(rows)} decisions\n")
    print("| situation | steps | most frequent device actions |\n|---|---|---|")
    for name, good in (("link good and edge accepting", True), ("otherwise", False)):
        sub = [r[:3] for r in rows if r[3] == good]
        counts = collections.Counter(sub).most_common(4)
        acts = ", ".join(f"{tiers[t]} {RATES[r]} fps {'edge' if p else 'local'} {n / len(sub):.0%}" for (t, r, p), n in counts)
        print(f"| {name} | {len(sub)} | {acts} |")
    local = np.array([r[0] for r in rows if r[2] == 0])
    print(f"\nlocal decisions by tier: " + ", ".join(f"{tiers[t]} {np.mean(local == t):.0%}" for t in range(len(tiers))))


if __name__ == "__main__":
    main()
