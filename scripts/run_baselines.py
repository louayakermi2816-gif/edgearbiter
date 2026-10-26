"""Evaluate the static / rule-based device policies (edge follows the rule) on seeded test episodes.

    python scripts/run_baselines.py --episodes 20
    python scripts/run_baselines.py --scenario energy_constrained
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgearbiter.costs import CostTable
from edgearbiter.evaluate import save_results, summarize
from edgearbiter.experiments import SCENARIOS, evaluate_baselines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="default")
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    costs = CostTable.load(args.cost_table)
    print(f"cost table: {costs.source} · scenario: {args.scenario}")
    results = evaluate_baselines(costs, SCENARIOS[args.scenario], args.episodes)
    ref = next(n for n in results if n.startswith("best static"))
    print(summarize(results, ref))
    name = "baselines" if args.scenario == "default" else f"baselines_{args.scenario}"
    save_results(results, Path(args.out), name, ref=ref,
                 header=f"Scenario `{args.scenario}` · cost table `{costs.source}` · {args.episodes} test episodes · "
                        f"Δ return: paired difference vs `{ref}` with 95 % CI")


if __name__ == "__main__":
    main()
