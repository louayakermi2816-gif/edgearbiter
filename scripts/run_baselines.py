"""Evaluate the static / rule-based device policies (edge follows the rule) on seeded episodes.

    python scripts/run_baselines.py --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgearbiter.baselines import BASELINES
from edgearbiter.costs import CostTable
from edgearbiter.evaluate import evaluate, save_results, summarize


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    costs = CostTable.load(args.cost_table)
    print(f"cost table: {costs.source}")
    results = {name: evaluate(pol, costs=costs, n_episodes=args.episodes) for name, pol in BASELINES.items()}
    print(summarize(results))
    save_results(results, Path(args.out), "baselines")


if __name__ == "__main__":
    main()
