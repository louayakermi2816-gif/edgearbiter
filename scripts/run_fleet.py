"""Multi-device study: three devices (near / mid / far from the access point) share one edge.

    python scripts/run_fleet.py --steps 1000000 --seeds 0 1 2 --workers 4
    python scripts/run_fleet.py --skip-training          # re-evaluate saved actors, redraw the figure

Needs the measured cost table and the single-device study (its PPO actors are the transfer baseline):
scripts/run_experiments.py --budget-fraction 0.7. Writes results/fleet.md, results/fleet_energy_constrained.{md,csv,json}
and results/fleet_learning_curves{,_dark}.png; actors go to models/fleet/, learning curves to results/runs/fleet/.
"""
from __future__ import annotations

import argparse
import os
import sys

from edgearbiter.fleet_study import ALGOS, run_fleet_study


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--episodes", type=int, default=20, help="test episodes per evaluation")
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--no-transfer", action="store_true", help="skip the single-device PPO baseline")
    ap.add_argument("--budget-fraction", type=float, default=0.7)
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--name", default="fleet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    print(run_fleet_study(args.algos, args.seeds, args.steps, out_dir=args.out, models_dir=args.models,
                          workers=args.workers, train=not args.skip_training, n_episodes=args.episodes,
                          eval_every=args.eval_every, cost_table=args.cost_table, budget_fraction=args.budget_fraction,
                          name=args.name, transfer=not args.no_transfer))


if __name__ == "__main__":
    main()
