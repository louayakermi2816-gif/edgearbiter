"""Full experiment: baselines + PPO / IPPO / MAPPO over several seeds, on every scenario.

    python scripts/run_experiments.py --steps 1500000 --seeds 0 1 2 --workers 4
    python scripts/run_experiments.py --skip-training        # re-evaluate saved actors, redraw the figures
    python scripts/run_experiments.py --cost-table synthetic --name experiments_synthetic --steps 1000000   # dev. table

Writes results/<name>.md, results/<name>_<scenario>.{md,csv,json} and
results/<name>_learning_curves{,_dark}.png (name: "experiments" unless --name). Actors go to
models/<name>/<scenario>/ and the validation learning curves to results/runs/<name>/<scenario>/
(both git-ignored).
"""
from __future__ import annotations

import argparse
import os
import sys

from edgearbiter.experiments import ALGOS, SCENARIOS, run_experiment


def parser(algos=tuple(ALGOS), steps=1_500_000, seeds=(0, 1, 2)) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=list(algos), choices=list(ALGOS))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--steps", type=int, default=steps)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(seeds))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--episodes", type=int, default=20, help="test episodes per evaluation")
    ap.add_argument("--eval-every", type=int, default=50_000, help="env steps between validation evaluations")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--name", default=None, help="study name: results/<name>*.md, runs and models namespaced by it")
    ap.add_argument("--cost-table", default=None, help="cost table JSON, or 'synthetic' (default: results/cost_table.json)")
    ap.add_argument("--budget-fraction", type=float, default=None,
                    help="energy_constrained budget = this fraction of the energy the best fixed configuration uses "
                         "in default, instead of 40 J (0.7 gives the synthetic table's 40 J)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    return ap


def run(name: str, args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = run_experiment(args.name or name, args.algos, args.scenarios, args.seeds, args.steps, out_dir=args.out,
                            models_dir=args.models, workers=args.workers, train=not args.skip_training,
                            n_episodes=args.episodes, eval_every=args.eval_every, cost_table=args.cost_table,
                            budget_fraction=args.budget_fraction)
    print(report)


if __name__ == "__main__":
    run("experiments", parser().parse_args())
