"""Full experiment: baselines + PPO / IPPO / MAPPO over several seeds, on every scenario.

    python scripts/run_experiments.py --steps 1500000 --seeds 0 1 2 --workers 4
    python scripts/run_experiments.py --skip-training        # re-evaluate saved actors, redraw the figures

Writes results/experiments.md, results/experiments_<scenario>.{md,csv,json} and
results/experiments_learning_curves{,_dark}.png. Actors go to models/<scenario>/ and the
validation learning curves to results/runs/<scenario>/ (both git-ignored).
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
    ap.add_argument("--cost-table", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    return ap


def run(name: str, args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = run_experiment(name, args.algos, args.scenarios, args.seeds, args.steps, out_dir=args.out,
                            models_dir=args.models, workers=args.workers, train=not args.skip_training,
                            n_episodes=args.episodes, eval_every=args.eval_every, cost_table=args.cost_table)
    print(report)


if __name__ == "__main__":
    run("experiments", parser().parse_args())
