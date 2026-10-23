"""Train device + edge agents jointly: IPPO (independent critics) and MAPPO (centralised critic).

    python scripts/train_marl.py --steps 1500000 --seeds 0
"""
from __future__ import annotations

from run_experiments import parser, run

if __name__ == "__main__":
    run("marl_vs_baselines", parser(algos=("ippo", "mappo"), seeds=(0,)).parse_args())
