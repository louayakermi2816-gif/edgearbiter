"""Train the device agent alone with PPO (edge follows the fixed rule) and compare with baselines.

    python scripts/train_single_agent.py --steps 1500000 --seeds 0
"""
from __future__ import annotations

from run_experiments import parser, run

if __name__ == "__main__":
    run("single_agent_vs_baselines", parser(algos=("ppo",), seeds=(0,)).parse_args())
