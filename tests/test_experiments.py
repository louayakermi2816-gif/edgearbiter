"""The experiment protocol runs end to end on a tiny budget and writes every artefact."""
import json

import pytest

torch = pytest.importorskip("torch")

from edgearbiter.costs import CostTable
from edgearbiter.experiments import SCENARIOS, best_static, load_curves, run_experiment
from edgearbiter.sim.env import SimConfig


def test_best_static_is_the_argmax_over_fixed_actions():
    name, pol, scores = best_static(CostTable.synthetic(), SimConfig(horizon=30), n_episodes=2)
    assert len(scores) == 18 and pol(None, None) == max(scores, key=scores.get)
    assert name.startswith("best static ")


def test_run_experiment_writes_tables_curves_and_actors(tmp_path):
    out, models = tmp_path / "results", tmp_path / "models"
    report = run_experiment("smoke", ["ppo"], ["default"], seeds=[0], steps=1024, out_dir=out, models_dir=models,
                            n_episodes=2, eval_every=512, mappo={"n_envs": 2, "rollout_len": 256, "minibatch": 128},
                            log=lambda *a: None)
    assert "PPO (device, edge rule)" in report and "Scenario `default`" in report
    for f in ("smoke.md", "smoke_default.md", "smoke_default.csv", "smoke_default.json"):
        assert (out / f).exists(), f
    table = json.loads((out / "smoke_default.json").read_text())
    assert len(table["PPO (device, edge rule)"]["return"]) == 2
    assert (models / "smoke" / "default" / "ppo_seed0_device.pt").exists()
    curve = load_curves(out, "default", "ppo", [0], "smoke")
    assert curve["n_seeds"] == 1 and len(curve["steps"]) == 2
    assert set(SCENARIOS) == {"default", "energy_constrained"}
