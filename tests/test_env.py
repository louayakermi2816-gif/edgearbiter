import numpy as np
import pytest

from edgearbiter.baselines import BASELINES
from edgearbiter.costs import SYNTHETIC_TIERS, CostTable
from edgearbiter.evaluate import evaluate, run_episode
from edgearbiter.sim.env import RATES, SHARES, EdgeArbiterEnv, SimConfig, edge_rule


def test_spaces_and_reset():
    env = EdgeArbiterEnv(CostTable.synthetic())
    obs, info = env.reset(seed=0)
    assert obs["device"].shape == (11,) and obs["edge"].shape == (8,)
    assert env.observation_spaces["device"].contains(obs["device"])
    assert env.action_spaces["device"].n == len(SYNTHETIC_TIERS) * len(RATES) * 2
    assert env.action_spaces["edge"].n == 2 * len(SHARES)
    assert info["kpis"]["processed"] == 0


def test_encode_decode_roundtrip():
    env = EdgeArbiterEnv(CostTable.synthetic())
    for tier in range(env.n_tiers):
        for r in range(len(RATES)):
            for place in (0, 1):
                assert env.decode_device(env.encode_device(tier, r, place)) == (tier, r, place)
    for acc in (0, 1):
        for s in range(len(SHARES)):
            assert env.decode_edge(env.encode_edge(acc, s)) == (acc, s)


def test_episode_runs_and_is_deterministic():
    env = EdgeArbiterEnv(CostTable.synthetic(), SimConfig(horizon=120))
    a = run_episode(env, BASELINES["rule_based"], edge_rule, seed=3)
    b = run_episode(env, BASELINES["rule_based"], edge_rule, seed=3)
    assert a == b
    assert a["processed"] > 0 and 0.0 <= a["violation_rate"] <= 1.0


def test_offload_uses_edge_and_rejection_falls_back():
    env = EdgeArbiterEnv(CostTable.synthetic())
    env.reset(seed=1)
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(1, 1)})
    assert info["last"]["where"] == "edge" and info["last"]["rejected"] == 0
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(0, 1)})
    assert info["last"]["where"] == "device" and info["last"]["tier"] == 0 and info["last"]["rejected"] == 1


def test_battery_drains_and_terminates():
    env = EdgeArbiterEnv(CostTable.synthetic(), SimConfig(horizon=10_000, battery_mj=2_000.0))
    obs, _ = env.reset(seed=2)
    term = False; steps = 0
    while not term and steps < 5000:
        obs, r, term, trunc, info = env.step({"device": env.encode_device(2, 2, 0), "edge": 0}); steps += 1
    assert term and info["kpis"]["battery"] == 0.0


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_evaluate(name):
    res = evaluate(BASELINES[name], costs=CostTable.synthetic(), cfg=SimConfig(horizon=60), n_episodes=2)
    assert np.all(np.isfinite(res["return"]))
