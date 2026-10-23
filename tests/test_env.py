import numpy as np
import pytest

from edgearbiter.baselines import BASELINES, EDGE_BASELINES, deadline_greedy
from edgearbiter.costs import SYNTHETIC_TIERS, CostTable
from edgearbiter.evaluate import evaluate, run_episode
from edgearbiter.sim.env import DEVICE_OBS, EDGE_OBS, RATES, SHARES, EdgeArbiterEnv, SimConfig, edge_rule, miss_probability
from edgearbiter.sim.link import Link, classify_link, congestion

I = {n: i for i, n in enumerate(DEVICE_OBS)}


def test_spaces_and_reset():
    env = EdgeArbiterEnv(CostTable.synthetic())
    obs, info = env.reset(seed=0)
    assert obs["device"].shape == (len(DEVICE_OBS),) and obs["edge"].shape == (len(EDGE_OBS),)
    assert env.observation_spaces["device"].contains(obs["device"])
    assert env.observation_spaces["edge"].contains(obs["edge"])
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


def test_deadline_misses_follow_the_measured_latency_tail():
    tiers = [dict(t) for t in SYNTHETIC_TIERS]
    tiers[1] = dict(tiers[1], latency_ms={"device": 80.0, "edge": 7.0}, latency_p95_ms={"device": 120.0, "edge": None})
    costs = CostTable(tiers=tiers)
    sigma = costs.latency_sigma(1, "device")
    assert sigma > 0 and costs.latency_sigma(1, "edge") == 0.0 and costs.latency_sigma(0, "device") == 0.0
    assert miss_probability(80.0, 120.0, sigma) == pytest.approx(0.05)            # the p95 itself
    assert 0.05 < miss_probability(80.0, 100.0, sigma) < 0.5
    assert miss_probability(100.0, 100.0, sigma, fixed_ms=60.0) == pytest.approx(0.5)   # 40 ms median left for compute
    assert miss_probability(100.0, 50.0, sigma, fixed_ms=60.0) == 1.0              # the network alone is too slow
    assert miss_probability(80.0, 100.0) == 0.0 and miss_probability(120.0, 100.0) == 1.0   # no tail: deterministic
    env = EdgeArbiterEnv(costs)
    env.reset(seed=0); env.queue = 20
    _, _, _, _, info = env.step({"device": env.encode_device(1, 2, 0), "edge": env.encode_edge(1, 1)})
    last = info["last"]
    assert last["processed"] == 10 and last["violations"] == pytest.approx(10 * miss_probability(80.0, 100.0, sigma))


def test_offload_uses_edge_and_rejection_falls_back():
    env = EdgeArbiterEnv(CostTable.synthetic())
    env.reset(seed=1)
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(1, 1)})
    assert info["last"]["where"] == "edge" and info["last"]["rejected"] == 0
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(0, 1)})
    assert info["last"]["where"] == "device" and info["last"]["tier"] == 0 and info["last"]["rejected"] == 1


def test_device_sees_previous_edge_decision_not_its_load():
    env = EdgeArbiterEnv(CostTable.synthetic())
    obs, _ = env.reset(seed=4)
    assert obs["device"][I["edge_accepting"]] == 1.0                      # the edge starts out accepting
    obs, *_ = env.step({"device": env.encode_device(0, 1, 0), "edge": env.encode_edge(0, 1)})
    assert obs["device"][I["edge_accepting"]] == 0.0 and obs["device"][I["edge_share"]] == 0.0
    obs, *_ = env.step({"device": env.encode_device(0, 1, 0), "edge": env.encode_edge(1, 0)})
    assert obs["device"][I["edge_accepting"]] == 1.0 and obs["device"][I["edge_share"]] == SHARES[0]
    assert obs["edge"][0] == pytest.approx(env.edge_load.value)          # only the edge observes its load


def test_uplink_limits_offload_throughput():
    link = Link(state=2, rtt_ms=150.0, bandwidth_mbps=0.8)
    assert link.uplink_fps(40.0) == 2                                    # 800 kbit/s / 320 kbit per frame
    env = EdgeArbiterEnv(CostTable.synthetic())
    env.reset(seed=0)
    env.queue = 20
    env.link.step = lambda rng: None                                     # freeze the link in a poor state
    env.link.state, env.link.rtt_ms, env.link.bandwidth_mbps = 2, 150.0, 0.8
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(1, 1)})
    assert info["last"]["where"] == "edge" and info["last"]["processed"] == 2


def test_edge_cost_charged_only_for_used_share():
    assert congestion(0.0) == 0.0 and congestion(0.5) == pytest.approx(1.0) and congestion(0.9) > congestion(0.5)
    env = EdgeArbiterEnv(CostTable.synthetic())
    env.reset(seed=2); env.queue = 10
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 0), "edge": env.encode_edge(1, 1)})
    assert info["last"]["edge_cost"] == 0.0                              # local run: the reserved share is unused
    env.queue = 10
    _, _, _, _, info = env.step({"device": env.encode_device(2, 2, 1), "edge": env.encode_edge(1, 1)})
    load = info["last"]["edge_load"]
    assert info["last"]["edge_cost"] == pytest.approx(SHARES[1] * load / (1 - load))


def test_classify_link_matches_simulator_ranges():
    assert classify_link(12, 30) == 0 and classify_link(40, 10) == 1 and classify_link(150, 1) == 2


def test_battery_drains_and_terminates():
    env = EdgeArbiterEnv(CostTable.synthetic(), SimConfig(horizon=10_000, battery_mj=2_000.0))
    obs, _ = env.reset(seed=2)
    term = False; steps = 0
    while not term and steps < 5000:
        obs, r, term, trunc, info = env.step({"device": env.encode_device(2, 2, 0), "edge": 0}); steps += 1
    assert term and info["kpis"]["battery"] == 0.0


def test_deadline_greedy_respects_deadline_and_admission():
    env = EdgeArbiterEnv(CostTable.synthetic())
    obs, _ = env.reset(seed=0)
    o = obs["device"].copy()
    o[I["rtt"]], o[I["bandwidth"]] = 10 / 200, 40 / 50                   # good link, edge accepting at full share
    tier, _, place = env.decode_device(deadline_greedy({"device": o, "edge": obs["edge"]}, env))
    assert (tier, place) == (2, 1)                                       # medium is only feasible on the edge
    o[I["edge_accepting"]], o[I["edge_share"]] = 0.0, 0.0
    tier, _, place = env.decode_device(deadline_greedy({"device": o, "edge": obs["edge"]}, env))
    assert (tier, place) == (1, 0)                                       # best local tier within 100 ms


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_evaluate(name):
    res = evaluate(BASELINES[name], costs=CostTable.synthetic(), cfg=SimConfig(horizon=60), n_episodes=2)
    assert np.all(np.isfinite(res["return"]))


@pytest.mark.parametrize("name", list(EDGE_BASELINES))
def test_edge_baselines_evaluate(name):
    res = evaluate(BASELINES["rule_based"], EDGE_BASELINES[name], costs=CostTable.synthetic(), cfg=SimConfig(horizon=60), n_episodes=2)
    assert np.all(np.isfinite(res["return"]))
