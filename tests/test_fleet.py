"""Several devices sharing one edge: spaces, determinism, contention, per-device admission."""
import numpy as np

from edgearbiter.baselines import BASELINES
from edgearbiter.costs import CostTable
from edgearbiter.sim.env import SimConfig
from edgearbiter.sim.fleet import EDGE_FIELDS, FleetEnv, fleet_edge_rule, link_aware_edge_rule, on_every_device


def random_actions(env, rng):
    return {a: int(rng.integers(s.n)) for a, s in env.action_spaces.items()}


def test_spaces_and_reproducible_episodes():
    env = FleetEnv(CostTable.synthetic(), SimConfig(horizon=40))
    assert env.action_spaces["edge"].n == 16 and env.observation_spaces["edge"].shape == (2 + 7 * 3,)

    def run(seed):
        obs, _ = env.reset(seed=seed)
        rng, total, done = np.random.default_rng(0), 0.0, False
        while not done:
            assert all(env.observation_spaces[a].contains(o) for a, o in obs.items())
            obs, r, term, trunc, info = env.step(random_actions(env, rng))
            total, done = total + r, term or trunc
        return total, info["kpis"]
    assert run(3) == run(3) and run(3)[1]["t"] == 40


def test_links_and_workloads_do_not_depend_on_the_actions():
    env = FleetEnv(CostTable.synthetic(), SimConfig(horizon=30))
    traces = []
    for policy_seed in (1, 2):
        env.reset(seed=7); rng, trace = np.random.default_rng(policy_seed), []
        for _ in range(30):
            _, _, _, _, info = env.step(random_actions(env, rng))
            trace.append([(link.state, round(link.rtt_ms, 6)) for link in env.links] + [d["arrivals"] for d in info["last"]["devices"]])
        traces.append(trace)
    assert traces[0] == traces[1]


def test_devices_offloading_together_share_airtime_and_cpu():
    env = FleetEnv(CostTable.synthetic())
    out = {}
    for mask in (0b001, 0b111):                       # admit device0 alone, or all three
        env.reset(seed=5)
        env.queue = [10, 10, 10]
        _, _, _, _, info = env.step({**{d: env.encode_device(2, 2, 1) for d in env.devices}, "edge": env.encode_edge(mask, 1)})
        out[mask] = info["last"]
    assert out[0b111]["devices"][0]["latency_ms"] > out[0b001]["devices"][0]["latency_ms"]
    assert out[0b001]["devices"][1]["rejected"] == 1 and out[0b001]["devices"][1]["tier"] == 0     # falls back locally
    assert out[0b111]["edge"]["sharing"] == 3 and out[0b001]["edge"]["sharing"] == 1


def test_each_device_sees_its_own_admission():
    env = FleetEnv(CostTable.synthetic())
    env.reset(seed=1)
    obs, *_ = env.step({**{d: env.encode_device(0, 1, 0) for d in env.devices}, "edge": env.encode_edge(0b010, 1)})
    from edgearbiter.sim.env import DEVICE_OBS
    accepting = DEVICE_OBS.index("edge_accepting")
    assert [obs[d][accepting] for d in env.devices] == [0.0, 1.0, 0.0]


def test_edge_rules():
    env = FleetEnv(CostTable.synthetic())
    obs, _ = env.reset(seed=0)
    obs["edge"][0] = 0.3
    for i, state in enumerate((0, 2, 0)):             # device1 reports a poor link
        base = 1 + len(EDGE_FIELDS) * i + EDGE_FIELDS.index("link_good")
        obs["edge"][base:base + 3] = [1.0 if s == state else 0.0 for s in range(3)]
    assert env.decode_edge(fleet_edge_rule(obs, env)) == (0b111, 1)
    assert env.decode_edge(link_aware_edge_rule(obs, env)) == (0b101, 1)
    obs["edge"][0] = 0.8
    assert env.decode_edge(link_aware_edge_rule(obs, env))[0] == 0


def test_empty_device_drops_out_and_the_episode_goes_on():
    env = FleetEnv(CostTable.synthetic(), SimConfig(horizon=20))
    env.reset(seed=2)
    env.battery[1] = 1e-6
    policies = {**on_every_device(BASELINES["rule_based"], env), "edge": fleet_edge_rule}
    obs, done, steps = env._obs(), False, 0
    while not done:
        obs, r, term, trunc, info = env.step({a: p(obs, env) for a, p in policies.items()})
        done, steps = term or trunc, steps + 1
    assert steps == 20 and info["kpis"]["alive"] == 2 and info["last"]["devices"][1]["processed"] == 0


def test_fleet_study_runs_end_to_end(tmp_path):
    import json
    from edgearbiter.fleet_study import run_fleet_study
    out, models = tmp_path / "results", tmp_path / "models"
    report = run_fleet_study(["mappo", "ippo_devices"], [0], 1024, out_dir=out, models_dir=models, n_episodes=1,
                             eval_every=512, cost_table="synthetic", transfer=False, val_episodes=1,
                             mappo={"n_envs": 2, "rollout_len": 256, "minibatch": 128, "bc_steps": 500}, log=lambda *a: None)
    assert "MAPPO, devices + edge" in report and "deadline_greedy (link-aware edge)" in report
    table = json.loads((out / "fleet_energy_constrained.json").read_text())
    assert len(table["IPPO, devices (edge rule)"]["return"]) == 1 and "alive" in table["IPPO, devices (edge rule)"]
    assert (models / "fleet" / "mappo_seed0_edge.pt").exists() and not (models / "fleet" / "ippo_devices_seed0_edge.pt").exists()
