"""IPPO / MAPPO trainers run end to end on a tiny budget, plus the trainer's building blocks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edgearbiter.agents.mappo import (Actor, MAPPOConfig, RunningNorm, gae, load_actor, policies_from_actors, save_actor,
                                      train_mappo)
from edgearbiter.costs import CostTable
from edgearbiter.evaluate import evaluate
from edgearbiter.sim.env import EdgeArbiterEnv, SimConfig, edge_rule

SMALL = SimConfig(horizon=40)
TINY = dict(total_steps=160, n_envs=2, rollout_len=80, minibatch=40, epochs=2)


def test_single_agent_ppo_smoke():
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    actors, critics, hist = train_mappo(env, MAPPOConfig(**TINY, centralized_critic=False),
                                        controlled=("device",), fixed_policies={"edge": edge_rule}, log=None)
    pol = policies_from_actors(actors)
    res = evaluate(pol["device"], edge_rule, costs=CostTable.synthetic(), cfg=SMALL, n_episodes=2)
    assert np.all(np.isfinite(res["return"])) and len(hist) >= 1


@pytest.mark.parametrize("central", [False, True])
def test_two_agent_smoke(central):
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    calls = []
    actors, critics, _ = train_mappo(env, MAPPOConfig(**TINY, centralized_critic=central, callback_every=80), log=None,
                                     callback=lambda steps, actors: calls.append(steps))
    assert set(actors) == {"device", "edge"} and calls
    assert (critics["device"] is critics["edge"]) == central             # MAPPO shares one centralised critic
    pol = policies_from_actors(actors)
    res = evaluate(pol["device"], pol["edge"], costs=CostTable.synthetic(), cfg=SMALL, n_episodes=2)
    assert np.all(np.isfinite(res["return"]))


def test_training_is_seeded():
    runs = [train_mappo(EdgeArbiterEnv(CostTable.synthetic(), SMALL), MAPPOConfig(**TINY, seed=7), log=None)[2] for _ in range(2)]
    assert [h["return"] for h in runs[0]] == [h["return"] for h in runs[1]]


def test_gae_vectorised_matches_per_env():
    rng = np.random.default_rng(0)
    r, v, d, last = rng.normal(size=(6, 3)), rng.normal(size=(6, 3)), (rng.random((6, 3)) < 0.3).astype(float), rng.normal(size=3)
    adv, ret = gae(r, v, d, last, 0.99, 0.95)
    for i in range(3):
        a_i, r_i = gae(r[:, i], v[:, i], d[:, i], last[i], 0.99, 0.95)
        assert np.allclose(adv[:, i], a_i, atol=1e-5) and np.allclose(ret[:, i], r_i, atol=1e-5)


def test_running_norm_tracks_mean_and_std():
    n = RunningNorm(); x = np.random.default_rng(1).normal(50.0, 10.0, size=(4, 5000))
    for chunk in x:
        n.update(chunk)
    assert n.mean == pytest.approx(50.0, abs=0.5) and n.std == pytest.approx(10.0, rel=0.05)
    assert np.allclose(n.denormalize(n.normalize(x)), x)


def test_save_and_load_actor(tmp_path):
    actor = Actor(12, 18, hidden=32)
    save_actor(actor, tmp_path / "a.pt", agent="device")
    loaded = load_actor(tmp_path / "a.pt")
    obs = np.random.default_rng(0).random(12).astype(np.float32)
    assert loaded.act(obs) == actor.act(obs) and loaded.net[0].out_features == 32


def test_behaviour_cloning_warm_start_imitates_the_edge_rule():
    from edgearbiter.agents.mappo import behaviour_clone
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    actor = Actor(env.observation_spaces["edge"].shape[0], env.action_spaces["edge"].n, hidden=64)
    agreement = behaviour_clone({"edge": actor}, {"edge": edge_rule}, env, steps=4000, epochs=30)
    assert agreement["edge"] > 0.9
    actors, _, _ = train_mappo(EdgeArbiterEnv(CostTable.synthetic(), SMALL), MAPPOConfig(**TINY, bc_steps=1000), log=None,
                               init_policies={"edge": edge_rule})
    assert set(actors) == {"device", "edge"}


def test_critic_warmup_keeps_the_actors_still():
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    torch.manual_seed(0)
    reference = {a: Actor(env.observation_spaces[a].shape[0], env.action_spaces[a].n).state_dict() for a in ("device", "edge")}
    actors, critics, _ = train_mappo(env, MAPPOConfig(**TINY, critic_warmup_steps=160), log=None)
    assert all(torch.equal(actors[a].state_dict()[k], reference[a][k]) for a in actors for k in reference[a])
    moved, _, _ = train_mappo(env, MAPPOConfig(**TINY), log=None)
    assert not torch.equal(moved["device"].state_dict()["net.0.weight"], reference["device"]["net.0.weight"])
