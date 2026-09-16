"""IPPO / MAPPO trainers run end to end on a tiny budget."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edgearbiter.agents.mappo import MAPPOConfig, policies_from_actors, train_mappo
from edgearbiter.costs import CostTable
from edgearbiter.evaluate import evaluate
from edgearbiter.sim.env import EdgeArbiterEnv, SimConfig, edge_rule

SMALL = SimConfig(horizon=40)


def test_single_agent_ppo_smoke():
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    actors, critics, hist = train_mappo(env, MAPPOConfig(total_steps=160, rollout_len=80, minibatch=40, epochs=2, centralized_critic=False),
                                        controlled=("device",), fixed_policies={"edge": edge_rule}, log=None)
    pol = policies_from_actors(actors)
    res = evaluate(pol["device"], edge_rule, costs=CostTable.synthetic(), cfg=SMALL, n_episodes=2)
    assert np.all(np.isfinite(res["return"])) and len(hist) >= 1


@pytest.mark.parametrize("central", [False, True])
def test_two_agent_smoke(central):
    env = EdgeArbiterEnv(CostTable.synthetic(), SMALL)
    actors, critics, _ = train_mappo(env, MAPPOConfig(total_steps=160, rollout_len=80, minibatch=40, epochs=2, centralized_critic=central), log=None)
    assert set(actors) == {"device", "edge"}
    pol = policies_from_actors(actors)
    res = evaluate(pol["device"], pol["edge"], costs=CostTable.synthetic(), cfg=SMALL, n_episodes=2)
    assert np.all(np.isfinite(res["return"]))
