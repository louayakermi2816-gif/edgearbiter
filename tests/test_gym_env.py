"""The single-agent Gymnasium view of the simulator is a valid gymnasium.Env."""
from gymnasium.utils.env_checker import check_env

from edgearbiter.costs import CostTable
from edgearbiter.sim.env import SimConfig
from edgearbiter.sim.gym_env import DeviceEnv


def test_passes_gymnasium_env_checker():
    check_env(DeviceEnv(CostTable.synthetic()), skip_render_check=True)


def test_seeded_and_runs_to_the_horizon():
    env = DeviceEnv(CostTable.synthetic(), SimConfig(horizon=25))
    runs = []
    for _ in range(2):
        env.reset(seed=11); rewards, done = [], False
        while not done:
            _, r, term, trunc, info = env.step(env.sim.encode_device(1, 2, 0))
            rewards.append(r); done = term or trunc
        runs.append(rewards)
    assert runs[0] == runs[1] and len(runs[0]) == 25 and info["kpis"]["t"] == 25
