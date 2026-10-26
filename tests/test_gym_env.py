"""The single-agent Gymnasium view of the simulator is a valid gymnasium.Env."""
import pytest
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


def test_stable_baselines3_trains_on_it():
    sb3 = pytest.importorskip("stable_baselines3")                 # optional: not a project dependency
    env = DeviceEnv(CostTable.synthetic(), SimConfig(horizon=30))
    model = sb3.PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu").learn(128)
    obs, _ = env.reset(seed=1)
    assert env.action_space.contains(int(model.predict(obs, deterministic=True)[0]))
