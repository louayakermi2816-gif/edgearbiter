"""Multi-device study: three devices share one edge (`sim/fleet.py`).

Questions, on the measured cost table with the energy-constrained per-device budget:
    - does training the devices together beat deploying the single-device policy on each device?
    - does a learned edge (per-device admission) beat the threshold rule?   (edge rule vs learned edge)
    - does a centralised critic help?                                        (IPPO vs MAPPO)
Same protocol as `experiments.py`: validation episodes for learning curves, model selection and the
best static configuration; held-out test episodes for the tables, paired against the best static policy.
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .baselines import BASELINES
from .costs import CostTable
from .evaluate import KPIS, TEST_SEED0, VAL_SEED0, save_results, summarize
from .experiments import DEFAULT_MAPPO, run_info, run_paths, scenario_configs, static_policy
from .sim.env import RATES, SimConfig
from .sim.fleet import PROFILES, FleetEnv, fleet_edge_rule, link_aware_edge_rule, on_every_device

FLEET_KPIS = KPIS + ["alive"]
ALGOS = {   # name -> (the edge learns, centralised critic); the devices always learn
    "ippo_devices": (False, False),
    "mappo_devices": (False, True),
    "ippo": (True, False),
    "mappo": (True, True),
}
LABELS = {"ippo_devices": "IPPO, devices (edge rule)", "mappo_devices": "MAPPO, devices (edge rule)",
          "ippo": "IPPO, devices + edge", "mappo": "MAPPO, devices + edge"}
SCENARIO = "energy_constrained"


def evaluate_fleet(policies: Dict[str, Callable], costs: CostTable, cfg: SimConfig, n_episodes: int = 20,
                   seed0: int = TEST_SEED0, profiles: Sequence[str] = PROFILES) -> Dict[str, np.ndarray]:
    env = FleetEnv(costs, cfg, profiles)
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step({a: p(obs, env) for a, p in policies.items()})
            done = term or trunc
        rows.append(info["kpis"])
    return {k: np.array([float(r[k]) for r in rows]) for k in FLEET_KPIS}


def _concat(results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([r[k] for r in results]) for k in results[0]}


def fleet_paths(out_dir: Path, models_dir: Path, algo: str, seed: int, name: str = "fleet", agents: Sequence[str] = ()):
    return (Path(out_dir) / "runs" / name / f"{algo}_seed{seed}.json",
            {a: Path(models_dir) / name / f"{algo}_seed{seed}_{a}.pt" for a in agents})


def train_fleet_run(job: dict) -> dict:
    """Train one (algo, seed) job; keeps the actors of the best validation evaluation."""
    import torch
    from .agents.mappo import MAPPOConfig, policies_from_actors, save_actor, train_mappo
    torch.set_num_threads(1)
    algo, seed = job["algo"], job["seed"]
    costs, cfg = CostTable.load(job.get("cost_table")), SimConfig(**job["sim"])
    env = FleetEnv(costs, cfg, job.get("profiles", PROFILES))
    learn_edge, central = ALGOS[algo]
    controlled = env.devices + (["edge"] if learn_edge else [])
    mcfg = MAPPOConfig(total_steps=job["steps"], seed=seed, centralized_critic=central,
                       callback_every=job.get("eval_every", 50_000), **job.get("mappo", {}))
    curve, best, t0 = [], {"return": -np.inf, "steps": 0, "state": None}, time.time()

    def on_eval(steps, actors):
        pol = policies_from_actors(actors)
        if not learn_edge:
            pol["edge"] = fleet_edge_rule
        r = evaluate_fleet(pol, costs, cfg, job.get("val_episodes", 10), VAL_SEED0, env.profiles)
        curve.append({"steps": steps, "seconds": round(time.time() - t0, 1), **{k: float(v.mean()) for k, v in r.items()}})
        if curve[-1]["return"] > best["return"]:
            best.update(steps=steps, state={a: copy.deepcopy(m.state_dict()) for a, m in actors.items()},
                        **{"return": curve[-1]["return"]})

    actors, _, _ = train_mappo(env, mcfg, controlled=controlled,
                               fixed_policies=None if learn_edge else {"edge": fleet_edge_rule}, log=None,
                               callback=on_eval, init_policies={"edge": fleet_edge_rule} if learn_edge else None)
    if best["state"] is not None:
        for a, m in actors.items():
            m.load_state_dict(best["state"][a])
    curve_path, actor_paths = fleet_paths(job["out_dir"], job["models_dir"], algo, seed, job["name"], controlled)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    for a, path in actor_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_actor(actors[a], path, agent=a, algo=algo, scenario=f"fleet/{SCENARIO}", seed=seed)
    curve_path.write_text(json.dumps({"job": {k: v for k, v in job.items() if k not in ("out_dir", "models_dir")},
                                      "cost_table": {"source": costs.source, "fingerprint": costs.fingerprint},
                                      "agents": controlled, "config": asdict(mcfg), "curve": curve,
                                      "selected_steps": best["steps"], "selected_val_return": best["return"],
                                      "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"[fleet] {algo}/seed{seed}: best val return {best['return']:.1f} at {best['steps']} steps, "
          f"final {curve[-1]['return']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    return {"algo": algo, "seed": seed}


def _run_jobs(jobs: Sequence[dict], workers: int) -> None:
    if workers <= 1:
        for j in jobs:
            train_fleet_run(j)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(train_fleet_run, jobs, chunksize=1)


def _learned_policies(out_dir: Path, models_dir: Path, algo: str, seed: int, name: str, costs: CostTable,
                      cfg: SimConfig) -> Dict[str, Callable]:
    from .agents.mappo import load_actor, policies_from_actors
    path, _ = fleet_paths(out_dir, models_dir, algo, seed, name)
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["cost_table"]["fingerprint"] != costs.fingerprint or SimConfig(**info["job"]["sim"]) != cfg:
        raise ValueError(f"{path} was trained on another cost table or scenario")
    _, paths = fleet_paths(out_dir, models_dir, algo, seed, name, info["agents"])
    pol = policies_from_actors({a: load_actor(p) for a, p in paths.items()})
    return pol if "edge" in pol else {**pol, "edge": fleet_edge_rule}


def _single_device_ppo(out_dir: Path, models_dir: Path, seed: int, costs: CostTable, cfg: SimConfig) -> Callable:
    """The device actor of the single-device study (PPO, same table and budget), for every device of the fleet."""
    from .agents.mappo import load_actor
    info = run_info(out_dir, SCENARIO, "ppo", seed, "experiments") or {}
    if info.get("cost_table", {}).get("fingerprint") != costs.fingerprint or SimConfig(**info["job"]["sim"]) != cfg:
        raise ValueError("the single-device PPO actors were trained on another cost table or budget: "
                         "run scripts/run_experiments.py --budget-fraction 0.7 first")
    actor = load_actor(run_paths(out_dir, models_dir, SCENARIO, "ppo", seed, "experiments")[1]["device"])
    return lambda obs, env: actor.act(obs["device"])


def best_static_fleet(costs: CostTable, cfg: SimConfig, n_episodes: int = 10):
    """Best configuration applied by every device (edge rule), searched on validation episodes."""
    env = FleetEnv(costs, cfg)
    scores = {}
    for a in range(env.action_spaces["device0"].n):
        pol = {**on_every_device(static_policy(a), env), "edge": fleet_edge_rule}
        scores[a] = float(evaluate_fleet(pol, costs, cfg, n_episodes, VAL_SEED0)["return"].mean())
    a = max(scores, key=scores.get)
    tier, rate_idx, place = env.decode_device(a)
    name = f"best static {costs.tiers[tier]['name']} {RATES[rate_idx]}fps {'edge' if place else 'local'}"
    return name, static_policy(a), scores


def run_fleet_study(algos: Sequence[str], seeds: Sequence[int], steps: int, out_dir: Path = Path("results"),
                    models_dir: Path = Path("models"), workers: int = 1, train: bool = True, n_episodes: int = 20,
                    eval_every: int = 50_000, cost_table: Optional[str] = None, budget_fraction: float = 0.7,
                    name: str = "fleet", mappo: Optional[dict] = None, transfer: bool = True, val_episodes: int = 10,
                    log: Callable = print) -> str:
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    costs = CostTable.load(cost_table)
    cfg = scenario_configs(costs, budget_fraction)[SCENARIO]
    if train:
        jobs = [{"name": name, "algo": a, "seed": k, "steps": steps, "eval_every": eval_every,
                 "mappo": dict(DEFAULT_MAPPO if mappo is None else mappo), "cost_table": cost_table, "sim": asdict(cfg),
                 "profiles": list(PROFILES), "val_episodes": val_episodes, "out_dir": str(out_dir),
                 "models_dir": str(models_dir)}
                for a in algos for k in seeds]
        log(f"[fleet] {len(jobs)} training runs on {workers} worker(s)")
        t0 = time.time(); _run_jobs(jobs, workers); log(f"[fleet] training took {time.time() - t0:.0f}s")

    env = FleetEnv(costs, cfg)
    static_name, static_pol, static_scores = best_static_fleet(costs, cfg, val_episodes)
    test = lambda pol: evaluate_fleet(pol, costs, cfg, n_episodes, TEST_SEED0)
    val = lambda pol: float(evaluate_fleet(pol, costs, cfg, val_episodes, VAL_SEED0)["return"].mean())
    every = lambda p: on_every_device(p, env)
    candidates = {
        "rule_based (edge rule)": {**every(BASELINES["rule_based"]), "edge": fleet_edge_rule},
        "deadline_greedy (edge rule)": {**every(BASELINES["deadline_greedy"]), "edge": fleet_edge_rule},
        "deadline_greedy (link-aware edge)": {**every(BASELINES["deadline_greedy"]), "edge": link_aware_edge_rule},
        static_name: {**every(static_pol), "edge": fleet_edge_rule},
    }
    results = {n: test(p) for n, p in candidates.items()}
    refs = {"best static": max(static_scores.values())}
    heuristics = {n: val(candidates[n]) for n in candidates if n != static_name}
    top = max(heuristics, key=heuristics.get)
    refs[top.replace("_", "-")] = heuristics[top]
    if transfer:
        for edge_name, edge in (("edge rule", fleet_edge_rule), ("link-aware edge", link_aware_edge_rule)):
            label = f"single-device PPO on each device ({edge_name})"
            results[label] = _concat([test({**every(_single_device_ppo(out_dir, models_dir, k, costs, cfg)), "edge": edge})
                                      for k in seeds])
    seed_rows, curves = [], {}
    for a in algos:
        per_seed = [test(_learned_policies(out_dir, models_dir, a, k, name, costs, cfg)) for k in seeds]
        results[LABELS[a]] = _concat(per_seed)
        infos = [json.loads(fleet_paths(out_dir, models_dir, a, k, name)[0].read_text(encoding="utf-8")) for k in seeds]
        seed_rows.append(f"| {LABELS[a]} | " + " | ".join(f"{float(r['return'].mean()):.1f} (checkpoint "
                                                           f"{i['selected_steps'] / 1e3:.0f}k)" for r, i in zip(per_seed, infos)) + " |")
        n_pts = min(len(i["curve"]) for i in infos)
        ret = np.array([[p["return"] for p in i["curve"][:n_pts]] for i in infos])
        curves[a] = {"steps": [p["steps"] for p in infos[0]["curve"][:n_pts]], "mean": ret.mean(0).tolist(),
                     "std": ret.std(0).tolist(), "n_seeds": len(infos)}
    header = (f"### Three devices ({', '.join(PROFILES)} links) sharing one edge, battery {cfg.battery_mj / 1000:g} J "
              f"per device per {cfg.horizon} s episode\n\nMean ± std over {n_episodes} test episodes (seeds {TEST_SEED0}+; "
              f"learned methods: pooled over {len(seeds)} training seeds). Δ return: paired difference vs `{static_name}`, "
              f"95 % CI.")
    save_results(results, out_dir, f"{name}_{SCENARIO}", ref=static_name, header=header)
    figure = ""
    try:
        from .plots import learning_curves
        for theme, suffix in (("light", ""), ("dark", "_dark")):
            learning_curves({"three devices, one edge": curves}, {"three devices, one edge": refs}, LABELS,
                            out_dir / f"{name}_learning_curves{suffix}.png", theme)
        figure = (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{name}_learning_curves_dark.png">'
                  f'<img alt="Validation return vs training steps, three devices sharing one edge" '
                  f'src="{name}_learning_curves.png"></picture>\n\n')
    except ImportError:
        log("[fleet] matplotlib not installed: figure skipped")
    alive = "\n".join(f"| {n} | {r['alive'].mean():.2f} |" for n, r in results.items())
    per_seed = ("| method | " + " | ".join(f"seed {k}" for k in seeds) + " |\n|" + "---|" * (len(seeds) + 1) + "\n"
                + "\n".join(seed_rows))
    report = (f"## {name}\n\nCost table: `{costs.source}` ({costs.fingerprint}) · {len(seeds)} seeds × {steps:,} steps "
              f"per run\n\n{figure}{header}\n\n{summarize(results, static_name)}\n\nDevices still running at the end of "
              f"the episode (of 3):\n\n| policy | alive |\n|---|---|\n{alive}\n\nTest return per training seed (step of "
              f"the selected checkpoint):\n\n{per_seed}\n")
    (out_dir / f"{name}.md").write_text(report, encoding="utf-8")
    return report
