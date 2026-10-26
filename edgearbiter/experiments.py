"""Experiment protocol: scenarios, training runs, evaluation and learning curves.

Protocol
    - every method is evaluated on the same test episodes (seeds TEST_SEED0 ...)
    - learning curves, the choice of the best static configuration and model selection use
      validation episodes (seeds VAL_SEED0 ...), never the test episodes
    - each training run keeps the actors of its best validation evaluation (early stopping)
    - IPPO / MAPPO start the edge actor from the edge rule (behaviour cloning) and then learn both
      agents jointly; `mappo_scratch` is the ablation without that warm start
    - learned methods are trained with several seeds; tables pool the test episodes of all
      seeds (paired with the baselines episode by episode) and per-seed means are reported
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import BASELINES
from .costs import CostTable
from .evaluate import TEST_SEED0, VAL_SEED0, concat, evaluate
from .sim.env import RATES, EdgeArbiterEnv, SimConfig, edge_rule

SCENARIOS: Dict[str, SimConfig] = {
    "default": SimConfig(),
    "energy_constrained": replace(SimConfig(), battery_mj=40_000.0),   # 40 J per 300 s episode (~133 mW)
}
# On the synthetic table, 40 J is ~70 % of the energy the best fixed configuration uses in `default` on the
# validation episodes (56.6 J): the budget binds. `scenario_configs(costs, budget_fraction=0.7)` sets the same
# relative budget on any cost table.
ALGOS = {   # name -> (controlled agents, centralised critic, edge actor warm-started from the edge rule)
    "ppo": (("device",), False, False),             # single agent: the edge follows the fixed rule
    "ippo": (("device", "edge"), False, True),
    "mappo": (("device", "edge"), True, True),
    "mappo_scratch": (("device", "edge"), True, False),   # ablation: both agents from random initialisation
}
LABELS = {"ppo": "PPO (device, edge rule)", "ippo": "IPPO (device+edge)", "mappo": "MAPPO (device+edge)",
          "mappo_scratch": "MAPPO, edge from scratch"}
# trainer settings chosen in pilot runs on validation episodes (see README): a longer discount for the
# episode-long energy budget, a larger entropy bonus for the device, a near-deterministic warm start
# and a small entropy bonus for the edge (whose learning signal is weak: its action only matters when
# the device offloads), one softmax over the joint action
DEFAULT_MAPPO = {"gamma": 0.995, "ent_coef": 0.02, "ent_coefs": {"edge": 0.001}, "bc_smoothing": 0.05}


def static_policy(action: int) -> Callable:
    return lambda obs, env: action


def static_name(env: EdgeArbiterEnv, action: int) -> str:
    tier, rate_idx, place = env.decode_device(action)
    return f"static {env.costs.tiers[tier]['name']} {RATES[rate_idx]}fps {'edge' if place else 'local'}"


def best_static(costs: CostTable, cfg: SimConfig, n_episodes: int = 10) -> Tuple[str, Callable, Dict[int, float]]:
    """Exhaustive search over the fixed device actions (edge rule) on validation episodes."""
    env = EdgeArbiterEnv(costs, cfg)
    scores = {a: float(evaluate(static_policy(a), costs=costs, cfg=cfg, n_episodes=n_episodes, seed0=VAL_SEED0)["return"].mean())
              for a in range(env.action_spaces["device"].n)}
    a = max(scores, key=scores.get)
    return f"best {static_name(env, a)}", static_policy(a), scores


def scenario_configs(costs: CostTable, budget_fraction: Optional[float] = None) -> Dict[str, SimConfig]:
    """The scenarios on a cost table. With `budget_fraction`, the energy_constrained budget is that fraction of
    the energy the best fixed configuration uses in `default` (validation episodes), instead of 40 J."""
    if budget_fraction is None:
        return dict(SCENARIOS)
    _, pol, _ = best_static(costs, SCENARIOS["default"])
    energy_j = float(evaluate(pol, costs=costs, cfg=SCENARIOS["default"], n_episodes=10, seed0=VAL_SEED0)["energy_j"].mean())
    budget = replace(SCENARIOS["energy_constrained"], battery_mj=round(budget_fraction * energy_j * 1000.0, -2))
    return {**SCENARIOS, "energy_constrained": budget}


def evaluate_baselines(costs: CostTable, cfg: SimConfig, n_episodes: int = 20, best=None) -> Dict[str, Dict[str, np.ndarray]]:
    name, pol, _ = best or best_static(costs, cfg)
    pols = {**BASELINES, name: pol}
    return {n: evaluate(p, edge_rule, costs=costs, cfg=cfg, n_episodes=n_episodes, seed0=TEST_SEED0) for n, p in pols.items()}


# ------------------------------------------------------------------ training runs
def run_paths(out_dir: Path, models_dir: Path, scenario: str, algo: str, seed: int, name: str = "experiments"):
    """Learning-curve record and actor checkpoints of one run, namespaced by study `name`."""
    return (Path(out_dir) / "runs" / name / scenario / f"{algo}_seed{seed}.json",
            {a: Path(models_dir) / name / scenario / f"{algo}_seed{seed}_{a}.pt" for a in ALGOS[algo][0]})


def train_run(job: dict) -> dict:
    """Train one (scenario, algo, seed) job; saves the actors and the validation learning curve."""
    import torch
    from .agents.mappo import MAPPOConfig, policies_from_actors, save_actor, train_mappo
    torch.set_num_threads(1)
    scenario, algo, seed = job["scenario"], job["algo"], job["seed"]
    costs = CostTable.load(job.get("cost_table"))
    cfg = SimConfig(**job["sim"]) if "sim" in job else SCENARIOS[scenario]
    controlled, central, warm_edge = ALGOS[algo]
    mcfg = MAPPOConfig(total_steps=job["steps"], seed=seed, centralized_critic=central,
                       callback_every=job.get("eval_every", 50_000), **job.get("mappo", {}))
    curve: List[dict] = []
    best = {"return": -np.inf, "steps": 0, "state": None}
    t0 = time.time()

    def on_eval(steps, actors):     # validation episodes only; keeps the best actors (early stopping)
        pol = policies_from_actors(actors)
        r = evaluate(pol["device"], pol.get("edge", edge_rule), costs=costs, cfg=cfg, n_episodes=job.get("val_episodes", 10),
                     seed0=VAL_SEED0)
        curve.append({"steps": steps, "seconds": round(time.time() - t0, 1),
                      **{k: float(v.mean()) for k, v in r.items()}})
        if curve[-1]["return"] > best["return"]:
            best.update(steps=steps, state={a: copy.deepcopy(m.state_dict()) for a, m in actors.items()},
                        **{"return": curve[-1]["return"]})

    actors, _, _ = train_mappo(EdgeArbiterEnv(costs, cfg), mcfg, controlled=controlled,
                               fixed_policies={"edge": edge_rule} if "edge" not in controlled else None,
                               log=None, callback=on_eval, init_policies={"edge": edge_rule} if warm_edge else None)
    if best["state"] is not None:
        for a, m in actors.items():
            m.load_state_dict(best["state"][a])
    curve_path, actor_paths = run_paths(job["out_dir"], job["models_dir"], scenario, algo, seed, job.get("name", "experiments"))
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    for a, path in actor_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_actor(actors[a], path, agent=a, algo=algo, scenario=scenario, seed=seed)
    curve_path.write_text(json.dumps({"job": {k: v for k, v in job.items() if k not in ("out_dir", "models_dir")},
                                      "cost_table": {"source": costs.source, "fingerprint": costs.fingerprint},
                                      "config": asdict(mcfg), "curve": curve, "selected_steps": best["steps"],
                                      "selected_val_return": best["return"], "seconds": round(time.time() - t0, 1)},
                                     indent=1), encoding="utf-8")
    print(f"[train] {scenario}/{algo}/seed{seed}: best val return {best['return']:.1f} at {best['steps']} steps, "
          f"final {curve[-1]['return']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    return {"scenario": scenario, "algo": algo, "seed": seed, "curve_path": str(curve_path)}


def run_jobs(jobs: Sequence[dict], workers: int = 1) -> List[dict]:
    if workers <= 1:
        return [train_run(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(train_run, jobs, chunksize=1)


def evaluate_learned(scenario: str, algo: str, seeds: Sequence[int], out_dir: Path, models_dir: Path,
                     costs: CostTable, n_episodes: int = 20, name: str = "experiments",
                     cfg: Optional[SimConfig] = None) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """Test-episode KPIs of one learned method, concatenated over its training seeds."""
    from .agents.mappo import load_actor
    cfg = cfg or SCENARIOS[scenario]
    per_seed = []
    for seed in seeds:
        info = run_info(out_dir, scenario, algo, seed, name) or {}
        trained_on = info.get("cost_table", {})
        if trained_on and trained_on["fingerprint"] != costs.fingerprint:
            raise ValueError(f"{name}/{scenario}/{algo}/seed{seed} was trained on cost table {trained_on['source']} "
                             f"({trained_on['fingerprint']}), not on {costs.source} ({costs.fingerprint})")
        if "sim" in info.get("job", {}) and SimConfig(**info["job"]["sim"]) != cfg:
            raise ValueError(f"{name}/{scenario}/{algo}/seed{seed} was trained with {info['job']['sim']}, not {asdict(cfg)}")
        _, paths = run_paths(out_dir, models_dir, scenario, algo, seed, name)
        actors = {a: load_actor(p) for a, p in paths.items()}
        dev = (lambda obs, env, m=actors["device"]: m.act(obs["device"]))
        edge = (lambda obs, env, m=actors["edge"]: m.act(obs["edge"])) if "edge" in actors else edge_rule
        per_seed.append(evaluate(dev, edge, costs=costs, cfg=cfg, n_episodes=n_episodes, seed0=TEST_SEED0))
    return concat(per_seed), [float(r["return"].mean()) for r in per_seed]


def validation_references(costs: CostTable, cfg: SimConfig, n_episodes: int = 10, best=None) -> Dict[str, float]:
    """Validation return of the best static policy and of the best heuristic (reference lines on the curves)."""
    _, _, scores = best or best_static(costs, cfg, n_episodes)
    heuristics = {n.replace("_", "-"): float(evaluate(BASELINES[n], costs=costs, cfg=cfg, n_episodes=n_episodes,
                                                      seed0=VAL_SEED0)["return"].mean())
                  for n in ("rule_based", "deadline_greedy")}
    top = max(heuristics, key=heuristics.get)
    return {"best static": max(scores.values()), top: heuristics[top]}


def run_experiment(name: str, algos: Sequence[str], scenarios: Sequence[str], seeds: Sequence[int], steps: int,
                   out_dir: Path = Path("results"), models_dir: Path = Path("models"), workers: int = 1,
                   train: bool = True, n_episodes: int = 20, eval_every: int = 50_000, mappo: Optional[dict] = None,
                   cost_table: Optional[str] = None, budget_fraction: Optional[float] = None, log: Callable = print) -> str:
    """Train (optionally), evaluate against the baselines and write results/<name>*.md|csv|json + figures."""
    from .evaluate import save_results, summarize
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    mappo = dict(DEFAULT_MAPPO if mappo is None else mappo)
    costs = CostTable.load(cost_table)
    configs = scenario_configs(costs, budget_fraction)
    if train:
        jobs = [{"name": name, "scenario": s, "algo": a, "seed": k, "steps": steps, "eval_every": eval_every, "mappo": mappo,
                 "cost_table": cost_table, "sim": asdict(configs[s]), "out_dir": str(out_dir), "models_dir": str(models_dir)}
                for s in scenarios for a in algos for k in seeds]
        log(f"[experiment] {len(jobs)} training runs on {workers} worker(s)")
        t0 = time.time(); run_jobs(jobs, workers); log(f"[experiment] training took {time.time() - t0:.0f}s")
    sections, curves, refs, runs = [], {}, {}, []
    for s in scenarios:
        cfg = configs[s]
        best = best_static(costs, cfg)
        results = evaluate_baselines(costs, cfg, n_episodes, best)
        ref = next(n for n in results if n.startswith("best static"))
        seed_rows = []
        for a in algos:
            res, per_seed = evaluate_learned(s, a, seeds, out_dir, models_dir, costs, n_episodes, name, cfg)
            results[LABELS[a]] = res
            info = [run_info(out_dir, s, a, k, name) or {} for k in seeds]
            runs += [i for i in info if i]
            seed_rows.append(f"| {LABELS[a]} | " + " | ".join(
                f"{m:.1f} (checkpoint {i.get('selected_steps', 0) / 1e3:.0f}k)" for m, i in zip(per_seed, info)) + " |")
        rule = (f", {100 * budget_fraction:.0f} % of the energy of the best fixed configuration in `default`"
                if budget_fraction is not None and cfg != SCENARIOS[s] else "")
        header = (f"### Scenario `{s}` (battery {cfg.battery_mj / 1000:g} J per {cfg.horizon} s episode{rule})\n\n"
                  f"Mean ± std over {n_episodes} test episodes (seeds {TEST_SEED0}+; learned methods: pooled over "
                  f"{len(seeds)} training seeds, actors from each run's best validation evaluation). Δ return: paired "
                  f"difference vs `{ref}` with 95 % CI.")
        save_results(results, out_dir, f"{name}_{s}", ref=ref, header=header)
        per_seed_table = ("| method | " + " | ".join(f"seed {k}" for k in seeds) + " |\n|" + "---|" * (len(seeds) + 1)
                          + "\n" + "\n".join(seed_rows))
        sections.append(f"{header}\n\n{summarize(results, ref)}\n\nTest return per training seed (step of the selected "
                        f"checkpoint):\n\n{per_seed_table}")
        curves[s] = {a: load_curves(out_dir, s, a, seeds, name) for a in algos}
        refs[s] = validation_references(costs, cfg, best=best)
    figure = ""
    try:
        from .plots import learning_curves
        for theme, suffix in (("light", ""), ("dark", "_dark")):
            learning_curves(curves, refs, LABELS, out_dir / f"{name}_learning_curves{suffix}.png", theme)
        figure = (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{name}_learning_curves_dark.png">'
                  f'<img alt="Validation return vs training steps" src="{name}_learning_curves.png"></picture>\n\n'
                  "Lines: mean validation return over training seeds (band: ± std); horizontal lines: baselines "
                  "on the same validation episodes.\n\n")
    except ImportError:
        log("[experiment] matplotlib not installed: figures skipped")
    budgets = sorted({r["job"]["steps"] for r in runs}) or [steps]           # what the saved runs were trained with
    overrides = runs[0]["job"].get("mappo", mappo) if runs else mappo
    report = (f"## {name.replace('_', ' ')}\n\nCost table: `{costs.source}` ({costs.fingerprint}) · {len(seeds)} seeds × "
              f"{' / '.join(f'{b:,}' for b in budgets)} steps per run · trainer overrides: `{overrides}`\n\n{figure}"
              + "\n\n".join(sections) + "\n")
    (out_dir / f"{name}.md").write_text(report, encoding="utf-8")
    return report


def run_info(out_dir: Path, scenario: str, algo: str, seed: int, name: str = "experiments") -> Optional[dict]:
    """The saved record of one training run (job, config, learning curve, selected checkpoint)."""
    path, _ = run_paths(out_dir, Path("."), scenario, algo, seed, name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_curves(out_dir: Path, scenario: str, algo: str, seeds: Sequence[int], name: str = "experiments") -> Optional[dict]:
    """Validation learning curve (mean / std over seeds) of one method."""
    curves = []
    for seed in seeds:
        path, _ = run_paths(out_dir, Path("."), scenario, algo, seed, name)
        if path.exists():
            curves.append(json.loads(path.read_text(encoding="utf-8"))["curve"])
    if not curves:
        return None
    n = min(len(c) for c in curves)
    ret = np.array([[p["return"] for p in c[:n]] for c in curves])
    return {"steps": [p["steps"] for p in curves[0][:n]], "mean": ret.mean(0).tolist(), "std": ret.std(0).tolist(),
            "n_seeds": len(curves)}
