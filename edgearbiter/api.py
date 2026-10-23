"""FastAPI service: policy decisions for a real device and on-demand inference.

    GET  /health
    GET  /cost_table                     the cost profiles driving the simulator
    POST /policy/decide                  telemetry from a device -> (tier, rate, placement) + edge admission
    POST /infer?tier=0                   image upload -> top-5 classes with the chosen tier's model
    POST /sim/reset?seed=0, /sim/step    drive the simulator remotely (dashboard demos)

The decision endpoint accepts the fields the ESP32 node publishes over MQTT
(battery, queue, rtt_ms, bandwidth_mbps, arrivals) plus the edge server's load,
and runs both agents exactly as in the simulator: the edge decides admission and
CPU share from its own observation, the device decides from its own observation,
which carries the admission decision the edge published at the previous call.
MODEL_DEVICE / MODEL_EDGE may point to trained actors (scripts/train_marl.py);
otherwise the rule-based device and the edge rule answer.
Run: uvicorn edgearbiter.api:app --reload
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .baselines import rule_based
from .costs import CostTable
from .mqtt_bridge import Publisher
from .sim.env import RATES, SHARES, EdgeArbiterEnv, device_observation, edge_observation, edge_rule
from .sim.link import classify_link

_costs = CostTable.load(os.environ.get("COST_TABLE"))
_env = EdgeArbiterEnv(_costs)
_obs: Optional[Dict[str, np.ndarray]] = None
_publisher = Publisher()
_actors: Dict[str, object] = {}
_models: Dict[int, object] = {}
_edge_state = {"accepting": 1, "share": SHARES[-1], "offload_request": 0, "requested_tier": 0, "offload_frames": 0}


def _load_actors():
    from .agents.mappo import load_actor
    for agent, var in (("device", "MODEL_DEVICE"), ("edge", "MODEL_EDGE")):
        path = os.getenv(var)
        if not (path and Path(path).exists()):
            continue
        actor = load_actor(path)
        expected = _env.observation_spaces[agent].shape[0]
        if actor.net[0].in_features != expected:
            print(f"[api] {var}={path} expects {actor.net[0].in_features} observations, simulator has {expected}: ignored")
            continue
        _actors[agent] = actor


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        _load_actors()
    except Exception as exc:
        print(f"[api] actors not loaded: {exc}")
    yield
    _publisher.close()


app = FastAPI(title="EdgeArbiter", version="0.2.0", description="Multi-agent RL model selection for IoT-edge devices", lifespan=_lifespan)


class Telemetry(BaseModel):
    battery: float = 1.0
    queue: int = 0
    rtt_ms: float = 15.0
    bandwidth_mbps: float = 30.0
    arrivals: int = 2
    edge_load: float = 0.3          # reported by the edge host, used by the edge policy only
    violation_rate: float = 0.0
    t_fraction: float = 0.0


def _observations(t: Telemetry) -> Dict[str, np.ndarray]:
    link, queue_frac, s = classify_link(t.rtt_ms, t.bandwidth_mbps), t.queue / _env.cfg.max_queue, _edge_state
    device = device_observation(t.battery, queue_frac, link, t.rtt_ms, t.bandwidth_mbps, t.arrivals, t.violation_rate,
                                s["accepting"], s["share"], t.t_fraction)
    edge = edge_observation(t.edge_load, float(s["offload_request"]), s["offload_frames"], link, queue_frac,
                            s["requested_tier"] / max(1, _env.n_tiers - 1), t.t_fraction)
    return {"device": device, "edge": edge}


@app.get("/health")
def health():
    return {"status": "ok", "cost_table_source": _costs.source, "actors": sorted(_actors), "mqtt": _publisher.enabled}


@app.get("/cost_table")
def cost_table():
    return {"tiers": _costs.tiers, "frame_kb": _costs.frame_kb, "source": _costs.source}


@app.post("/policy/decide")
def decide(t: Telemetry):
    obs = _observations(t)
    a = _actors["device"].act(obs["device"]) if "device" in _actors else rule_based(obs, _env)
    ea = _actors["edge"].act(obs["edge"]) if "edge" in _actors else edge_rule(obs, _env)
    tier, rate_idx, place = _env.decode_device(a)
    accept, share_idx = _env.decode_edge(ea)
    _edge_state.update(accepting=accept, share=SHARES[share_idx], offload_request=place,
                       requested_tier=tier if place else 0,
                       offload_frames=min(t.queue + t.arrivals, RATES[rate_idx]) if place and accept else 0)
    out = {"action": int(a), "tier": tier, "model": _costs.tiers[tier]["model"], "rate_fps": RATES[rate_idx],
           "placement": "edge" if place else "local", "edge_accept": bool(accept), "edge_share": SHARES[share_idx],
           "policy": {agent: ("actor" if agent in _actors else "rule") for agent in ("device", "edge")}}
    if place and not accept:                     # same fallback as the simulator: tiny model locally
        out.update(tier=0, model=_costs.tiers[0]["model"], placement="local", rejected=True)
    _publisher.publish("api/decision", out)
    return out


@app.post("/infer")
async def infer(file: UploadFile = File(...), tier: int = 0, topk: int = 5):
    if not 0 <= tier < _costs.n_tiers:
        raise HTTPException(400, f"tier must be in [0, {_costs.n_tiers - 1}]")
    try:
        import torch
        from PIL import Image
        from .benchmark import build, model_zoo, set_quant_engine
        from .data import eval_transform, class_names
    except Exception as exc:
        raise HTTPException(503, f"inference dependencies missing: {exc}")
    spec = model_zoo()[_costs.tiers[tier]["model"]]
    if tier not in _models:
        try:
            _models[tier] = build(spec, pretrained=True)
        except Exception as exc:
            raise HTTPException(503, f"model weights unavailable: {exc}")
    if spec.backend:
        set_quant_engine(spec.backend)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(_models[tier](eval_transform(spec.img)(img).unsqueeze(0))[0], 0)
    top = torch.topk(probs, min(topk, len(probs)))
    names = class_names(spec.imagenet_head)
    return {"tier": tier, "model": _costs.tiers[tier]["model"],
            "top": [{"class_index": int(i), "label": names[int(i)], "prob": round(float(p), 4)} for p, i in zip(top.values, top.indices)]}


@app.post("/sim/reset")
def sim_reset(seed: int = 0):
    global _obs
    _obs, info = _env.reset(seed=seed)
    return {"seed": seed, "kpis": info["kpis"]}


@app.post("/sim/step")
def sim_step(device_action: Optional[int] = None, edge_action: Optional[int] = None):
    global _obs
    if _obs is None:
        raise HTTPException(409, "call /sim/reset first")
    da = device_action if device_action is not None else (_actors["device"].act(_obs["device"]) if "device" in _actors else rule_based(_obs, _env))
    ea = edge_action if edge_action is not None else (_actors["edge"].act(_obs["edge"]) if "edge" in _actors else edge_rule(_obs, _env))
    _obs, r, term, trunc, info = _env.step({"device": int(da), "edge": int(ea)})
    payload = {"reward": r, "terminated": term, "truncated": trunc, "kpis": info["kpis"], "last": info["last"]}
    _publisher.publish("sim", payload)
    return payload
