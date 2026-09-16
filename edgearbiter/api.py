"""FastAPI service: policy decisions for a real device and on-demand inference.

    GET  /health
    GET  /cost_table                     the cost profiles driving the simulator
    POST /policy/decide                  telemetry from a device -> (tier, rate, placement)
    POST /infer?tier=0                   image upload -> top-5 ImageNet classes with the chosen tier's model
    POST /sim/reset?seed=0, /sim/step    drive the simulator remotely (dashboard demos)

The decision endpoint accepts the same fields the ESP32 node publishes over MQTT
(battery, queue, rtt_ms, bandwidth_mbps, arrivals, edge_load) and builds the
device observation. If MODEL_DEVICE points to a trained actor it is used,
otherwise the rule-based baseline answers.
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
from .sim.env import RATES, EdgeArbiterEnv, edge_rule
from .sim.link import LINK_STATES

_costs = CostTable.load()
_env = EdgeArbiterEnv(_costs)
_obs: Optional[Dict[str, np.ndarray]] = None
_publisher = Publisher()
_actor = None
_models: Dict[int, object] = {}


def _load_actor():
    global _actor
    path = os.getenv("MODEL_DEVICE")
    if path and Path(path).exists():
        import torch
        from .agents.mappo import Actor
        _actor = Actor(_env.observation_spaces["device"].shape[0], _env.action_spaces["device"].n)
        _actor.load_state_dict(torch.load(path, map_location="cpu")); _actor.eval()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        _load_actor()
    except Exception as exc:
        print(f"[api] actor not loaded: {exc}")
    yield
    _publisher.close()


app = FastAPI(title="EdgeArbiter", version="0.1.0", description="Multi-agent RL model selection for IoT-edge devices", lifespan=_lifespan)


class Telemetry(BaseModel):
    battery: float = 1.0
    queue: int = 0
    rtt_ms: float = 15.0
    bandwidth_mbps: float = 30.0
    arrivals: int = 2
    edge_load: float = 0.3
    violation_rate: float = 0.0
    t_fraction: float = 0.0


def _link_state(rtt_ms: float, bw: float) -> int:
    if rtt_ms <= 20 and bw >= 20: return 0
    if rtt_ms <= 60 and bw >= 5: return 1
    return 2


def _device_obs(t: Telemetry) -> np.ndarray:
    onehot = [1.0 if i == _link_state(t.rtt_ms, t.bandwidth_mbps) else 0.0 for i in range(len(LINK_STATES))]
    return np.asarray([t.battery, min(1.0, t.queue / _env.cfg.max_queue), *onehot, min(1.0, t.rtt_ms / 200.0),
                       min(1.0, t.bandwidth_mbps / 50.0), min(1.0, t.arrivals / 12.0), t.violation_rate, t.edge_load, t.t_fraction], dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "cost_table_source": _costs.source, "actor": _actor is not None, "mqtt": _publisher.enabled}


@app.get("/cost_table")
def cost_table():
    return {"tiers": _costs.tiers, "frame_kb": _costs.frame_kb, "source": _costs.source}


@app.post("/policy/decide")
def decide(t: Telemetry):
    obs = {"device": _device_obs(t), "edge": np.zeros(8, dtype=np.float32)}
    a = _actor.act(obs["device"]) if _actor is not None else rule_based(obs, _env)
    tier, rate_idx, place = _env.decode_device(a)
    out = {"action": int(a), "tier": tier, "model": _costs.tiers[tier]["model"], "rate_fps": RATES[rate_idx],
           "placement": "edge" if place else "local", "policy": "actor" if _actor is not None else "rule_based"}
    _publisher.publish("decision", out)
    return out


@app.post("/infer")
async def infer(file: UploadFile = File(...), tier: int = 0, topk: int = 5):
    if not 0 <= tier < _costs.n_tiers:
        raise HTTPException(400, f"tier must be in [0, {_costs.n_tiers - 1}]")
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from .benchmark import model_zoo
    except Exception as exc:
        raise HTTPException(503, f"inference dependencies missing: {exc}")
    if tier not in _models:
        _models[tier] = model_zoo()[_costs.tiers[tier]["model"]]().eval()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        probs = torch.softmax(_models[tier](tf(img).unsqueeze(0))[0], 0)
    top = torch.topk(probs, topk)
    return {"tier": tier, "model": _costs.tiers[tier]["model"], "top": [{"class_index": int(i), "prob": round(float(p), 4)} for p, i in zip(top.values, top.indices)]}


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
    da = device_action if device_action is not None else (_actor.act(_obs["device"]) if _actor is not None else rule_based(_obs, _env))
    ea = edge_action if edge_action is not None else edge_rule(_obs, _env)
    _obs, r, term, trunc, info = _env.step({"device": int(da), "edge": int(ea)})
    payload = {"reward": r, "terminated": term, "truncated": trunc, "kpis": info["kpis"], "last": info["last"]}
    _publisher.publish("sim", payload)
    return payload
