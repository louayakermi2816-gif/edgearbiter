# EdgeArbiter — multi-agent RL for adaptive AI model selection on IoT-edge devices

Real-time AI on IoT devices has to trade **accuracy against latency, energy and
network quality**: which model to run, how often to sense, and whether to run it
on the device or offload it to an edge server. EdgeArbiter studies that decision
in three layers:

1. **Measured cost profiles** — a PyTorch/torchvision benchmark of FP32 and INT8
   classifiers and detectors (MobileNetV3, ResNet-18, EfficientNet-B0, SSDLite)
   for top-1 accuracy, p50/p95 latency and memory, plus a supervised
   **fine-tuning pipeline** (MobileNetV3-Small on Imagenette).
2. **An IoT-edge simulator** driven by those profiles: bursty workload, Markov
   wireless link, bounded device queue, battery, edge background load.
3. **Cooperative agents** — a *device* agent (model tier × sensing rate ×
   local/offload) and an *edge* agent (accept × CPU share) trained with
   **independent PPO (IPPO)** and **MAPPO** (centralised critic), against static
   and rule-based baselines.

A **FastAPI** service turns telemetry from a real **ESP32** node (MQTT) into
decisions, and a **Node-RED** flow closes the loop and displays the live choice.

Status: **in development** (started September 2026). Result tables in `results/`
are produced by the scripts below and reported only once measured. Until the
benchmark has run, the simulator uses a clearly labelled synthetic cost table.

```
torchvision models ──benchmark.py──▶ results/cost_table.json ──▶ simulator (sim/env.py)
                                                                      │
                                    baselines  ◀──────  IPPO / MAPPO (agents/mappo.py)
                                                                      │
ESP32 ──MQTT──▶ Node-RED ──HTTP──▶ FastAPI /policy/decide ──MQTT──▶ ESP32 (LED shows tier)
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt && pip install -e .
pytest -q                                              # simulator, costs, benchmark helpers, agents, API

python scripts/run_benchmark.py --no-data --iters 20   # latency/size only (fast) -> results/cost_table.json
python scripts/run_benchmark.py                        # + Imagenette top-1 (downloads ~100 MB)
python scripts/finetune_imagenette.py --epochs 5       # supervised fine-tuning -> models/*.pt + metrics JSON
python scripts/run_baselines.py                        # static / rule-based policies
python scripts/train_single_agent.py --steps 60000     # device PPO, edge follows the rule
python scripts/train_marl.py --steps 120000            # IPPO vs MAPPO vs baselines
uvicorn edgearbiter.api:app --reload                   # http://127.0.0.1:8000/docs
```

Full stack (API + Mosquitto + Node-RED): `docker compose up`; import
`node-red/flows.json`; flash `firmware/esp32/edge_node/edge_node.ino`.

## The simulator

One step = one second. Frames arrive from a calm/busy workload and queue on the
device (bounded; extra frames are dropped). The device processes up to its
sensing rate, locally or on the edge; the edge may reject (the device then falls
back to the tiny model). Latency, energy and accuracy come from the cost table,
transfer time from the link model (RTT + size/bandwidth).

| Agent | Observation | Action |
|---|---|---|
| device | battery, queue, link state, RTT, bandwidth, arrivals, last violation rate, edge load, time | tier (3) × rate {1, 5, 10 fps} × placement {local, edge} |
| edge | edge load, offload request, offloaded frames, link state, device queue, time | accept {0, 1} × CPU share {0.5, 1.0} |

Shared reward: `Σ accuracy(processed) − 0.5·deadline_violations − 0.5·energy_J − 0.3·dropped − 0.2·rejected`,
with a penalty when the battery empties. Weights in `edgearbiter/sim/env.py`.

## Results

Populated by the scripts (mean ± std over seeded episodes):

- `results/benchmark_device.md` — per-model p50/p95 latency, size, top-1 (FP32 vs INT8)
- `models/mobilenetv3_small_imagenette.json` — fine-tuning curve and best top-1
- `results/baselines.md` · `results/single_agent_vs_baselines.md` · `results/marl_vs_baselines.md`

## Layout

```
edgearbiter/costs.py          cost table (measured or synthetic default)
edgearbiter/benchmark.py      latency / size / accuracy benchmark, INT8 variants, cost-table builder
edgearbiter/finetune.py       MobileNetV3-Small fine-tuning on Imagenette
edgearbiter/sim/link.py       link, workload and edge-load models
edgearbiter/sim/env.py        two-agent simulator + fixed edge rule
edgearbiter/agents/mappo.py   IPPO / MAPPO trainer (from scratch, PyTorch)
edgearbiter/baselines.py      static and rule-based device policies
edgearbiter/evaluate.py       seeded evaluation + tables
edgearbiter/api.py            FastAPI: /policy/decide, /infer, /sim/*
firmware/esp32/edge_node/     ESP32 telemetry + decision node (Arduino)
node-red/flows.json           telemetry -> decision loop, sim KPIs
scripts/                      run_benchmark, finetune_imagenette, run_baselines, train_single_agent, train_marl
tests/                        simulator, costs, benchmark helpers, agents, API
```

## Limitations (known, on purpose)

- Energy per inference is derived from latency (`ENERGY_MJ_PER_MS`) until measured with a power meter.
- The edge latency is estimated as device/8 until the benchmark is run on the edge host (`--role edge`).
- The detector is benchmarked for latency only (no COCO evaluation wired in yet).
- Single simulator instance, CPU training; no vectorised environments.

## License

MIT — see `LICENSE`.
