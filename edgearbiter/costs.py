"""Cost profiles of the candidate AI models.

The simulator is driven by a *cost table*: for each model tier, the measured
accuracy, latency on the device and on the edge, energy per inference and input
size. The real table is produced by `scripts/run_benchmark.py` (see
`edgearbiter/benchmark.py`) and saved to `results/cost_table.json`. Until it
exists, a clearly labelled synthetic default is used so the simulator and the
agents can be developed and tested offline; `CostTable.load("synthetic")` selects
it explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TABLE_PATH = Path("results/cost_table.json")

# Synthetic development profile. Order = tier index used by the agents.
SYNTHETIC_TIERS: List[Dict] = [
    {"name": "tiny",   "model": "mobilenet_v3_small",       "accuracy": 0.68, "latency_ms": {"device": 25.0, "edge": 4.0},  "energy_mj": 15.0, "size_mb": 10.0},
    {"name": "small",  "model": "mobilenet_v3_large_int8",  "accuracy": 0.74, "latency_ms": {"device": 55.0, "edge": 7.0},  "energy_mj": 32.0, "size_mb": 6.0},
    {"name": "medium", "model": "resnet18",                 "accuracy": 0.81, "latency_ms": {"device": 140.0, "edge": 16.0}, "energy_mj": 90.0, "size_mb": 45.0},
]
FRAME_KB = 40.0          # JPEG frame size sent when offloading
TX_ENERGY_MJ_PER_KB = 0.25
Z95 = 1.6448536269514722  # standard normal 95th percentile


@dataclass
class CostTable:
    tiers: List[Dict]
    frame_kb: float = FRAME_KB
    tx_energy_mj_per_kb: float = TX_ENERGY_MJ_PER_KB
    source: str = "synthetic"

    @property
    def n_tiers(self) -> int:
        return len(self.tiers)

    @property
    def fingerprint(self) -> str:
        """Short hash of the numbers that drive the simulator (identifies a table whatever its path)."""
        keys = ("accuracy", "latency_ms", "latency_p95_ms", "energy_mj")
        data = [{k: t.get(k) for k in keys} for t in self.tiers] + [self.frame_kb, self.tx_energy_mj_per_kb]
        return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def accuracy(self, tier: int) -> float:
        return float(self.tiers[tier]["accuracy"])

    def latency(self, tier: int, where: str) -> float:
        """Median (p50) latency of one inference."""
        return float(self.tiers[tier]["latency_ms"][where])

    def latency_sigma(self, tier: int, where: str) -> float:
        """Log-spread of the per-inference latency: the log-normal through the measured median and p95.
        0 when the table has no p95 for it (deterministic latency, as in the synthetic table)."""
        p95 = (self.tiers[tier].get("latency_p95_ms") or {}).get(where)
        p50 = self.latency(tier, where)
        if not p95 or p95 <= p50:
            return 0.0
        return math.log(float(p95) / p50) / Z95

    def energy(self, tier: int) -> float:
        return float(self.tiers[tier]["energy_mj"])

    def tx_energy(self) -> float:
        return self.frame_kb * self.tx_energy_mj_per_kb

    @staticmethod
    def synthetic() -> "CostTable":
        return CostTable(tiers=[dict(t) for t in SYNTHETIC_TIERS])

    @staticmethod
    def load(path: Optional[Path] = None) -> "CostTable":
        """Load the measured table if present, else the synthetic default ("synthetic": always the latter)."""
        if str(path) == "synthetic":
            return CostTable.synthetic()
        path = Path(path or DEFAULT_TABLE_PATH)
        if not path.exists():
            return CostTable.synthetic()
        data = json.loads(path.read_text(encoding="utf-8"))
        tiers = data.get("tiers")
        if not tiers:
            return CostTable.synthetic()
        for i, t in enumerate(tiers):                       # accuracy is null when the benchmark ran without data
            if t.get("accuracy") is None and i < len(SYNTHETIC_TIERS):
                t["accuracy"] = SYNTHETIC_TIERS[i]["accuracy"]
                t["accuracy_source"] = "synthetic"
        return CostTable(tiers=tiers, frame_kb=float(data.get("frame_kb", FRAME_KB)),
                         tx_energy_mj_per_kb=float(data.get("tx_energy_mj_per_kb", TX_ENERGY_MJ_PER_KB)),
                         source=str(path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tiers": self.tiers, "frame_kb": self.frame_kb,
                                    "tx_energy_mj_per_kb": self.tx_energy_mj_per_kb, "source": self.source}, indent=1), encoding="utf-8")
