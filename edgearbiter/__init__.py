"""EdgeArbiter: multi-agent RL for adaptive AI model selection on IoT-edge devices."""
from .costs import CostTable
from .sim.env import EdgeArbiterEnv, SimConfig, edge_rule

__all__ = ["CostTable", "EdgeArbiterEnv", "SimConfig", "edge_rule"]
__version__ = "0.2.0"
