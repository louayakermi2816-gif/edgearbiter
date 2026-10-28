"""scripts/demo.py: the policy service, the broker and the virtual node in one command."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("uvicorn")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("extra", [[], ["--rules", "--no-mqtt"]])
def test_demo_runs_the_loop_with_the_trained_agents_or_the_rules(extra):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--steps", "4", "--interval", "0", *extra],
                         capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    expected = "rule-based policies" if extra else "trained MAPPO agents (device, edge)"
    assert expected in out.stdout and "4 decisions from the policy service (0 timeouts)" in out.stdout
