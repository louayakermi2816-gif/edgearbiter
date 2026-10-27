"""Run the ESP32 node in Espressif's QEMU, its serial port on a TCP socket.

QEMU emulates the ESP32 (CPU, flash, UART, timers, GPIO) but not its radio, so this runs the
serial-transport build of the firmware; edgearbiter/serial_bridge.py connects it to the broker.

    firmware/esp32/build.sh serial                              # -> firmware/esp32/build/serial/
    python scripts/run_qemu_node.py --port 5555 &               # boots once the bridge connects
    python -m edgearbiter.serial_bridge --tcp localhost:5555 --broker localhost

    python scripts/run_qemu_node.py --check                     # smoke test, no broker needed

--check boots the firmware, waits for its telemetry, sends one decision (medium tier, 10 fps,
offloaded) and checks that the node applies it. QEMU: qemu-system-xtensa from
https://github.com/espressif/qemu/releases (qemu-xtensa-softmmu-*), on PATH or --qemu.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgearbiter.serial_bridge import TcpLink  # noqa: E402

DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "firmware/esp32/build/serial/edge_node.ino.merged.bin"


def qemu_command(qemu: str, image: Path, port: int) -> list:
    return [qemu, "-machine", "esp32", "-display", "none", "-monitor", "none",
            "-drive", f"file={image},if=mtd,format=raw",
            "-global", "driver=timer.esp32.timg,property=wdt_disable,value=true",   # emulation is slower than silicon
            "-serial", f"tcp:127.0.0.1:{port},server=on,wait=on"]                 # start when the client connects


def read_message(link: TcpLink, sub: str, deadline: float, log=None) -> dict:
    """The next "<sub> <json>" line from the node, before `deadline`."""
    while time.time() < deadline:
        line = link.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if log:
            log(line)
        kind, _, payload = line.partition(" ")
        if kind == sub and payload.startswith("{"):
            return json.loads(payload)
    raise TimeoutError(f"no {sub} line from the node")


def check(link: TcpLink, boot_timeout: float = 90.0, log=None) -> dict:
    """Telemetry arrives, and a decision sent over the serial port is applied within two intervals."""
    t0 = time.time()
    first = read_message(link, "telemetry", t0 + boot_timeout, log)
    booted = time.time() - t0
    decision = {"seq": first["seq"], "tier": 2, "rate_fps": 10, "placement": "edge"}
    link.write(f"decision {json.dumps(decision, separators=(',', ':'))}\n".encode())
    sent, deadline = time.time(), time.time() + 10.0
    while True:
        node = read_message(link, "node", deadline, log)
        if node["tier"] == 2 and node["placement"] == "edge" and node["rate_fps"] == 10:
            break
    return {"first_telemetry": first, "boot_s": round(booted, 1), "applied_after_s": round(time.time() - sent, 1),
            "node": node}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="merged flash image of the serial build")
    ap.add_argument("--qemu", default=shutil.which("qemu-system-xtensa") or "qemu-system-xtensa")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--check", action="store_true", help="smoke test: boot, telemetry, apply one decision")
    ap.add_argument("--verbose", action="store_true", help="print the node's serial output during --check")
    args = ap.parse_args()
    if not args.image.exists():
        sys.exit(f"{args.image} not found: run firmware/esp32/build.sh serial first")

    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "flash.bin"                  # QEMU writes to the flash (NVS): keep the build clean
        shutil.copy(args.image, image)
        cmd = qemu_command(args.qemu, image, args.port)
        if not args.check:
            print(f"QEMU ESP32 on tcp://127.0.0.1:{args.port} (boots when a client connects)", flush=True)
            proc = subprocess.Popen(cmd)
            signal.signal(signal.SIGTERM, lambda *_: proc.terminate())        # stop the emulator with us
            sys.exit(proc.wait())
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            link = TcpLink("127.0.0.1", args.port)
            link.sock.settimeout(30)                         # a silent node fails the check instead of hanging
            result = check(link, log=print if args.verbose else None)
            link.close()
        except (TimeoutError, OSError) as exc:
            proc.kill()
            print(proc.communicate()[0][-2000:])
            sys.exit(f"QEMU check failed: {exc}")
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=10)
    print(f"ESP32 firmware in QEMU: first telemetry {result['boot_s']} s after start "
          f"({json.dumps(result['first_telemetry'])}); decision applied {result['applied_after_s']} s after sending it: "
          f"tier {result['node']['tier']}, {result['node']['rate_fps']} fps, {result['node']['placement']}")


if __name__ == "__main__":
    main()
