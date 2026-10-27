# ESP32 node — build it, run it in QEMU or Wokwi (no hardware needed)

The sketch publishes telemetry every second and applies the decisions that come back from the
policy service through Node-RED (topics `edgearbiter/telemetry`, `edgearbiter/node` and
`edgearbiter/decision`). It does not run the image models: their costs come from the
emulated-device benchmark; the node is the sensor side of the control loop.

| build option | what it selects |
|---|---|
| `TRANSPORT_SERIAL` 0 (default) | Wi-Fi + MQTT (PubSubClient) |
| `TRANSPORT_SERIAL` 1 | the same JSON messages as lines on the UART, `"<subtopic> <json>"`, relayed by `edgearbiter/serial_bridge.py` — for QEMU (no radio there) or a board on USB |
| `WOKWI` 1 (default) | Wi-Fi `Wokwi-GUEST`, the public broker `broker.hivemq.com`, frame arrivals from the simulator's calm/busy workload |
| `BATTERY_MODEL` 0 / 1 | battery read on GPIO34 (Wokwi potentiometer, board divider) / simulated from the measured energy per inference (default with the serial transport) |

## Build (verified)

arduino-cli, ESP32 Arduino core 3.3.12 (Espressif's toolchain), PubSubClient 2.8, ArduinoJson 7:

```bash
arduino-cli core install esp32:esp32@3.3.12 \
    --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install PubSubClient@2.8 ArduinoJson@7.4.3
firmware/esp32/build.sh mqtt      # Wi-Fi + MQTT: 919 KB of flash (70 %)
firmware/esp32/build.sh serial    # serial transport, DIO flash mode: 287 KB (21 %)
```

The CI workflow `firmware` builds both on every push.

## Run in Espressif's QEMU (verified)

QEMU (`qemu-system-xtensa` from [espressif/qemu releases](https://github.com/espressif/qemu/releases))
emulates the ESP32's CPU, flash, UART and timers, not its radio, so it runs the serial build:

```bash
python scripts/run_qemu_node.py --check          # boot, telemetry, apply one decision (also in CI)
python scripts/run_qemu_node.py --port 5555 &    # then connect it to the broker:
python -m edgearbiter.serial_bridge --tcp localhost:5555 --broker localhost
```

With the broker, the API and Node-RED running (see the main README), the board shows this node
(`esp32-1`): in the recorded run all 92 telemetry messages were answered within one interval.

## Run in Wokwi (not verified here)

**Web simulator** — create a new ESP32 project on wokwi.com, paste `edge_node.ino`, replace
`diagram.json`, add the libraries from `libraries.txt` (Library Manager), start the simulation.
Turn the potentiometer (the battery) down and watch the policy react.

**VS Code** — Wokwi extension + PlatformIO: `pio run` in this folder, then *Wokwi: Start Simulator*
(`wokwi.toml` points at the PlatformIO build).

**Close the loop** — run the policy service and Node-RED on your machine against the same broker:

```bash
uvicorn edgearbiter.api:app --port 8000
MQTT_HOST=broker.hivemq.com API_URL=http://localhost:8000 node-red node-red/flows.json
```

On a public broker anyone can read these topics: change `TOPIC` in the sketch and the MQTT nodes
of the flow to a prefix of your own. Run one node at a time (decisions are broadcast). Without
Wokwi, the Python virtual node speaks the same protocol: `python -m edgearbiter.virtual_node`.

wokwi.com and the PlatformIO registry were not reachable from the build environment, so the Wokwi
and PlatformIO paths are untested; the arduino-cli build and the QEMU run are.
