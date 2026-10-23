# ESP32 node — run it in the Wokwi simulator (no hardware)

The sketch publishes telemetry every second and applies the decisions that come back from
the policy service through Node-RED (topics `edgearbiter/telemetry` and `edgearbiter/decision`).
With `#define WOKWI 1` (default) it uses Wokwi's simulated Wi-Fi (`Wokwi-GUEST`), the public
broker `broker.hivemq.com`, generates frame arrivals with the simulator's calm/busy workload, and
reads the battery level from the potentiometer — turn it down and watch the policy react.

**Web simulator** — create a new ESP32 project on wokwi.com, paste `edge_node.ino`, replace
`diagram.json`, add the libraries from `libraries.txt` (Library Manager), start the simulation.

**VS Code** — Wokwi extension + PlatformIO: `pio run` in this folder, then *Wokwi: Start Simulator*
(`wokwi.toml` points at the PlatformIO build).

**Close the loop** — run the policy service and Node-RED on your machine against the same broker:

```bash
uvicorn edgearbiter.api:app --port 8000
MQTT_HOST=broker.hivemq.com API_URL=http://localhost:8000 node-red node-red/flows.json
```

On a public broker anyone can read these topics: change `TOPIC` in the sketch and the five MQTT
nodes of the flow to a prefix of your own. Run one node at a time (decisions are broadcast).
Without Wokwi, the Python virtual node speaks the same protocol: `python -m edgearbiter.virtual_node`.

Not verified in the project's build environment (no Arduino toolchain or wokwi.com access there);
the Python virtual node and the Node-RED flow are tested end to end.
