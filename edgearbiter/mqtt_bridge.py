"""Optional MQTT publisher for decisions and simulator telemetry (consumed by Node-RED).

Configured through environment variables:
    MQTT_HOST   broker hostname (publisher disabled when unset)
    MQTT_PORT   default 1883
    MQTT_TOPIC  default "edgearbiter"
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


class Publisher:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, topic: Optional[str] = None):
        self.host = host or os.getenv("MQTT_HOST")
        self.port = int(port or os.getenv("MQTT_PORT", "1883"))
        self.topic = topic or os.getenv("MQTT_TOPIC", "edgearbiter")
        self._client = None
        if self.host:
            try:
                import paho.mqtt.client as mqtt
                self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                self._client.connect(self.host, self.port, keepalive=30)
                self._client.loop_start()
            except Exception as exc:  # broker unreachable: run without telemetry
                print(f"[mqtt] disabled: {exc}")
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def publish(self, subtopic: str, payload: Any) -> None:
        if self._client is None:
            return
        self._client.publish(f"{self.topic}/{subtopic}", json.dumps(payload), qos=0, retain=False)

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
