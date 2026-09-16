/*
 * EdgeArbiter device node — ESP32 (Arduino core)
 *
 * Publishes telemetry the policy needs (battery, queue, link RTT, frame arrivals)
 * on MQTT every second and applies the decision it receives back:
 *   publish   edgearbiter/telemetry   {"node":"esp32-1","battery":0.87,"queue":4,"rtt_ms":18,"arrivals":3,"uptime_s":120}
 *   subscribe edgearbiter/decision    {"tier":1,"rate_fps":5,"placement":"edge"}
 *
 * The LED blinks once per model tier (tiny=1, small=2, medium=3) each second so
 * the decision is visible on the bench. Frame "arrivals" are simulated by a PIR
 * or button on FRAME_PIN (rising edges); replace with a camera trigger if available.
 *
 * Libraries: WiFi (core), PubSubClient, ArduinoJson.
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* MQTT_HOST = "192.168.1.10";   // Mosquitto broker
const int   MQTT_PORT = 1883;
const char* NODE_ID   = "esp32-1";

const int LED_PIN   = 2;
const int FRAME_PIN = 4;                  // PIR / button simulating frame arrivals
const int BATT_PIN  = 34;                 // ADC on a divider from the battery (optional)

WiFiClient wifi;
PubSubClient mqtt(wifi);

volatile int  arrivals = 0;
int   queueLen = 0;
int   tier = 0, rateFps = 5;
bool  offload = false;
unsigned long lastTick = 0, lastRtt = 0;
float rttMs = 20.0f;

void IRAM_ATTR onFrame() { arrivals++; }

float readBattery() {
  int raw = analogRead(BATT_PIN);                      // 0..4095
  float v = raw / 4095.0f * 3.3f * 2.0f;               // 1:2 divider
  float frac = (v - 3.3f) / (4.2f - 3.3f);             // 1S Li-ion window
  if (raw < 100) return 1.0f;                          // no divider connected -> report full
  return constrain(frac, 0.0f, 1.0f);
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, payload, len)) return;
  tier    = doc["tier"] | tier;
  rateFps = doc["rate_fps"] | rateFps;
  offload = String((const char*)(doc["placement"] | "local")) == "edge";
}

void connectMqtt() {
  while (!mqtt.connected()) {
    if (mqtt.connect(NODE_ID)) { mqtt.subscribe("edgearbiter/decision"); }
    else delay(1000);
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(FRAME_PIN, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(FRAME_PIN), onFrame, RISING);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(250);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  connectMqtt();
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();
  unsigned long now = millis();
  if (now - lastTick < 1000) return;
  lastTick = now;

  // measure link RTT with a TCP connect to the broker (cheap proxy for latency)
  unsigned long t0 = millis();
  WiFiClient probe; if (probe.connect(MQTT_HOST, MQTT_PORT)) { rttMs = 0.7f * rttMs + 0.3f * (millis() - t0); probe.stop(); }

  // queue dynamics: frames arrive, the policy's sensing rate drains them
  int newFrames = arrivals; arrivals = 0;
  queueLen = max(0, min(30, queueLen + newFrames - rateFps));

  StaticJsonDocument<256> doc;
  doc["node"] = NODE_ID; doc["battery"] = readBattery(); doc["queue"] = queueLen;
  doc["rtt_ms"] = rttMs; doc["arrivals"] = newFrames; doc["uptime_s"] = now / 1000;
  doc["tier"] = tier; doc["placement"] = offload ? "edge" : "local";
  char buf[256]; size_t n = serializeJson(doc, buf);
  mqtt.publish("edgearbiter/telemetry", buf, n);

  for (int i = 0; i <= tier; i++) { digitalWrite(LED_PIN, HIGH); delay(60); digitalWrite(LED_PIN, LOW); delay(60); }
}
