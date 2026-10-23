/*
 * EdgeArbiter device node — ESP32 (Arduino core). Runs on a board or in the Wokwi simulator.
 *
 * Every second it publishes the telemetry the policy needs and applies the decision it
 * receives back (same protocol as the Python virtual node, edgearbiter/virtual_node.py):
 *   publish   <TOPIC>/telemetry  {"node","seq","battery","queue","rtt_ms","arrivals","uptime_s","tier","placement"}
 *   subscribe <TOPIC>/decision   {"tier","rate_fps","placement",...}   (from Node-RED -> /policy/decide)
 *
 * WOKWI 1 (default): Wi-Fi "Wokwi-GUEST", public MQTT broker, frame arrivals generated with the
 * simulator's calm/busy Poisson workload, battery read from the potentiometer on GPIO34.
 * Point Node-RED at the same broker (MQTT_HOST) and use your own TOPIC so you do not share
 * topics with other users of the public broker. Set WOKWI 0 for a real board.
 *
 * The LED blinks tier+1 times per second (tiny=1, small=2, medium=3) with a non-blocking state
 * machine; a button on GPIO4 adds a frame per press. rtt_ms is the control-loop round trip
 * (telemetry published -> decision received).
 *
 * Libraries: WiFi (core), PubSubClient, ArduinoJson.
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#define WOKWI 1

#if WOKWI
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";
const char* MQTT_HOST = "broker.hivemq.com";   // public broker reachable from the simulator
#else
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* MQTT_HOST = "192.168.1.10";        // Mosquitto on your machine
#endif
const int   MQTT_PORT = 1883;
const char* TOPIC     = "edgearbiter";         // use a unique prefix on a public broker
const char* NODE_ID   = "esp32-1";

const int LED_PIN   = 2;
const int FRAME_PIN = 4;                       // button: one frame per press
const int BATT_PIN  = 34;                      // potentiometer (Wokwi) or battery divider (board)

WiFiClient wifi;
PubSubClient mqtt(wifi);
char topicTelemetry[64], topicDecision[64];

volatile int pressed = 0;
int   queueLen = 0, tier = 0, rateFps = 5;
bool  offload = false, busy = false;
unsigned long seq = 0, lastTick = 0, lastPublish = 0;
float rttMs = 20.0f;

// non-blocking LED: blink tier+1 times at the start of every second
int ledToggles = 0;
unsigned long ledNext = 0;

void IRAM_ATTR onFrame() { pressed++; }

float readBattery() {
  int raw = analogRead(BATT_PIN);                      // 0..4095
#if WOKWI
  return raw / 4095.0f;                                // potentiometer = remaining energy budget
#else
  if (raw < 100) return 1.0f;                          // no divider connected -> report full
  float v = raw / 4095.0f * 3.3f * 2.0f;               // 1:2 divider
  return constrain((v - 3.3f) / (4.2f - 3.3f), 0.0f, 1.0f);   // 1S Li-ion window
#endif
}

int poisson(float lambda) {                            // Knuth's method, fine for small lambda
  float limit = expf(-lambda), p = 1.0f;
  int k = 0;
  do { k++; p *= random(1, 10001) / 10000.0f; } while (p > limit);
  return k - 1;
}

int frameArrivals() {
#if WOKWI
  if (random(0, 100) < 5) busy = !busy;                // calm (2 fps) / busy (9 fps) regimes
  return poisson(busy ? 9.0f : 2.0f);
#else
  return 0;                                            // frames come from the camera / button
#endif
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  StaticJsonDocument<512> doc;
  if (deserializeJson(doc, payload, len)) return;
  tier    = doc["tier"] | tier;
  rateFps = doc["rate_fps"] | rateFps;
  offload = String((const char*)(doc["placement"] | "local")) == "edge";
  rttMs   = 0.7f * rttMs + 0.3f * (millis() - lastPublish);
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String id = String(NODE_ID) + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    if (mqtt.connect(id.c_str())) mqtt.subscribe(topicDecision);
    else delay(1000);
  }
}

void updateLed(unsigned long now) {
  if (ledToggles > 0 && now >= ledNext) {
    digitalWrite(LED_PIN, ledToggles % 2 == 0 ? HIGH : LOW);
    ledToggles--;
    ledNext = now + 120;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FRAME_PIN, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(FRAME_PIN), onFrame, RISING);
  snprintf(topicTelemetry, sizeof(topicTelemetry), "%s/telemetry", TOPIC);
  snprintf(topicDecision, sizeof(topicDecision), "%s/decision", TOPIC);
  randomSeed(esp_random());
#if WOKWI
  WiFi.begin(WIFI_SSID, WIFI_PASS, 6);
#else
  WiFi.begin(WIFI_SSID, WIFI_PASS);
#endif
  while (WiFi.status() != WL_CONNECTED) delay(250);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setBufferSize(1024);                            // decisions are larger than PubSubClient's 256-byte default
  connectMqtt();
  Serial.println("EdgeArbiter node connected");
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();
  unsigned long now = millis();
  updateLed(now);
  if (now - lastTick < 1000) return;
  lastTick = now;

  // queue dynamics: frames arrive, the decided sensing rate drains them
  noInterrupts(); int buttonFrames = pressed; pressed = 0; interrupts();
  int newFrames = buttonFrames + frameArrivals();
  queueLen = max(0, min(30, queueLen + newFrames - rateFps));

  StaticJsonDocument<384> doc;
  doc["node"] = NODE_ID; doc["seq"] = ++seq; doc["battery"] = readBattery(); doc["queue"] = queueLen;
  doc["rtt_ms"] = rttMs; doc["arrivals"] = newFrames; doc["uptime_s"] = now / 1000;
  doc["tier"] = tier; doc["placement"] = offload ? "edge" : "local";
  char buf[384];
  serializeJson(doc, buf, sizeof(buf));
  lastPublish = millis();
  mqtt.publish(topicTelemetry, buf);                   // const char* overload: not retained

  ledToggles = 2 * (tier + 1); ledNext = now;          // start this second's blink pattern
}
