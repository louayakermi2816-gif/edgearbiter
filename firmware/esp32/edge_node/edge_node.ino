/*
 * EdgeArbiter device node — ESP32 (Arduino core 3.x). Runs on a board, in the Wokwi simulator, or in
 * Espressif's QEMU.
 *
 * Every second it publishes the telemetry the policy needs and applies the decision it receives back
 * (same protocol as the Python virtual node, edgearbiter/virtual_node.py):
 *   publish   <TOPIC>/telemetry  {"node","seq","battery","queue","rtt_ms","arrivals","uptime_s","tier","placement"}
 *             <TOPIC>/node       this second's outcome and running counts, for the Node-RED board
 *   subscribe <TOPIC>/decision   {"tier","rate_fps","placement",...}   (from Node-RED -> /policy/decide)
 *
 * Build options (-D on the compiler command line, or edit the defaults below):
 *   TRANSPORT_SERIAL 0  Wi-Fi + MQTT (PubSubClient).
 *                    1  the same messages as text lines on the serial port, "<subtopic> <json>":
 *                       telemetry and node lines out, decision lines in; edgearbiter/serial_bridge.py relays
 *                       them to and from the broker. For QEMU, which emulates the ESP32 but not its radio,
 *                       and for a board on USB without Wi-Fi.
 *   WOKWI            1  Wi-Fi "Wokwi-GUEST" and a public broker; frame arrivals generated with the
 *                       simulator's calm/busy Poisson workload. 0: a real board (your network and broker,
 *                       frames from the button or the camera).
 *   BATTERY_MODEL    0  battery read on GPIO34: the potentiometer in Wokwi, a 1:2 divider on a board.
 *                    1  simulated: each processed frame draws its tier's measured energy per inference
 *                       (results/cost_table.json), or the radio energy when offloaded, from the
 *                       energy_constrained budget. Default with TRANSPORT_SERIAL (QEMU has no ADC).
 *
 * The LED blinks tier+1 times per second (tiny=1, small=2, medium=3) with a non-blocking state machine; a
 * button on GPIO4 adds a frame per press. rtt_ms is the control-loop round trip (telemetry sent -> decision
 * received).
 *
 * Libraries: WiFi (core), PubSubClient, ArduinoJson 7.
 */
#ifndef TRANSPORT_SERIAL
#define TRANSPORT_SERIAL 0
#endif
#ifndef WOKWI
#define WOKWI 1
#endif
#ifndef BATTERY_MODEL
#define BATTERY_MODEL TRANSPORT_SERIAL
#endif

#include <ArduinoJson.h>
#if !TRANSPORT_SERIAL
#include <WiFi.h>
#include <PubSubClient.h>
#endif

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

// per tier (tiny, small, medium): top-1 and device energy per inference, from results/cost_table.json
const float TIER_ACCURACY[3]  = {0.826f, 0.836f, 0.869f};
const float TIER_ENERGY_MJ[3] = {24.9f, 35.3f, 18.1f};
const float RADIO_MJ = 10.0f;                  // sending one 40 KB frame to the edge
const float BUDGET_J = 22.8f;                  // energy_constrained budget per 300 s episode

// explicit prototypes: every toolchain (Arduino IDE / CLI, PlatformIO, Wokwi) builds the sketch the same way
void IRAM_ATTR onFrame();                              // the interrupt handler runs from IRAM
float readBattery();
int poisson(float lambda);
int frameArrivals();
void applyDecision(JsonDocument& doc);
void publishJson(const char* subtopic, JsonDocument& doc);
void pollTransport();
void updateLed(unsigned long now);
void setup();
void loop();
#if TRANSPORT_SERIAL
void readSerialLines();
#else
void onMessage(char* topic, byte* payload, unsigned int len);
void connectMqtt();
#endif

#if !TRANSPORT_SERIAL
WiFiClient wifi;
PubSubClient mqtt(wifi);
char topicDecision[64];
#endif

volatile int pressed = 0;
int   queueLen = 0, tier = 0, rateFps = 5;
bool  offload = false, busy = false;
unsigned long seq = 0, lastTick = 0, lastPublish = 0;
float rttMs = 20.0f;
unsigned long processedTotal = 0, offloadedTotal = 0, droppedTotal = 0;   // running counts for the board
float accuracySum = 0.0f, energyJ = 0.0f;

// non-blocking LED: blink tier+1 times at the start of every second
int ledToggles = 0;
unsigned long ledNext = 0;

void onFrame() { pressed = pressed + 1; }            // IRAM_ATTR on the prototype only: one section

float readBattery() {
#if BATTERY_MODEL
  return max(0.0f, 1.0f - energyJ / BUDGET_J);          // what is left of the episode's energy budget
#else
  int raw = analogRead(BATT_PIN);                      // 0..4095
#if WOKWI
  return raw / 4095.0f;                                // potentiometer = remaining energy budget
#else
  if (raw < 100) return 1.0f;                          // no divider connected -> report full
  float v = raw / 4095.0f * 3.3f * 2.0f;               // 1:2 divider
  return constrain((v - 3.3f) / (4.2f - 3.3f), 0.0f, 1.0f);   // 1S Li-ion window
#endif
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

void applyDecision(JsonDocument& doc) {
  tier    = constrain((int)(doc["tier"] | tier), 0, 2);
  rateFps = doc["rate_fps"] | rateFps;
  offload = strcmp(doc["placement"] | "local", "edge") == 0;
  rttMs   = 0.7f * rttMs + 0.3f * (millis() - lastPublish);
}

void publishJson(const char* subtopic, JsonDocument& doc) {
  char buf[512];
  size_t n = serializeJson(doc, buf, sizeof(buf));
#if TRANSPORT_SERIAL
  Serial.print(subtopic);
  Serial.print(' ');
  Serial.write((const uint8_t*)buf, n);
  Serial.print('\n');
#else
  char topic[64];
  snprintf(topic, sizeof(topic), "%s/%s", TOPIC, subtopic);
  mqtt.publish(topic, (const uint8_t*)buf, n, false);  // not retained
#endif
}

#if TRANSPORT_SERIAL
void readSerialLines() {                               // "decision {json}" lines from the bridge
  static char line[1024];
  static size_t len = 0;
  const char* prefix = "decision ";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') {
      if (len < sizeof(line) - 1) line[len++] = c;
      continue;
    }
    line[len] = '\0';
    len = 0;
    if (strncmp(line, prefix, strlen(prefix)) != 0) continue;
    JsonDocument doc;
    if (!deserializeJson(doc, line + strlen(prefix))) applyDecision(doc);
  }
}
#else
void onMessage(char* topic, byte* payload, unsigned int len) {
  JsonDocument doc;
  if (!deserializeJson(doc, payload, len)) applyDecision(doc);
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String id = String(NODE_ID) + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    if (mqtt.connect(id.c_str())) mqtt.subscribe(topicDecision);
    else delay(1000);
  }
}
#endif

void pollTransport() {
#if TRANSPORT_SERIAL
  readSerialLines();
#else
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();
#endif
}

void updateLed(unsigned long now) {
  if (ledToggles > 0 && now >= ledNext) {
    digitalWrite(LED_PIN, ledToggles % 2 == 0 ? HIGH : LOW);
    ledToggles--;
    ledNext = now + 120;
  }
}

void setup() {
  Serial.setRxBufferSize(1024);                        // a decision line is ~300 bytes (default buffer: 256)
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FRAME_PIN, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(FRAME_PIN), onFrame, RISING);
  randomSeed(esp_random());
#if TRANSPORT_SERIAL
  Serial.println("EdgeArbiter node ready (serial transport)");
#else
  snprintf(topicDecision, sizeof(topicDecision), "%s/decision", TOPIC);
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
#endif
}

void loop() {
  pollTransport();
  unsigned long now = millis();
  updateLed(now);
  if (now - lastTick < 1000) return;
  lastTick = now;

  // queue dynamics: frames arrive, the decided sensing rate drains them, a full queue drops the rest
  noInterrupts(); int buttonFrames = pressed; pressed = 0; interrupts();
  int newFrames = buttonFrames + frameArrivals();
  int waiting = queueLen + newFrames;
  int processed = min(waiting, rateFps);
  queueLen = min(30, waiting - processed);
  droppedTotal += waiting - processed - queueLen;
  processedTotal += processed;
  if (offload) offloadedTotal += processed;
  accuracySum += processed * TIER_ACCURACY[tier];
  energyJ += processed * (offload ? RADIO_MJ : TIER_ENERGY_MJ[tier]) / 1000.0f;

  JsonDocument tel;
  tel["node"] = NODE_ID; tel["seq"] = ++seq; tel["battery"] = readBattery(); tel["queue"] = queueLen;
  tel["rtt_ms"] = rttMs; tel["arrivals"] = newFrames; tel["uptime_s"] = now / 1000;
  tel["tier"] = tier; tel["placement"] = offload ? "edge" : "local";
  lastPublish = millis();
  publishJson("telemetry", tel);

  JsonDocument out;                                    // what the Node-RED board shows for this node
  out["node"] = NODE_ID; out["seq"] = seq; out["t"] = now / 1000; out["battery"] = readBattery();
  out["tier"] = tier; out["placement"] = offload ? "edge" : "local"; out["rate_fps"] = rateFps;
  out["processed"] = processed; out["queue"] = queueLen; out["loop_rtt_ms"] = roundf(rttMs * 10.0f) / 10.0f;
  JsonObject kpis = out["kpis"].to<JsonObject>();
  kpis["processed"] = processedTotal; kpis["offloaded"] = offloadedTotal; kpis["dropped"] = droppedTotal;
  kpis["energy_j"] = energyJ;
  if (processedTotal) kpis["mean_accuracy"] = accuracySum / processedTotal;
  publishJson("node", out);

  ledToggles = 2 * (tier + 1); ledNext = now;          // start this second's blink pattern
}
