#!/usr/bin/env sh
# Build the ESP32 node with arduino-cli (esp32 core 3.3.x, PubSubClient 2.8, ArduinoJson 7).
#
#   firmware/esp32/build.sh mqtt   [out-dir] [arduino-cli args...]   Wi-Fi + MQTT: Wokwi, a board
#   firmware/esp32/build.sh serial [out-dir] [arduino-cli args...]   serial transport: QEMU, a board on USB
#
# The serial build uses DIO flash mode, which Espressif's QEMU emulates (QIO, the board default, fails
# at boot there) and every ESP32 board supports.
#
# One-time setup:
#   arduino-cli core install esp32:esp32@3.3.12 \
#       --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
#   arduino-cli lib install PubSubClient@2.8 ArduinoJson@7.4.3
#
# The output directory gets edge_node.ino.bin (the app) and edge_node.ino.merged.bin (bootloader,
# partition table and app in one flash image: QEMU, or esptool write_flash 0x0).
set -eu
transport=${1:-mqtt}
here=$(cd "$(dirname "$0")" && pwd)
out=${2:-$here/build/$transport}
[ $# -ge 2 ] && shift 2 || shift $#
case $transport in
  mqtt) defines= options= ;;
  serial) defines=-DTRANSPORT_SERIAL=1 options=FlashMode=dio ;;
  *) echo "usage: $0 mqtt|serial [out-dir] [arduino-cli args...]" >&2; exit 2 ;;
esac
arduino-cli compile --fqbn esp32:esp32:esp32${options:+:$options} --warnings default --build-property "build.defines=$defines" \
    --output-dir "$out" "$@" "$here/edge_node"
