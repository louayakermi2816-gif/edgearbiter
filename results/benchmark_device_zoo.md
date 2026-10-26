Host: emulated device (1 thread, 0.25 CPU) (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: random · 2026-10-26T23:37:50Z

Emulated device: inference thread limited to 0.25 CPU (cgroup CFS quota).

Random-init weights: latency, size and memory are valid, accuracy is not measured.

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | FP32 | 224px | 1 | 37.0 | 40.3 | 19.6 | 10.3 | — |
| mobilenet_v3_large | FP32 | 224px | 1 | 81.4 | 102.6 | 45.2 | 22.13 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 1 | 119.7 | 132.3 | 19.5 | 5.65 | — |
| resnet18 | FP32 | 224px | 1 | 187.6 | 225.3 | 65.8 | 46.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 1 | 57.4 | 63.2 | 17.9 | 11.84 | — |
| efficientnet_b0 | FP32 | 224px | 1 | 125.3 | 140.4 | 43.1 | 21.45 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 1 | 338.9 | 401.1 | 54.6 | 14.06 | — |
| ssdlite320_mobilenet_v3_large_int8 | INT8 (x86, PTQ) | 320px | 1 | 237.9 | 318.3 | 28.9 | 4.09 | — |

Detectors, backbone + heads only (the part that runs in INT8; no box decoding or NMS):

| model | precision | threads | p50 ms | p95 ms |
|---|---|---|---|---|
| ssdlite320_mobilenet_v3_large | FP32 | 1 | 182.5 | 201.5 |
| ssdlite320_mobilenet_v3_large_int8 | INT8 | 1 | 75.1 | 85.1 |
