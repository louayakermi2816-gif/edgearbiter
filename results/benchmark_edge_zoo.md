Host: edge (this VM) (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: random · 2026-10-26T23:43:17Z

Random-init weights: latency, size and memory are valid, accuracy is not measured.

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | FP32 | 224px | 1 | 6.1 | 9.2 | 19.4 | 10.3 | — |
| mobilenet_v3_small | FP32 | 224px | 4 | 6.2 | 9.4 | 20.0 | 10.3 | — |
| mobilenet_v3_large | FP32 | 224px | 1 | 15.9 | 23.2 | 41.3 | 22.13 | — |
| mobilenet_v3_large | FP32 | 224px | 4 | 13.5 | 16.6 | 40.1 | 22.13 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 1 | 25.6 | 35.7 | 19.4 | 5.65 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 4 | 22.4 | 27.2 | 19.5 | 5.65 | — |
| resnet18 | FP32 | 224px | 1 | 42.3 | 48.4 | 65.7 | 46.84 | — |
| resnet18 | FP32 | 224px | 4 | 13.3 | 20.3 | 65.7 | 46.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 1 | 11.3 | 16.7 | 18.6 | 11.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 4 | 4.8 | 6.0 | 19.6 | 11.84 | — |
| efficientnet_b0 | FP32 | 224px | 1 | 25.9 | 27.8 | 43.2 | 21.45 | — |
| efficientnet_b0 | FP32 | 224px | 4 | 20.8 | 29.9 | 50.1 | 21.45 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 1 | 69.2 | 76.3 | 54.1 | 14.06 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 4 | 59.1 | 69.5 | 51.0 | 14.06 | — |
| ssdlite320_mobilenet_v3_large_int8 | INT8 (x86, PTQ) | 320px | 1 | 49.3 | 75.0 | 28.8 | 4.09 | — |
| ssdlite320_mobilenet_v3_large_int8 | INT8 (x86, PTQ) | 320px | 4 | 46.4 | 63.8 | 28.9 | 4.09 | — |

Detectors, backbone + heads only (the part that runs in INT8; no box decoding or NMS):

| model | precision | threads | p50 ms | p95 ms |
|---|---|---|---|---|
| ssdlite320_mobilenet_v3_large | FP32 | 1 | 34.5 | 37.0 |
| ssdlite320_mobilenet_v3_large | FP32 | 4 | 26.7 | 32.5 |
| ssdlite320_mobilenet_v3_large_int8 | INT8 | 1 | 13.7 | 14.5 |
| ssdlite320_mobilenet_v3_large_int8 | INT8 | 4 | 9.0 | 12.4 |
