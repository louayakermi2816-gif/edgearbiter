Host: edge (this VM) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: random · 2026-10-23T20:41:30Z

Random-init weights: latency, size and memory are valid, accuracy is not measured.

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | FP32 | 224px | 1 | 13.4 | 17.2 | 19.8 | 10.3 | — |
| mobilenet_v3_small | FP32 | 224px | 4 | 14.5 | 22.4 | 20.2 | 10.3 | — |
| mobilenet_v3_large | FP32 | 224px | 1 | 37.1 | 46.4 | 45.4 | 22.13 | — |
| mobilenet_v3_large | FP32 | 224px | 4 | 32.2 | 43.1 | 40.3 | 22.13 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 1 | 38.1 | 59.3 | 19.5 | 5.65 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 4 | 40.2 | 50.9 | 19.4 | 5.65 | — |
| resnet18 | FP32 | 224px | 1 | 72.9 | 101.2 | 65.7 | 46.84 | — |
| resnet18 | FP32 | 224px | 4 | 30.4 | 41.8 | 65.8 | 46.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 1 | 21.0 | 33.6 | 17.6 | 11.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 4 | 9.6 | 16.0 | 20.0 | 11.84 | — |
| efficientnet_b0 | FP32 | 224px | 1 | 59.1 | 72.6 | 48.3 | 21.45 | — |
| efficientnet_b0 | FP32 | 224px | 4 | 36.9 | 47.8 | 45.2 | 21.45 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 1 | 124.8 | 164.7 | 56.7 | 14.06 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 4 | 110.0 | 137.3 | 51.3 | 14.06 | — |
