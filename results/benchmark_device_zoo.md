Host: emulated device (1 thread, 0.25 CPU) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: random · 2026-10-23T20:36:02Z

Emulated device: inference thread limited to 0.25 CPU (cgroup CFS quota).

Random-init weights: latency, size and memory are valid, accuracy is not measured.

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | FP32 | 224px | 1 | 93.1 | 121.1 | 19.8 | 10.3 | — |
| mobilenet_v3_large | FP32 | 224px | 1 | 222.9 | 265.0 | 45.3 | 22.13 | — |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 1 | 199.8 | 255.5 | 19.1 | 5.65 | — |
| resnet18 | FP32 | 224px | 1 | 360.4 | 420.5 | 65.8 | 46.84 | — |
| resnet18_int8 | INT8 (fbgemm) | 224px | 1 | 120.4 | 160.6 | 18.2 | 11.84 | — |
| efficientnet_b0 | FP32 | 224px | 1 | 320.0 | 365.0 | 43.3 | 21.45 | — |
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 1 | 702.8 | 856.1 | 55.9 | 14.06 | — |
