Host: emulated device (1 thread, 0.25 CPU) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: trained · 2026-10-23T20:25:07Z

Emulated device: inference thread limited to 0.25 CPU (cgroup CFS quota).

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small_imagenette_scratch | FP32 | 128px | 1 | 55.4 | 75.0 | 12.2 | 6.24 | 82.6% |
| mobilenet_v3_small_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 60.9 | 85.7 | 8.5 | 1.83 | 82.5% |
| mobilenet_v3_large_imagenette_scratch | FP32 | 128px | 1 | 100.7 | 137.2 | 26.7 | 17.06 | 83.9% |
| mobilenet_v3_large_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 78.5 | 109.4 | 13.3 | 4.7 | 83.6% |
| resnet18_imagenette_scratch | FP32 | 128px | 1 | 146.0 | 191.5 | 53.0 | 44.8 | 86.7% |
| resnet18_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 40.3 | 56.5 | 26.1 | 11.31 | 86.9% |
