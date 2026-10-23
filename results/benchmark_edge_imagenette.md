Host: cloud VM, 4 vCPU (edge-server stand-in) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: trained · 2026-10-23T16:50:04Z

| model | int8 (engine) | input | threads | p50 ms | p95 ms | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_small_imagenette_scratch | no | 128px | 1 | 9.2 | 14.6 | 6.24 | 82.6% |
| mobilenet_v3_small_imagenette_scratch | no | 128px | 4 | 12.2 | 16.4 | 6.24 | 82.6% |
