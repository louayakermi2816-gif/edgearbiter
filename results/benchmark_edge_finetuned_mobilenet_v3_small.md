Host: GitHub-hosted runner (AMD EPYC 9V74 80-Core Processor, 4 logical CPUs) · torch 2.14.1+cpu · weights: trained · 2026-10-27T22:37:22Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small_imagenette | FP32 | 128px | 4 | 3.7 | 3.9 | 15.3 | 6.24 | 94.5% |
| mobilenet_v3_small_imagenette_int8 | INT8 (x86, PTQ) | 128px | 4 | 3.0 | 3.1 | 16.2 | 1.83 | 16.3% |
| mobilenet_v3_small_imagenette_int8qat | INT8 (x86, QAT) | 128px | 4 | 3.1 | 3.2 | 16.4 | 1.83 | 79.5% |
