Host: GitHub-hosted runner (AMD EPYC 9V74 80-Core Processor, 4 logical CPUs) · torch 2.14.1+cpu · weights: trained · 2026-10-27T22:42:30Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_imagenette | FP32 | 128px | 4 | 7.3 | 7.4 | 31.5 | 17.06 | 96.6% |
| mobilenet_v3_large_imagenette_int8 | INT8 (x86, PTQ) | 128px | 4 | 3.6 | 3.7 | 20.5 | 4.7 | 62.1% |
| mobilenet_v3_large_imagenette_int8qat | INT8 (x86, QAT) | 128px | 4 | 3.7 | 3.8 | 20.0 | 4.7 | 89.3% |
