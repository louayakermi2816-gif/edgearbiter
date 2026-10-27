Host: GitHub-hosted runner (AMD EPYC 9V45 96-Core Processor, 4 logical CPUs) · torch 2.14.1+cpu · weights: trained · 2026-10-27T22:42:34Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| resnet18_imagenette | FP32 | 128px | 4 | 8.9 | 9.0 | 56.4 | 44.8 | 94.4% |
| resnet18_imagenette_int8 | INT8 (x86, PTQ) | 128px | 4 | 1.2 | 1.2 | 26.1 | 11.31 | 94.5% |
| resnet18_imagenette_int8qat | INT8 (x86, QAT) | 128px | 4 | 1.1 | 1.2 | 25.7 | 11.31 | 91.5% |
