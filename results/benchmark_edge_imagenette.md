Host: edge (this VM) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: trained · 2026-10-23T20:30:34Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small_imagenette_scratch | FP32 | 128px | 1 | 9.1 | 13.6 | 12.1 | 6.24 | 82.6% |
| mobilenet_v3_small_imagenette_scratch | FP32 | 128px | 2 | 9.3 | 12.3 | 13.1 | 6.24 | 82.6% |
| mobilenet_v3_small_imagenette_scratch | FP32 | 128px | 4 | 9.2 | 14.2 | 13.2 | 6.24 | 82.6% |
| mobilenet_v3_small_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 9.4 | 15.6 | 8.5 | 1.83 | 82.5% |
| mobilenet_v3_small_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 2 | 8.5 | 12.8 | 8.7 | 1.83 | 82.5% |
| mobilenet_v3_small_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 4 | 9.0 | 10.9 | 9.7 | 1.83 | 82.5% |
| mobilenet_v3_large_imagenette_scratch | FP32 | 128px | 1 | 17.5 | 25.4 | 26.9 | 17.06 | 83.9% |
| mobilenet_v3_large_imagenette_scratch | FP32 | 128px | 2 | 17.5 | 23.0 | 28.4 | 17.06 | 83.9% |
| mobilenet_v3_large_imagenette_scratch | FP32 | 128px | 4 | 15.1 | 21.2 | 29.9 | 17.06 | 83.9% |
| mobilenet_v3_large_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 11.9 | 17.8 | 13.1 | 4.7 | 83.6% |
| mobilenet_v3_large_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 2 | 11.5 | 13.3 | 13.6 | 4.7 | 83.6% |
| mobilenet_v3_large_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 4 | 11.8 | 17.0 | 14.0 | 4.7 | 83.6% |
| resnet18_imagenette_scratch | FP32 | 128px | 1 | 28.0 | 38.8 | 53.0 | 44.8 | 86.7% |
| resnet18_imagenette_scratch | FP32 | 128px | 2 | 19.3 | 24.8 | 54.1 | 44.8 | 86.7% |
| resnet18_imagenette_scratch | FP32 | 128px | 4 | 13.7 | 19.0 | 55.7 | 44.8 | 86.7% |
| resnet18_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 1 | 7.5 | 11.6 | 23.5 | 11.31 | 86.9% |
| resnet18_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 2 | 5.9 | 8.5 | 26.8 | 11.31 | 86.9% |
| resnet18_imagenette_scratch_int8 | INT8 (x86, PTQ) | 128px | 4 | 5.0 | 7.1 | 26.8 | 11.31 | 86.9% |
