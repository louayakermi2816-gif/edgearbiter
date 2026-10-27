Host: GitHub-hosted runner (AMD EPYC 9V45 96-Core Processor, 4 logical CPUs) · torch 2.14.1+cpu · weights: pretrained · 2026-10-27T22:32:03Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | FP32 | 224px | 4 | 3.1 | 3.2 | 23.7 | 10.3 | 99.1% |
| mobilenet_v3_large | FP32 | 224px | 4 | 7.6 | 8.3 | 47.6 | 22.13 | 99.5% |
| mobilenet_v3_large_int8 | INT8 (qnnpack) | 224px | 4 | 16.0 | 17.7 | 19.4 | 5.65 | 99.8% |
| resnet18 | FP32 | 224px | 4 | 12.5 | 12.8 | 66.9 | 46.84 | 99.0% |
| resnet18_int8 | INT8 (fbgemm) | 224px | 4 | 3.4 | 3.5 | 21.4 | 11.84 | 99.0% |
| efficientnet_b0 | FP32 | 224px | 4 | 13.4 | 15.7 | 58.6 | 21.45 | 99.8% |
