Host: cloud VM, 4 vCPU (edge-server stand-in) (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs) · torch 2.14.1+cu130 · weights: random · 2026-10-23T15:55:38Z

Random-init weights: latency and size are valid, accuracy is not measured.

| model | int8 (engine) | input | threads | p50 ms | p95 ms | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | no | 224px | 1 | 11.9 | 13.9 | 10.3 | — |
| mobilenet_v3_small | no | 224px | 4 | 13.9 | 19.2 | 10.3 | — |
| mobilenet_v3_large | no | 224px | 1 | 25.9 | 34.6 | 22.13 | — |
| mobilenet_v3_large | no | 224px | 4 | 25.0 | 36.2 | 22.13 | — |
| mobilenet_v3_large_int8 | yes (qnnpack) | 224px | 1 | 36.1 | 50.7 | 5.65 | — |
| mobilenet_v3_large_int8 | yes (qnnpack) | 224px | 4 | 36.5 | 54.5 | 5.65 | — |
| resnet18 | no | 224px | 1 | 73.4 | 94.4 | 46.84 | — |
| resnet18 | no | 224px | 4 | 30.9 | 38.1 | 46.84 | — |
| resnet18_int8 | yes (fbgemm) | 224px | 1 | 20.6 | 31.9 | 11.84 | — |
| resnet18_int8 | yes (fbgemm) | 224px | 4 | 10.9 | 15.0 | 11.84 | — |
| efficientnet_b0 | no | 224px | 1 | 51.4 | 63.0 | 21.45 | — |
| efficientnet_b0 | no | 224px | 4 | 40.2 | 52.9 | 21.45 | — |
| ssdlite320_mobilenet_v3_large | no | 320px | 1 | 117.1 | 167.0 | 14.06 | — |
| ssdlite320_mobilenet_v3_large | no | 320px | 4 | 108.1 | 148.7 | 14.06 | — |
| mobilenet_v3_small_imagenette | no | 128px | 1 | 8.9 | 11.6 | 6.24 | — |
| mobilenet_v3_small_imagenette | no | 128px | 4 | 9.7 | 14.0 | 6.24 | — |
