Host: GitHub-hosted runner (AMD EPYC 7763 64-Core Processor, 4 logical CPUs) · torch 2.14.1+cpu · weights: pretrained · 2026-10-27T22:32:11Z

| model | precision | input | threads | p50 ms | p95 ms | memory MB | size MB | top-1 (Imagenette, 10-way) |
|---|---|---|---|---|---|---|---|---|
| ssdlite320_mobilenet_v3_large | FP32 | 320px | 4 | 60.5 | 66.7 | 59.4 | 14.06 | — |
| ssdlite320_mobilenet_v3_large_int8 | INT8 (x86, PTQ) | 320px | 4 | 40.9 | 41.6 | 28.0 | 4.09 | — |

Detectors, backbone + heads only (the part that runs in INT8; no box decoding or NMS):

| model | precision | threads | p50 ms | p95 ms |
|---|---|---|---|---|
| ssdlite320_mobilenet_v3_large | FP32 | 4 | 30.7 | 34.9 |
| ssdlite320_mobilenet_v3_large_int8 | INT8 | 4 | 15.9 | 16.2 |
