COCO val2017 box mAP · AMD EPYC 7763 64-Core Processor, torch 2.14.1+cpu · 2026-10-27T22:44:24Z

| model | precision | mAP@[.5:.95] | mAP@.5 | mAP@.75 | images | evaluation time |
|---|---|---|---|---|---|---|
| ssdlite320_mobilenet_v3_large | FP32 | 21.3 | 34.3 | 22.1 | 5000 | 4.3 min |
| ssdlite320_mobilenet_v3_large_int8 | INT8 (x86, PTQ) | 19.3 | 31.6 | 20.1 | 5000 | 3.3 min |
