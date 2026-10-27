MobileNetV3-Small on Imagenette (160px release, 128px input), **imagenet** initialisation · 5 epochs · batch 64 · AdamW, one-cycle lr 0.001 · best validation top-1 (10 classes, 128px centre crop): **94.5%** · 5 min on CPU (4 threads)

| epoch | train loss | val top-1 | seconds |
|---|---|---|---|
| 1 | 1.281 | 90.9% | 62.3 |
| 2 | 0.762 | 91.2% | 61.5 |
| 3 | 0.682 | 93.0% | 61.7 |
| 4 | 0.630 | 94.1% | 61.8 |
| 5 | 0.591 | 94.5% | 61.3 |
