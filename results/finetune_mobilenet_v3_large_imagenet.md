MobileNetV3-Large on Imagenette (160px release, 128px input), **imagenet** initialisation · 5 epochs · batch 64 · AdamW, one-cycle lr 0.001 · best validation top-1 (10 classes, 128px centre crop): **96.6%** · 10 min on CPU (4 threads)

| epoch | train loss | val top-1 | seconds |
|---|---|---|---|
| 1 | 1.103 | 94.0% | 123.8 |
| 2 | 0.672 | 91.3% | 122.3 |
| 3 | 0.615 | 94.3% | 124.4 |
| 4 | 0.563 | 96.5% | 122.8 |
| 5 | 0.535 | 96.6% | 122.7 |
