ResNet-18 on Imagenette (160px release, 128px input), **imagenet** initialisation · 5 epochs · batch 64 · AdamW, one-cycle lr 0.001 · best validation top-1 (10 classes, 128px centre crop): **94.4%** · 11 min on CPU (4 threads)

| epoch | train loss | val top-1 | seconds |
|---|---|---|---|
| 1 | 0.954 | 81.4% | 124.2 |
| 2 | 0.848 | 86.2% | 126.8 |
| 3 | 0.753 | 90.6% | 121.6 |
| 4 | 0.641 | 93.4% | 128.2 |
| 5 | 0.582 | 94.4% | 128.8 |
