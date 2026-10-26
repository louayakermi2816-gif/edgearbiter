## experiments synthetic

Cost table: `synthetic` (3718ae62b18a) · 3 seeds × 1,000,000 steps per run · trainer overrides: `{'gamma': 0.995, 'ent_coef': 0.02, 'ent_coefs': {'edge': 0.001}, 'bc_smoothing': 0.05}`

<picture><source media="(prefers-color-scheme: dark)" srcset="experiments_synthetic_learning_curves_dark.png"><img alt="Validation return vs training steps" src="experiments_synthetic_learning_curves.png"></picture>

Lines: mean validation return over training seeds (band: ± std); horizontal lines: baselines on the same validation episodes.

### Scenario `default` (battery 60 J per 300 s episode)

Mean ± std over 20 test episodes (seeds 1000+; learned methods: pooled over 3 training seeds, actors from each run's best validation evaluation). Δ return: paired difference vs `best static small 10fps local` with 95 % CI.

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static small 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 640.1 ± 48.4 | 0.680 ± 0.000 | 0.000 ± 0.000 | 17.6 ± 2.1 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -463.6 ± 73.0 |
| always_medium_local | 91.9 ± 26.0 | 0.810 ± 0.000 | 1.000 ± 0.000 | 60.2 ± 0.1 | 213.4 ± 87.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -1011.8 ± 100.2 |
| always_offload_medium | 853.8 ± 198.9 | 0.787 ± 0.027 | 0.330 ± 0.067 | 17.3 ± 4.4 | 44.0 ± 35.3 | 1218.7 ± 365.1 | 54.0 ± 57.9 | 93.1 ± 22.3 | -249.9 ± 46.3 |
| rule_based | 1057.5 ± 206.1 | 0.725 ± 0.016 | 0.020 ± 0.009 | 20.9 ± 4.0 | 12.8 ± 12.3 | 523.1 ± 219.1 | 1.4 ± 1.7 | 39.2 ± 10.4 | -46.3 ± 16.1 |
| deadline_greedy | 1065.4 ± 220.0 | 0.772 ± 0.011 | 0.099 ± 0.032 | 30.7 ± 9.1 | 5.0 ± 18.9 | 861.6 ± 315.0 | 1.9 ± 1.9 | 61.5 ± 15.2 | -38.3 ± 18.7 |
| best static small 10fps local | 1103.7 ± 202.9 | 0.740 ± 0.000 | 0.000 ± 0.000 | 49.6 ± 9.2 | 0.1 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| PPO (device, edge rule) | 1109.2 ± 219.6 | 0.747 ± 0.009 | 0.013 ± 0.011 | 38.8 ± 9.8 | 1.2 ± 3.1 | 369.6 ± 227.3 | 6.2 ± 12.6 | 24.1 ± 9.9 | +5.5 ± 9.0 |
| IPPO (device+edge) | 1109.9 ± 215.1 | 0.750 ± 0.010 | 0.011 ± 0.010 | 43.4 ± 9.8 | 1.3 ± 3.2 | 271.4 ± 227.9 | 8.9 ± 12.5 | 22.1 ± 15.9 | +6.2 ± 7.5 |
| MAPPO (device+edge) | 1108.4 ± 220.0 | 0.748 ± 0.011 | 0.012 ± 0.011 | 40.0 ± 9.5 | 2.1 ± 3.5 | 353.5 ± 253.0 | 11.2 ± 18.6 | 23.5 ± 18.9 | +4.7 ± 9.6 |
| MAPPO, edge from scratch | 1102.0 ± 204.5 | 0.739 ± 0.001 | 0.000 ± 0.000 | 49.1 ± 9.3 | 1.4 ± 3.4 | 1.1 ± 1.9 | 0.3 ± 0.5 | 0.8 ± 2.4 | -1.7 ± 1.5 |

Test return per training seed (step of the selected checkpoint):

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| PPO (device, edge rule) | 1112.0 (checkpoint 900k) | 1111.9 (checkpoint 650k) | 1103.6 (checkpoint 650k) |
| IPPO (device+edge) | 1101.5 (checkpoint 50k) | 1111.5 (checkpoint 900k) | 1116.6 (checkpoint 50k) |
| MAPPO (device+edge) | 1111.1 (checkpoint 851k) | 1113.9 (checkpoint 950k) | 1100.2 (checkpoint 400k) |
| MAPPO, edge from scratch | 1103.7 (checkpoint 50k) | 1099.5 (checkpoint 551k) | 1102.9 (checkpoint 300k) |

### Scenario `energy_constrained` (battery 40 J per 300 s episode)

Mean ± std over 20 test episodes (seeds 1000+; learned methods: pooled over 3 training seeds, actors from each run's best validation evaluation). Δ return: paired difference vs `best static tiny 10fps local` with 95 % CI.

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static tiny 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 640.1 ± 48.4 | 0.680 ± 0.000 | 0.000 ± 0.000 | 17.6 ± 2.1 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -397.7 ± 73.1 |
| always_medium_local | 58.0 ± 18.2 | 0.810 ± 0.000 | 1.000 ± 0.000 | 40.2 ± 0.1 | 131.8 ± 61.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -979.8 ± 96.7 |
| always_offload_medium | 853.8 ± 198.9 | 0.787 ± 0.027 | 0.330 ± 0.067 | 17.3 ± 4.4 | 44.0 ± 35.3 | 1218.7 ± 365.1 | 54.0 ± 57.9 | 93.1 ± 22.3 | -184.0 ± 42.5 |
| rule_based | 1057.5 ± 206.1 | 0.725 ± 0.016 | 0.020 ± 0.009 | 20.9 ± 4.0 | 12.8 ± 12.3 | 523.1 ± 219.1 | 1.4 ± 1.7 | 39.2 ± 10.4 | +19.6 ± 10.6 |
| deadline_greedy | 1014.8 ± 204.7 | 0.773 ± 0.010 | 0.100 ± 0.032 | 29.2 ± 6.8 | 52.1 ± 107.0 | 858.1 ± 316.1 | 1.9 ± 1.9 | 61.6 ± 15.2 | -23.1 ± 49.2 |
| best static tiny 10fps local | 1037.8 ± 204.0 | 0.680 ± 0.000 | 0.000 ± 0.000 | 23.8 ± 4.5 | 0.1 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| PPO (device, edge rule) | 1087.1 ± 212.8 | 0.740 ± 0.016 | 0.020 ± 0.010 | 29.5 ± 5.1 | 4.0 ± 6.3 | 496.3 ± 217.7 | 15.6 ± 24.3 | 28.9 ± 11.6 | +49.3 ± 7.4 |
| IPPO (device+edge) | 1087.2 ± 212.9 | 0.736 ± 0.015 | 0.017 ± 0.012 | 30.8 ± 6.1 | 1.2 ± 2.6 | 429.3 ± 239.7 | 4.9 ± 9.6 | 20.7 ± 13.0 | +49.4 ± 10.0 |
| MAPPO (device+edge) | 1084.4 ± 213.3 | 0.736 ± 0.017 | 0.017 ± 0.010 | 29.6 ± 5.3 | 2.0 ± 4.4 | 461.4 ± 240.0 | 17.5 ± 27.8 | 34.8 ± 27.3 | +46.5 ± 7.4 |
| MAPPO, edge from scratch | 1052.3 ± 196.6 | 0.701 ± 0.005 | 0.000 ± 0.000 | 32.4 ± 4.8 | 4.8 ± 6.7 | 0.0 ± 0.0 | 0.7 ± 0.7 | 0.0 ± 0.0 | +14.5 ± 6.2 |

Test return per training seed (step of the selected checkpoint):

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| PPO (device, edge rule) | 1098.5 (checkpoint 1000k) | 1084.9 (checkpoint 751k) | 1078.0 (checkpoint 950k) |
| IPPO (device+edge) | 1084.8 (checkpoint 700k) | 1083.6 (checkpoint 801k) | 1093.3 (checkpoint 751k) |
| MAPPO (device+edge) | 1093.1 (checkpoint 1000k) | 1080.8 (checkpoint 801k) | 1079.2 (checkpoint 201k) |
| MAPPO, edge from scratch | 1036.9 (checkpoint 801k) | 1061.5 (checkpoint 350k) | 1058.5 (checkpoint 451k) |
