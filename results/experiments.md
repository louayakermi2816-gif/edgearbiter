## experiments

Cost table: `results/cost_table.json` (2a623037e244) · 3 seeds × 1,000,000 steps per run · trainer overrides: `{'gamma': 0.995, 'ent_coef': 0.02, 'ent_coefs': {'edge': 0.001}, 'bc_smoothing': 0.05}`

<picture><source media="(prefers-color-scheme: dark)" srcset="experiments_learning_curves_dark.png"><img alt="Validation return vs training steps" src="experiments_learning_curves.png"></picture>

Lines: mean validation return over training seeds (band: ± std); horizontal lines: baselines on the same validation episodes.

### Scenario `default` (battery 60 J per 300 s episode)

Mean ± std over 20 test episodes (seeds 1000+; learned methods: pooled over 3 training seeds, actors from each run's best validation evaluation). Δ return: paired difference vs `best static medium 10fps local` with 95 % CI.

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static medium 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 799.6 ± 67.3 | 0.826 ± 0.000 | 0.001 ± 0.000 | 28.8 ± 3.5 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -527.6 ± 90.1 |
| always_medium_local | 852.3 ± 73.7 | 0.869 ± 0.000 | 0.000 ± 0.000 | 21.1 ± 2.5 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -475.0 ± 87.3 |
| always_offload_medium | 1024.6 ± 223.9 | 0.862 ± 0.009 | 0.248 ± 0.055 | 20.1 ± 7.7 | 44.0 ± 35.3 | 1218.7 ± 365.1 | 54.0 ± 57.9 | 93.1 ± 22.3 | -302.7 ± 44.8 |
| rule_based | 1236.8 ± 239.6 | 0.841 ± 0.005 | 0.011 ± 0.007 | 30.9 ± 6.6 | 12.8 ± 12.3 | 523.1 ± 219.1 | 1.4 ± 1.7 | 39.2 ± 10.4 | -90.4 ± 11.4 |
| deadline_greedy | 1249.7 ± 258.3 | 0.869 ± 0.000 | 0.062 ± 0.024 | 21.4 ± 5.2 | 1.1 ± 2.5 | 888.5 ± 313.5 | 2.0 ± 2.0 | 64.0 ± 16.2 | -77.6 ± 9.7 |
| best static medium 10fps local | 1327.3 ± 260.9 | 0.869 ± 0.000 | 0.000 ± 0.000 | 28.6 ± 5.5 | 0.1 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| PPO (device, edge rule) | 1323.5 ± 260.2 | 0.868 ± 0.003 | 0.001 ± 0.001 | 29.0 ± 5.7 | 0.6 ± 1.9 | 0.3 ± 1.3 | 0.2 ± 0.7 | 0.0 ± 0.2 | -3.8 ± 1.8 |
| IPPO (device+edge) | 1315.9 ± 261.9 | 0.867 ± 0.002 | 0.005 ± 0.005 | 29.1 ± 5.6 | 0.6 ± 1.4 | 60.8 ± 95.8 | 2.2 ± 5.2 | 6.0 ± 9.7 | -11.4 ± 1.9 |
| MAPPO (device+edge) | 1320.6 ± 259.4 | 0.869 ± 0.001 | 0.000 ± 0.001 | 28.5 ± 5.5 | 2.8 ± 4.5 | 2.6 ± 5.1 | 0.6 ± 1.4 | 0.3 ± 0.6 | -6.6 ± 1.7 |
| MAPPO, edge from scratch | 1325.2 ± 261.4 | 0.868 ± 0.001 | 0.000 ± 0.000 | 28.8 ± 5.4 | 0.2 ± 0.6 | 0.0 ± 0.0 | 0.8 ± 1.8 | 0.0 ± 0.0 | -2.1 ± 0.7 |

Test return per training seed (step of the selected checkpoint):

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| PPO (device, edge rule) | 1320.2 (checkpoint 50k) | 1327.3 (checkpoint 50k) | 1323.0 (checkpoint 700k) |
| IPPO (device+edge) | 1313.6 (checkpoint 451k) | 1324.5 (checkpoint 50k) | 1309.7 (checkpoint 201k) |
| MAPPO (device+edge) | 1319.9 (checkpoint 551k) | 1320.1 (checkpoint 551k) | 1321.9 (checkpoint 100k) |
| MAPPO, edge from scratch | 1327.3 (checkpoint 50k) | 1321.7 (checkpoint 400k) | 1326.6 (checkpoint 551k) |

### Scenario `energy_constrained` (battery 22.8 J per 300 s episode, 70 % of the energy of the best fixed configuration in `default`)

Mean ± std over 20 test episodes (seeds 1000+; learned methods: pooled over 3 training seeds, actors from each run's best validation evaluation). Δ return: paired difference vs `best static medium 10fps edge` with 95 % CI.

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static medium 10fps edge |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 619.7 ± 28.3 | 0.826 ± 0.000 | 0.001 ± 0.000 | 22.8 ± 0.3 | 296.6 ± 106.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -344.6 ± 109.2 |
| always_medium_local | 831.3 ± 52.1 | 0.869 ± 0.000 | 0.000 ± 0.000 | 20.8 ± 2.1 | 400.2 ± 164.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -133.0 ± 90.5 |
| always_offload_medium | 964.3 ± 228.8 | 0.862 ± 0.008 | 0.252 ± 0.066 | 18.0 ± 3.7 | 43.4 ± 34.6 | 1206.1 ± 382.5 | 42.1 ± 37.0 | 90.3 ± 24.7 | +0.0 ± 0.0 |
| rule_based | 910.0 ± 94.5 | 0.842 ± 0.005 | 0.011 ± 0.008 | 22.6 ± 1.1 | 6.5 ± 5.7 | 428.4 ± 163.3 | 0.8 ± 1.1 | 31.3 ± 12.1 | -54.3 ± 73.9 |
| deadline_greedy | 1112.7 ± 179.0 | 0.869 ± 0.000 | 0.064 ± 0.023 | 19.5 ± 3.0 | 112.5 ± 154.3 | 857.4 ± 296.0 | 2.0 ± 2.0 | 64.3 ± 16.2 | +148.4 ± 49.2 |
| best static medium 10fps edge | 964.3 ± 228.8 | 0.862 ± 0.008 | 0.252 ± 0.066 | 18.0 ± 3.7 | 43.4 ± 34.6 | 1206.1 ± 382.5 | 42.1 ± 37.0 | 90.3 ± 24.7 | — |
| PPO (device, edge rule) | 1172.5 ± 197.6 | 0.867 ± 0.002 | 0.057 ± 0.022 | 20.2 ± 3.1 | 4.0 ± 7.7 | 840.0 ± 341.0 | 1.2 ± 1.4 | 50.2 ± 19.7 | +208.2 ± 24.9 |
| IPPO (device+edge) | 1134.7 ± 202.5 | 0.864 ± 0.003 | 0.073 ± 0.035 | 20.0 ± 3.2 | 4.6 ± 6.6 | 863.6 ± 338.7 | 7.0 ± 8.8 | 74.0 ± 32.4 | +170.4 ± 25.6 |
| MAPPO (device+edge) | 1162.0 ± 204.0 | 0.865 ± 0.003 | 0.067 ± 0.028 | 20.0 ± 3.0 | 3.9 ± 5.9 | 922.4 ± 328.0 | 2.3 ± 4.0 | 69.2 ± 33.6 | +197.7 ± 25.7 |
| MAPPO, edge from scratch | 1163.8 ± 192.5 | 0.869 ± 0.001 | 0.070 ± 0.034 | 20.1 ± 2.9 | 21.2 ± 17.6 | 895.9 ± 256.9 | 0.0 ± 0.0 | 89.7 ± 119.4 | +199.5 ± 31.9 |

Test return per training seed (step of the selected checkpoint):

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| PPO (device, edge rule) | 1169.0 (checkpoint 551k) | 1175.4 (checkpoint 650k) | 1173.1 (checkpoint 451k) |
| IPPO (device+edge) | 1147.2 (checkpoint 801k) | 1144.2 (checkpoint 201k) | 1112.7 (checkpoint 501k) |
| MAPPO (device+edge) | 1161.0 (checkpoint 501k) | 1148.2 (checkpoint 400k) | 1176.8 (checkpoint 600k) |
| MAPPO, edge from scratch | 1157.2 (checkpoint 551k) | 1167.8 (checkpoint 201k) | 1166.2 (checkpoint 300k) |
