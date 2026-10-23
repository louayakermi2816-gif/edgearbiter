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
