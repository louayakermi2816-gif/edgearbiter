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
