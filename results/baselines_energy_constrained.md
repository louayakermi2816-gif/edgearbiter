Scenario `energy_constrained` · cost table `synthetic` · 20 test episodes · Δ return: paired difference vs `best static tiny 10fps local` with 95 % CI

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static tiny 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 640.1 ± 48.4 | 0.680 ± 0.000 | 0.000 ± 0.000 | 17.6 ± 2.1 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -397.7 ± 73.1 |
| always_medium_local | 58.0 ± 18.2 | 0.810 ± 0.000 | 1.000 ± 0.000 | 40.2 ± 0.1 | 131.8 ± 61.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -979.8 ± 96.7 |
| always_offload_medium | 853.8 ± 198.9 | 0.787 ± 0.027 | 0.330 ± 0.067 | 17.3 ± 4.4 | 44.0 ± 35.3 | 1218.7 ± 365.1 | 54.0 ± 57.9 | 93.1 ± 22.3 | -184.0 ± 42.5 |
| rule_based | 1057.5 ± 206.1 | 0.725 ± 0.016 | 0.020 ± 0.009 | 20.9 ± 4.0 | 12.8 ± 12.3 | 523.1 ± 219.1 | 1.4 ± 1.7 | 39.2 ± 10.4 | +19.6 ± 10.6 |
| deadline_greedy | 1014.8 ± 204.7 | 0.773 ± 0.010 | 0.100 ± 0.032 | 29.2 ± 6.8 | 52.1 ± 107.0 | 858.1 ± 316.1 | 1.9 ± 1.9 | 61.6 ± 15.2 | -23.1 ± 49.2 |
| best static tiny 10fps local | 1037.8 ± 204.0 | 0.680 ± 0.000 | 0.000 ± 0.000 | 23.8 ± 4.5 | 0.1 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
