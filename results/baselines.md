Scenario `default` · cost table `synthetic` · 20 test episodes · Δ return: paired difference vs `best static small 10fps local` with 95 % CI

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static small 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| always_tiny_local | 640.1 ± 48.4 | 0.680 ± 0.000 | 0.000 ± 0.000 | 17.6 ± 2.1 | 402.6 ± 166.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -463.6 ± 73.0 |
| always_medium_local | 91.9 ± 26.0 | 0.810 ± 0.000 | 1.000 ± 0.000 | 60.2 ± 0.1 | 213.4 ± 87.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | -1011.8 ± 100.2 |
| always_offload_medium | 853.8 ± 198.9 | 0.787 ± 0.027 | 0.330 ± 0.067 | 17.3 ± 4.4 | 44.0 ± 35.3 | 1218.7 ± 365.1 | 54.0 ± 57.9 | 93.1 ± 22.3 | -249.9 ± 46.3 |
| rule_based | 1057.5 ± 206.1 | 0.725 ± 0.016 | 0.020 ± 0.009 | 20.9 ± 4.0 | 12.8 ± 12.3 | 523.1 ± 219.1 | 1.4 ± 1.7 | 39.2 ± 10.4 | -46.3 ± 16.1 |
| deadline_greedy | 1065.4 ± 220.0 | 0.772 ± 0.011 | 0.099 ± 0.032 | 30.7 ± 9.1 | 5.0 ± 18.9 | 861.6 ± 315.0 | 1.9 ± 1.9 | 61.5 ± 15.2 | -38.3 ± 18.7 |
| best static small 10fps local | 1103.7 ± 202.9 | 0.740 ± 0.000 | 0.000 ± 0.000 | 49.6 ± 9.2 | 0.1 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
