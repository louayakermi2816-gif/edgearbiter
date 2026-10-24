### Three devices (near, mid, far links) sharing one edge, battery 22.8 J per device per 300 s episode

Mean ± std over 20 test episodes (seeds 1000+; learned methods: pooled over 3 training seeds). Δ return: paired difference vs `best static medium 10fps local`, 95 % CI.

| policy | return | mean_accuracy | violation_rate | energy_j | dropped | offloaded | rejected | edge_cost | Δ return vs best static medium 10fps local |
|---|---|---|---|---|---|---|---|---|---|
| rule_based (edge rule) | 2752.5 ± 243.7 | 0.843 ± 0.005 | 0.020 ± 0.007 | 67.6 ± 1.7 | 28.2 ± 15.8 | 1351.5 ± 493.5 | 4.2 ± 3.8 | 78.5 ± 34.4 | -372.8 ± 106.3 |
| deadline_greedy (edge rule) | 3193.9 ± 239.6 | 0.869 ± 0.000 | 0.143 ± 0.039 | 62.3 ± 4.5 | 526.8 ± 276.7 | 2253.4 ± 777.7 | 8.6 ± 6.7 | 101.9 ± 36.1 | +68.6 ± 100.8 |
| deadline_greedy (link-aware edge) | 3157.4 ± 173.9 | 0.868 ± 0.000 | 0.025 ± 0.008 | 65.0 ± 2.9 | 730.9 ± 296.9 | 1399.0 ± 467.4 | 21.2 ± 4.9 | 87.3 ± 31.7 | +32.1 ± 73.4 |
| best static medium 10fps local | 3125.3 ± 24.1 | 0.869 ± 0.000 | 0.000 ± 0.000 | 68.4 ± 0.5 | 1.6 ± 2.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| single-device PPO on each device (edge rule) | 3451.7 ± 236.6 | 0.867 ± 0.002 | 0.094 ± 0.038 | 64.7 ± 3.9 | 17.9 ± 22.0 | 2034.2 ± 802.1 | 5.0 ± 4.6 | 87.3 ± 35.0 | +326.4 ± 56.5 |
| single-device PPO on each device (link-aware edge) | 3420.1 ± 184.6 | 0.867 ± 0.001 | 0.020 ± 0.007 | 66.4 ± 2.6 | 4.5 ± 7.0 | 1337.3 ± 485.8 | 12.9 ± 4.6 | 70.5 ± 32.9 | +294.9 ± 43.3 |
| IPPO, devices (edge rule) | 3410.8 ± 230.0 | 0.866 ± 0.001 | 0.033 ± 0.016 | 65.9 ± 3.0 | 23.4 ± 23.3 | 1523.9 ± 617.7 | 14.8 ± 16.8 | 81.3 ± 32.6 | +285.5 ± 55.3 |
| MAPPO, devices (edge rule) | 3408.5 ± 210.4 | 0.866 ± 0.002 | 0.027 ± 0.012 | 66.1 ± 2.9 | 18.0 ± 15.7 | 1427.6 ± 567.6 | 8.8 ± 10.1 | 76.2 ± 30.6 | +283.2 ± 50.0 |
| IPPO, devices + edge | 3400.2 ± 236.8 | 0.865 ± 0.002 | 0.030 ± 0.013 | 66.1 ± 2.9 | 11.2 ± 8.5 | 1485.0 ± 606.4 | 21.9 ± 19.5 | 87.0 ± 37.2 | +274.9 ± 57.1 |
| MAPPO, devices + edge | 3397.6 ± 204.0 | 0.865 ± 0.003 | 0.031 ± 0.017 | 66.1 ± 2.7 | 21.8 ± 12.9 | 1490.0 ± 481.8 | 8.8 ± 9.1 | 131.4 ± 77.6 | +272.4 ± 48.3 |
