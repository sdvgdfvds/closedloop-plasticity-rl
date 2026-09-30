# Formal Pairwise Tests

| scope | context | steps | metric | reference_algo | candidate_algo | pairs | reference_mean | candidate_mean | mean_delta_candidate_minus_reference | paired_signflip_pvalue | better_algo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | HalfCheetah-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | -821.762 | -1130.400 | -308.638 | 0.0312 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 300000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 3 | 123.665 | 82.059 | -41.605 | 0.2500 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | 135.142 | 133.440 | -1.702 | 0.9062 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Walker2d-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | 104.956 | 81.668 | -23.288 | 0.4688 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | HalfCheetah-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | -763.698 | -955.875 | -192.177 | 0.0312 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 300000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 3 | 90.144 | 87.793 | -2.351 | 0.7500 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | 96.675 | 114.537 | 17.862 | 0.1562 | P3O-ClosedLoopFull |
| standard | Walker2d-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopFull | 6 | 93.017 | 68.525 | -24.492 | 0.2812 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | HalfCheetah-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | -822.045 | -839.173 | -17.129 | 0.7500 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 300000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 3 | 123.665 | 96.417 | -27.248 | 0.7500 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | 140.207 | 135.312 | -4.894 | 0.7500 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Walker2d-v4 | 500000 | final_avg10 | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | 116.254 | 144.513 | 28.259 | 0.3750 | P3O-ClosedLoopAlpha |
| standard | HalfCheetah-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | -763.708 | -779.359 | -15.651 | 0.3750 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| standard | Hopper-v4 | 300000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 3 | 90.144 | 94.075 | 3.931 | 0.7500 | P3O-ClosedLoopAlpha |
| standard | Hopper-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | 98.874 | 104.590 | 5.716 | 0.5625 | P3O-ClosedLoopAlpha |
| standard | Walker2d-v4 | 500000 | auc_avg10_norm | P3O-dyn+EvtSBP+AdaReset+HardDistill | P3O-ClosedLoopAlpha | 5 | 98.117 | 97.450 | -0.667 | 0.9375 | P3O-dyn+EvtSBP+AdaReset+HardDistill |
| sequence | Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | 80000 | final_eval_a_reward | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 20 | 78.511 | 162.797 | 84.286 | 0.0054 | P3O-ClosedLoopMemory |
| sequence | Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | 80000 | final_eval_a_reward | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 5 | 195.246 | 140.281 | -54.965 | 0.5625 | P3O-ClosedLoopFull |
| sequence | Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | 80000 | final_retention_a | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 20 | 0.466 | 0.758 | 0.291 | 0.0443 | P3O-ClosedLoopMemory |
| sequence | Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | 80000 | final_retention_a | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 5 | 5.826 | 0.541 | -5.284 | 0.0625 | P3O-ClosedLoopFull |
| sequence | Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | 80000 | final_forgetting_a | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 20 | 147.808 | 116.473 | -31.334 | 0.1802 | P3O-ClosedLoopMemory |
| sequence | Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | 80000 | final_forgetting_a | P3O-ClosedLoopFull | P3O-ClosedLoopMemory | 5 | 49.239 | 114.056 | 64.817 | 0.1250 | P3O-ClosedLoopFull |
