# Formal Statistics Overview

- Standard runs indexed: 155
- Sequence runs indexed: 85
- Experiment groups: alpha_ablation=14, closed_loop_breakthrough=41, event_mechanism=22, extra_innovation=29, main=49
- Environments: HalfCheetah-v4=32, Hopper-v4=77, Walker2d-v4=46

## Best Final Avg10 By Environment

- HalfCheetah-v4: `PPO` (main), final avg10 mean = -813.114
- Hopper-v4: `PPO+Cycle` (main), final avg10 mean = 195.263
- Walker2d-v4: `P3O-dyn+HardDistill` (extra_innovation), final avg10 mean = 213.376

## Sequence Coverage

- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O` | steps/phase=20000 | runs=1
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O` | steps/phase=80000 | runs=10
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O-ClosedLoopFull` | steps/phase=20000 | runs=1
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O-ClosedLoopFull` | steps/phase=80000 | runs=20
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O-ClosedLoopMemory` | steps/phase=20000 | runs=1
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`P3O-ClosedLoopMemory` | steps/phase=80000 | runs=20
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`PPO` | steps/phase=20000 | runs=1
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`PPO` | steps/phase=80000 | runs=10
- Hopper-v4 -> Walker2d-v4 -> Hopper-v4 | algo=`legacy-unlabeled` | steps/phase=20000 | runs=1
- Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | algo=`P3O` | steps/phase=80000 | runs=5
- Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | algo=`P3O-ClosedLoopFull` | steps/phase=80000 | runs=5
- Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | algo=`P3O-ClosedLoopMemory` | steps/phase=80000 | runs=5
- Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4 | algo=`PPO` | steps/phase=80000 | runs=5
