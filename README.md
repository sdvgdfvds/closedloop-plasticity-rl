# Closed-Loop Plasticity RL

Code accompanying the paper on closed-loop memory-enhanced adaptive plasticity control in deep reinforcement learning.

## Included source files

- `run_p3o_extra_innovations.py`: single-task experiments and single-task ablations
- `run_sequence_plasticity.py`: continual learning sequence experiments
- `run_closedloop_seq_ablation.py`: progressive sequence ablation
- `plot_closedloop_paper.py`: figure and table generation

## Included paper results

- `results/`: figures and summary tables used for the paper
- `logs/main_sequence/`: main continual-learning sequence raw logs (`metrics.csv` and `config.json`)
- `logs/single_task/`: single-task experiment raw logs for Table 1 / Figure 2
- `logs/single_task_ablation/`: single-task ablation raw logs for Table 3
- `logs/supp_sequence/`: supplementary continual-learning sequence raw logs for Table 4 / Figure 7
- `key_main_sequence_logs.zip`: compressed copy of the key main-sequence logs for convenient download

## Environment

- Python 3.11
- Gymnasium
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- PyTorch

## Notes

- The original scripts were used directly for the experiments reported in the paper.
- Experiment outputs are written under the run directory configured in the scripts or by environment variables / CLI arguments where supported.
- The plotting script reads experiment logs and writes figures/tables to the configured output directory.

## Repository link for the paper

`https://github.com/sdvgdfvds/closedloop-plasticity-rl`
