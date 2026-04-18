# -*- coding: utf-8 -*-
"""Progressive ablation for continual learning sequence.

Runs 2 new algorithm variants on Hopper->Walker2d->Hopper main sequence,
10 seeds each = 20 runs total.

Progressive chain:
  P3O (B)                         <- already have 10 seeds
  B+Alpha                         <- NEW (this script)
  B+Alpha+Reset                   <- NEW (this script)
  ClosedLoopFull (B+A+R+Distill)  <- already have 20 seeds
  ClosedLoopMemory (Full+Memory)  <- already have 20 seeds
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from run_sequence_plasticity import run_sequence, HP as SEQ_HP


def _worker_init():
    import warnings
    warnings.filterwarnings("ignore")


# Progressive ablation configs
SEQ_ABLATION_ALGOS = {
    # B+Alpha: P3O + dynamic/adaptive/dual alpha, no reset/distill
    "Seq-B+Alpha": {
        "use_sbp": True,
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": False,
        "hard_distill": False,
        "selective_reset": False,
        "event_trigger_sbp": False,
        "anchor_memory": False,
    },
    # B+Alpha+Reset: above + adaptive reset + event trigger + selective reset
    "Seq-B+Alpha+Reset": {
        "use_sbp": True,
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": False,
        "selective_reset": True,
        "event_trigger_sbp": True,
        "anchor_memory": False,
    },
}

SEQUENCE = ["Hopper-v4", "Walker2d-v4", "Hopper-v4"]
SEEDS = list(range(10))


def run_one(algo_name, cfg, seed):
    hp = dict(SEQ_HP)
    run_sequence(SEQUENCE, seed, hp, algo_name, cfg)
    return f"{algo_name}_seed{seed}"


def main():
    import gymnasium as gym
    for env_name in ["Hopper-v4", "Walker2d-v4"]:
        e = gym.make(env_name)
        e.close()

    tasks = []
    for algo_name, cfg in SEQ_ABLATION_ALGOS.items():
        for seed in SEEDS:
            tasks.append((algo_name, dict(cfg), seed))

    n_workers = min(12, len(tasks))
    print(f"Sequence Progressive Ablation: {len(tasks)} runs, {n_workers} workers")
    for algo in SEQ_ABLATION_ALGOS:
        print(f"  {algo}: seeds {SEEDS[0]}-{SEEDS[-1]}")
    print(flush=True)

    start = time.time()
    done_count = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        futures = {}
        for algo_name, cfg, seed in tasks:
            f = ex.submit(run_one, algo_name, cfg, seed)
            futures[f] = f"{algo_name}_seed{seed}"

        for f in as_completed(futures):
            done_count += 1
            name = futures[f]
            try:
                msg = f.result()
                elapsed = time.time() - start
                remaining = (len(tasks) - done_count) * elapsed / done_count
                print(f"[{done_count}/{len(tasks)}] {msg}  (~{remaining/60:.0f}min left)",
                      flush=True)
            except Exception as e:
                print(f"[{done_count}/{len(tasks)}] {name} FAIL: {e}", flush=True)

    total = time.time() - start
    print(f"\nAll done in {total/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
