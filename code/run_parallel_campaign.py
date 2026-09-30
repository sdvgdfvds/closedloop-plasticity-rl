import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_MAIN_ALGOS = [
    "PPO",
    "PPO+Cycle",
    "P3O",
    "P3O-dynamic",
    "P3O-dyn+AdaReset",
    "P3O-dyn+HardDistill",
    "P3O-dyn+AdaReset+HardDistill",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill",
    "P3O-ClosedLoopAlpha",
    "P3O-ClosedLoopFull",
]
DEFAULT_MAIN_ENVS_QUICK = ["Hopper-v4"]
DEFAULT_MAIN_ENVS_FORMAL = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
DEFAULT_MAIN_SEEDS_QUICK = [0]
DEFAULT_MAIN_SEEDS_FORMAL = [0, 1, 2]
DEFAULT_SEQUENCE_ALGOS = [
    "PPO",
    "P3O",
    "P3O-ClosedLoopFull",
    "P3O-ClosedLoopMemory",
]
DEFAULT_SEQUENCE_SEEDS_QUICK = [0]
DEFAULT_SEQUENCE_SEEDS_FORMAL = [0, 1, 2]
BASELINE_MAIN_ALGOS = {"PPO", "PPO+Cycle", "P3O"}


@dataclass
class CampaignTask:
    name: str
    category: str
    command: List[str]
    log_path: Path


def sanitize_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def default_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 4))


def build_main_tasks(args, log_dir: Path, python_exe: str) -> List[CampaignTask]:
    algos = args.algos if args.algos else list(DEFAULT_MAIN_ALGOS)
    envs = args.main_envs if args.main_envs else (
        DEFAULT_MAIN_ENVS_QUICK if args.mode == "quick" else DEFAULT_MAIN_ENVS_FORMAL
    )
    seeds = args.main_seeds if args.main_seeds else (
        DEFAULT_MAIN_SEEDS_QUICK if args.mode == "quick" else DEFAULT_MAIN_SEEDS_FORMAL
    )
    tasks = []
    for algo in algos:
        for env in envs:
            for seed in seeds:
                if algo in BASELINE_MAIN_ALGOS:
                    cmd = [python_exe, "run_main_baseline_fill.py", "--algo", algo, "--env", env, "--seed", str(seed)]
                    if args.mode == "quick":
                        quick_steps = args.steps if args.steps is not None else 200000
                        cmd.extend(["--steps", str(quick_steps)])
                    elif args.steps is not None:
                        cmd.extend(["--steps", str(args.steps)])
                    if args.force:
                        cmd.append("--force")
                else:
                    cmd = [python_exe, "run_p3o_extra_innovations.py", "--only-algo", algo, "--env", env, "--seed", str(seed)]
                    if args.mode == "quick":
                        cmd.append("--quick")
                    if args.steps is not None:
                        cmd.extend(["--steps", str(args.steps)])
                    if args.force:
                        cmd.append("--force")
                task_name = f"main_{args.mode}_{algo}_{env}_seed{seed}"
                log_path = log_dir / f"{sanitize_name(task_name)}.log"
                tasks.append(
                    CampaignTask(
                        name=task_name,
                        category="main",
                        command=cmd,
                        log_path=log_path,
                    )
                )
    return tasks


def build_sequence_tasks(args, log_dir: Path, python_exe: str) -> List[CampaignTask]:
    algos = args.sequence_algos if args.sequence_algos else list(DEFAULT_SEQUENCE_ALGOS)
    seeds = (
        args.sequence_seeds
        if args.sequence_seeds
        else (DEFAULT_SEQUENCE_SEEDS_QUICK if args.mode == "quick" else DEFAULT_SEQUENCE_SEEDS_FORMAL)
    )
    tasks = []
    for algo in algos:
        for seed in seeds:
            cmd = [python_exe, "run_sequence_plasticity.py", "--algo", algo, "--seed", str(seed)]
            if args.mode == "quick":
                cmd.append("--quick")
            if args.force:
                cmd.append("--force")
            task_name = f"sequence_{args.mode}_{algo}_seed{seed}"
            log_path = log_dir / f"{sanitize_name(task_name)}.log"
            tasks.append(
                CampaignTask(
                    name=task_name,
                    category="sequence",
                    command=cmd,
                    log_path=log_path,
                )
            )
    return tasks


def build_tasks(args, log_dir: Path, python_exe: str) -> List[CampaignTask]:
    tasks: List[CampaignTask] = []
    if args.campaign in ("all", "main"):
        tasks.extend(build_main_tasks(args, log_dir, python_exe))
    if args.campaign in ("all", "sequence"):
        tasks.extend(build_sequence_tasks(args, log_dir, python_exe))
    return tasks


def write_manifest(tasks: List[CampaignTask], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category", "log_path", "command"])
        for task in tasks:
            writer.writerow(
                [
                    task.name,
                    task.category,
                    str(task.log_path),
                    subprocess.list2cmdline(task.command),
                ]
            )


def run_task(task: CampaignTask):
    start = time.time()
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    with task.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[start] {datetime.now().isoformat()}\n")
        log_file.write(f"[task] {task.name}\n")
        log_file.write(f"[category] {task.category}\n")
        log_file.write(f"[command] {subprocess.list2cmdline(task.command)}\n")
        log_file.write("-" * 80 + "\n")
        log_file.flush()

        completed = subprocess.run(
            task.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
        )

        duration = time.time() - start
        log_file.write("\n" + "-" * 80 + "\n")
        log_file.write(f"[end] {datetime.now().isoformat()}\n")
        log_file.write(f"[returncode] {completed.returncode}\n")
        log_file.write(f"[duration_sec] {duration:.2f}\n")
        log_file.flush()

    return {
        "name": task.name,
        "category": task.category,
        "returncode": completed.returncode,
        "duration_sec": round(duration, 2),
        "log_path": str(task.log_path),
        "command": subprocess.list2cmdline(task.command),
    }


def print_plan(tasks: List[CampaignTask], max_workers: int) -> None:
    print(f"[campaign] tasks={len(tasks)}, max_workers={max_workers}")
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx:02d}] {task.name}")
        print(f"     {subprocess.list2cmdline(task.command)}")
        print(f"     log -> {task.log_path}")


def save_summary(results, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "category", "returncode", "duration_sec", "log_path", "command"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_int_list(values):
    parsed = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                parsed.append(int(piece))
    return parsed


def parse_str_list(values):
    parsed = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                parsed.append(piece)
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel campaign runner for main and sequence experiments."
    )
    parser.add_argument(
        "--campaign",
        choices=["all", "main", "sequence"],
        default="all",
        help="Choose which built-in campaign to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "formal"],
        default="formal",
        help="Quick for smoke runs, formal for full built-in runs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_workers(),
        help="Maximum number of parallel subprocesses.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child processes.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional output directory for logs and manifests.",
    )
    parser.add_argument(
        "--algos",
        nargs="*",
        default=None,
        help="Optional subset of main algorithms to run.",
    )
    parser.add_argument(
        "--main-envs",
        nargs="*",
        default=None,
        help="Optional envs for main experiments.",
    )
    parser.add_argument(
        "--main-seeds",
        nargs="*",
        default=None,
        help="Optional main seeds, e.g. --main-seeds 0 1 2",
    )
    parser.add_argument(
        "--sequence-seeds",
        nargs="*",
        default=None,
        help="Optional sequence seeds, e.g. --sequence-seeds 0 1 2",
    )
    parser.add_argument(
        "--sequence-algos",
        nargs="*",
        default=None,
        help="Optional sequence algorithms to run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional train steps override forwarded to main experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forward --force to child scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only, do not launch subprocesses.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.main_envs:
        args.main_envs = parse_str_list(args.main_envs)
    if args.main_seeds:
        args.main_seeds = parse_int_list(args.main_seeds)
    if args.sequence_seeds:
        args.sequence_seeds = parse_int_list(args.sequence_seeds)
    if args.sequence_algos:
        args.sequence_algos = parse_str_list(args.sequence_algos)
    if args.algos:
        args.algos = parse_str_list(args.algos)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path("campaign_logs") / f"{args.campaign}_{args.mode}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, log_dir, args.python)
    if not tasks:
        print("No tasks were generated.")
        return

    manifest_path = log_dir / "campaign_manifest.csv"
    write_manifest(tasks, manifest_path)
    print_plan(tasks, args.max_workers)
    print(f"[manifest] {manifest_path}")

    if args.dry_run:
        print("[dry-run] no subprocesses launched.")
        return

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
                status = "ok" if result["returncode"] == 0 else f"fail({result['returncode']})"
                print(f"[done] {task.name} -> {status} in {result['duration_sec']:.2f}s")
                print(f"       log -> {result['log_path']}")
                results.append(result)
            except Exception as exc:
                print(f"[error] {task.name} -> {exc}")
                results.append(
                    {
                        "name": task.name,
                        "category": task.category,
                        "returncode": -999,
                        "duration_sec": -1,
                        "log_path": str(task.log_path),
                        "command": subprocess.list2cmdline(task.command),
                    }
                )

    results.sort(key=lambda item: item["name"])
    summary_path = log_dir / "campaign_summary.csv"
    save_summary(results, summary_path)

    ok_count = sum(1 for row in results if row["returncode"] == 0)
    fail_count = len(results) - ok_count
    elapsed = time.time() - start
    print(f"[summary] total={len(results)} ok={ok_count} fail={fail_count} elapsed_sec={elapsed:.2f}")
    print(f"[summary-file] {summary_path}")


if __name__ == "__main__":
    main()
