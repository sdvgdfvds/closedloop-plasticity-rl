import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict


DEFAULT_RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "breakthrough_ready")


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_run_dir_name(folder_name):
    match = re.match(r"^(?P<env>[^_]+)_(?P<algo>.+)_seed(?P<seed>[^_]+)_steps(?P<steps>\d+)$", folder_name)
    if not match:
        return None
    info = match.groupdict()
    info["steps"] = int(info["steps"])
    return info


def infer_experiment_group(algo_name):
    if algo_name in {"PPO", "PPO+Cycle", "P3O"}:
        return "main"
    if "ClosedLoop" in algo_name:
        return "closed_loop_breakthrough"
    if "EvtSBP" in algo_name:
        return "event_mechanism"
    if algo_name in {"P3O-static", "P3O-dynamic"}:
        return "alpha_ablation"
    if "AdaReset" in algo_name or "HardDistill" in algo_name:
        return "extra_innovation"
    return "other"


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def format_float(value, digits=3, default="N/A"):
    if value is None:
        return default
    return f"{value:.{digits}f}"


def format_mean_std(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} +/- {std(values):.{digits}f}"


def load_metrics_summary(metrics_path):
    final_avg10 = None
    best_avg10 = None
    final_reward = None
    best_reward = None
    last_step = None
    mean_alpha_using_values = []
    mean_reset_rate_values = []
    speed_values = []
    trigger_counts = Counter()
    total_sbp_rows = 0

    with open(metrics_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2 or (row[0] == "step" and row[1] == "tag"):
                continue

            step = safe_float(row[0])
            if step is not None:
                last_step = step

            tag = row[1].strip()
            if tag == "reward":
                reward_value = safe_float(row[3]) if len(row) > 3 else None
                avg10_value = safe_float(row[4]) if len(row) > 4 else None
                if reward_value is not None:
                    final_reward = reward_value
                    best_reward = reward_value if best_reward is None else max(best_reward, reward_value)
                if avg10_value is not None:
                    final_avg10 = avg10_value
                    best_avg10 = avg10_value if best_avg10 is None else max(best_avg10, avg10_value)
            elif tag == "speed":
                sps_value = safe_float(row[2]) if len(row) > 2 else None
                if sps_value is not None:
                    speed_values.append(sps_value)
            elif tag == "sbp":
                total_sbp_rows += 1
                alpha_using = safe_float(row[4]) if len(row) > 4 else None
                reset_rate = safe_float(row[8]) if len(row) > 8 else None
                trigger_mode = row[24].strip() if len(row) > 24 and row[24].strip() else "periodic"
                if alpha_using is not None:
                    mean_alpha_using_values.append(alpha_using)
                if reset_rate is not None:
                    mean_reset_rate_values.append(reset_rate)
                trigger_counts[trigger_mode] += 1

    return {
        "final_avg10_reward": final_avg10,
        "best_avg10_reward": best_avg10,
        "final_episode_reward": final_reward,
        "best_episode_reward": best_reward,
        "mean_alpha_using": mean(mean_alpha_using_values),
        "mean_reset_rate": mean(mean_reset_rate_values),
        "sbp_trigger_count": total_sbp_rows,
        "event_trigger_count": trigger_counts.get("event", 0),
        "force_trigger_count": trigger_counts.get("force", 0),
        "periodic_trigger_count": trigger_counts.get("periodic", 0),
        "other_trigger_count": sum(count for key, count in trigger_counts.items() if key not in {"event", "force", "periodic"}),
        "mean_sps": mean(speed_values),
        "last_logged_step": int(last_step) if last_step is not None else None,
    }


def discover_runs(runs_dir):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs

    for entry in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        parsed = parse_run_dir_name(entry)
        if not parsed:
            continue

        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        run = dict(parsed)
        run["run_dir_name"] = entry
        run["run_dir"] = run_dir
        run["experiment_group"] = infer_experiment_group(run["algo"])
        run.update(load_metrics_summary(metrics_path))
        runs.append(run)

    runs.sort(key=lambda item: (item["experiment_group"], item["env"], item["algo"], int(item["seed"]), item["steps"]))
    for index, run in enumerate(runs, start=1):
        run["run_index"] = f"R{index:03d}"
    return runs


def build_run_rows(runs):
    rows = []
    for run in runs:
        rows.append(
            {
                "run_index": run["run_index"],
                "group": run["experiment_group"],
                "env": run["env"],
                "algo": run["algo"],
                "seed": run["seed"],
                "steps": run["steps"],
                "final_avg10_reward": format_float(run["final_avg10_reward"]),
                "best_avg10_reward": format_float(run["best_avg10_reward"]),
                "mean_reset_rate": format_float(run["mean_reset_rate"], digits=5),
                "mean_alpha_using": format_float(run["mean_alpha_using"], digits=5),
                "sbp_trigger_count": run["sbp_trigger_count"],
                "event_trigger_count": run["event_trigger_count"],
                "force_trigger_count": run["force_trigger_count"],
                "periodic_trigger_count": run["periodic_trigger_count"],
                "mean_sps": format_float(run["mean_sps"], digits=2),
                "last_logged_step": run["last_logged_step"] if run["last_logged_step"] is not None else "N/A",
                "run_dir_name": run["run_dir_name"],
            }
        )
    return rows


def build_aggregate_rows(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = (run["experiment_group"], run["env"], run["algo"], run["steps"])
        grouped[key].append(run)

    rows = []
    for (group, env, algo, steps), items in sorted(grouped.items()):
        rows.append(
            {
                "group": group,
                "env": env,
                "algo": algo,
                "steps": steps,
                "num_runs": len(items),
                "seeds": ",".join(sorted(str(item["seed"]) for item in items)),
                "final_avg10_reward": format_mean_std([item["final_avg10_reward"] for item in items]),
                "best_avg10_reward": format_mean_std([item["best_avg10_reward"] for item in items]),
                "mean_reset_rate": format_mean_std([item["mean_reset_rate"] for item in items], digits=5),
                "mean_alpha_using": format_mean_std([item["mean_alpha_using"] for item in items], digits=5),
                "mean_sps": format_mean_std([item["mean_sps"] for item in items], digits=2),
                "avg_sbp_trigger_count": format_mean_std([item["sbp_trigger_count"] for item in items], digits=2),
                "event_trigger_total": sum(item["event_trigger_count"] for item in items),
                "force_trigger_total": sum(item["force_trigger_count"] for item in items),
                "periodic_trigger_total": sum(item["periodic_trigger_count"] for item in items),
                "run_indices": ",".join(item["run_index"] for item in items),
            }
        )
    return rows


def build_index_rows(runs):
    return [
        {
            "run_index": run["run_index"],
            "group": run["experiment_group"],
            "env": run["env"],
            "algo": run["algo"],
            "seed": run["seed"],
            "steps": run["steps"],
            "run_dir": run["run_dir"],
        }
        for run in runs
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, title, rows, columns):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"# {title}\n\n")
        if not rows:
            file.write("No runs found.\n")
            return

        headers = [label for _, label in columns]
        keys = [key for key, _ in columns]
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("|" + " --- |" * len(headers) + "\n")
        for row in rows:
            file.write("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n")


def write_overview(path, runs, aggregate_rows):
    group_counter = Counter(run["experiment_group"] for run in runs)
    env_counter = Counter(run["env"] for run in runs)
    best_by_env = {}

    for row in aggregate_rows:
        env = row["env"]
        score_text = row["final_avg10_reward"]
        if score_text == "N/A":
            continue
        score = safe_float(score_text.split("+/-")[0].strip())
        if score is None:
            continue
        current = best_by_env.get(env)
        if current is None or score > current[0]:
            best_by_env[env] = (score, row["algo"], row["group"])

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Breakthrough Summary Overview\n\n")
        file.write(f"- Total runs indexed: {len(runs)}\n")
        file.write(f"- Experiment groups: {', '.join(f'{k}={v}' for k, v in sorted(group_counter.items()))}\n")
        file.write(f"- Environments covered: {', '.join(f'{k}={v}' for k, v in sorted(env_counter.items()))}\n\n")
        file.write("## Best Final Avg10 Reward By Environment\n\n")
        if not best_by_env:
            file.write("No valid reward runs found.\n")
        else:
            for env, (score, algo, group) in sorted(best_by_env.items()):
                file.write(f"- {env}: `{algo}` ({group}), final avg10 = {score:.3f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize breakthrough experiment runs into CSV and Markdown tables.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory that stores experiment run folders.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for generated summaries.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = discover_runs(args.runs_dir)
    run_rows = build_run_rows(runs)
    aggregate_rows = build_aggregate_rows(runs)
    index_rows = build_index_rows(runs)

    run_fields = [
        "run_index",
        "group",
        "env",
        "algo",
        "seed",
        "steps",
        "final_avg10_reward",
        "best_avg10_reward",
        "mean_reset_rate",
        "mean_alpha_using",
        "sbp_trigger_count",
        "event_trigger_count",
        "force_trigger_count",
        "periodic_trigger_count",
        "mean_sps",
        "last_logged_step",
        "run_dir_name",
    ]
    aggregate_fields = [
        "group",
        "env",
        "algo",
        "steps",
        "num_runs",
        "seeds",
        "final_avg10_reward",
        "best_avg10_reward",
        "mean_reset_rate",
        "mean_alpha_using",
        "mean_sps",
        "avg_sbp_trigger_count",
        "event_trigger_total",
        "force_trigger_total",
        "periodic_trigger_total",
        "run_indices",
    ]
    index_fields = ["run_index", "group", "env", "algo", "seed", "steps", "run_dir"]

    write_csv(os.path.join(args.out_dir, "breakthrough_run_summary.csv"), run_rows, run_fields)
    write_csv(os.path.join(args.out_dir, "breakthrough_aggregate_summary.csv"), aggregate_rows, aggregate_fields)
    write_csv(os.path.join(args.out_dir, "breakthrough_run_index.csv"), index_rows, index_fields)

    write_markdown(
        os.path.join(args.out_dir, "breakthrough_run_summary.md"),
        "Breakthrough Run Summary",
        run_rows,
        [(field, field) for field in run_fields],
    )
    write_markdown(
        os.path.join(args.out_dir, "breakthrough_aggregate_summary.md"),
        "Breakthrough Aggregate Summary",
        aggregate_rows,
        [(field, field) for field in aggregate_fields],
    )
    write_markdown(
        os.path.join(args.out_dir, "breakthrough_run_index.md"),
        "Breakthrough Run Index",
        index_rows,
        [(field, field) for field in index_fields],
    )
    write_overview(os.path.join(args.out_dir, "breakthrough_overview.md"), runs, aggregate_rows)

    print(f"[saved] {os.path.join(args.out_dir, 'breakthrough_run_summary.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'breakthrough_aggregate_summary.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'breakthrough_run_index.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'breakthrough_overview.md')}")


if __name__ == "__main__":
    main()
