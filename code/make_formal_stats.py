import argparse
import csv
import math
import os
import random
import re
from collections import Counter, defaultdict


DEFAULT_RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "formal_stats_ready")
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20260323
KNOWN_SEQUENCE_ENVS = ["Humanoid-v4", "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4"]
STANDARD_PAIRWISE_SPECS = [
    ("P3O-dyn+EvtSBP+AdaReset+HardDistill", "P3O-ClosedLoopFull", "final_avg10", "higher"),
    ("P3O-dyn+EvtSBP+AdaReset+HardDistill", "P3O-ClosedLoopFull", "auc_avg10_norm", "higher"),
    ("P3O-dyn+EvtSBP+AdaReset+HardDistill", "P3O-ClosedLoopAlpha", "final_avg10", "higher"),
    ("P3O-dyn+EvtSBP+AdaReset+HardDistill", "P3O-ClosedLoopAlpha", "auc_avg10_norm", "higher"),
]
SEQUENCE_PAIRWISE_SPECS = [
    ("P3O-ClosedLoopFull", "P3O-ClosedLoopMemory", "final_eval_a_reward", "higher"),
    ("P3O-ClosedLoopFull", "P3O-ClosedLoopMemory", "final_retention_a", "higher"),
    ("P3O-ClosedLoopFull", "P3O-ClosedLoopMemory", "final_forgetting_a", "lower"),
]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def format_float(value, digits=3, default="N/A"):
    if value is None:
        return default
    return f"{value:.{digits}f}"


def format_interval(interval, digits=3, default="N/A"):
    if not interval or interval[0] is None or interval[1] is None:
        return default
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def format_mean_std(values, digits=3, default="N/A"):
    values = [v for v in values if v is not None]
    if not values:
        return default
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} +/- {std(values):.{digits}f}"


def bootstrap_ci(values, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED, alpha=0.05):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed + len(values))
    samples = []
    for _ in range(rounds):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(samples)) - 1)
    hi_idx = min(len(samples) - 1, int((1.0 - alpha / 2.0) * len(samples)) - 1)
    return samples[lo_idx], samples[hi_idx]


def paired_sign_flip_pvalue(values_a, values_b):
    paired = [
        (a, b)
        for a, b in zip(values_a, values_b)
        if a is not None and b is not None
    ]
    if not paired:
        return None, 0, None
    deltas = [b - a for a, b in paired]
    n = len(deltas)
    observed = abs(sum(deltas) / n)
    total = 1 << n
    extreme = 0
    for mask in range(total):
        signed = []
        for idx, delta in enumerate(deltas):
            signed.append(delta if ((mask >> idx) & 1) else -delta)
        stat = abs(sum(signed) / n)
        if stat >= observed - 1e-12:
            extreme += 1
    return extreme / total, n, sum(deltas) / n


def trapz_auc(points):
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    if len(pts) < 2:
        return None, None
    pts.sort(key=lambda item: item[0])
    auc = 0.0
    for idx in range(1, len(pts)):
        x0, y0 = pts[idx - 1]
        x1, y1 = pts[idx]
        dx = x1 - x0
        if dx <= 0:
            continue
        auc += 0.5 * (y0 + y1) * dx
    duration = pts[-1][0] - pts[0][0]
    if duration <= 0:
        duration = pts[-1][0] if pts[-1][0] > 0 else None
    auc_per_step = (auc / duration) if duration else None
    return auc, auc_per_step


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


def parse_standard_dir(folder_name):
    match = re.match(r"^(?P<env>[^_]+)_(?P<algo>.+)_seed(?P<seed>[^_]+)_steps(?P<steps>\d+)$", folder_name)
    if not match or folder_name.startswith("sequence_"):
        return None
    info = match.groupdict()
    info["steps"] = int(info["steps"])
    info["run_type"] = "standard"
    return info


def parse_sequence_dir(folder_name):
    match = re.match(r"^sequence_(?P<sequence>.+)_seed(?P<seed>[^_]+)_steps(?P<steps>\d+)$", folder_name)
    if not match:
        return None
    info = match.groupdict()
    info["steps"] = int(info["steps"])
    sequence_blob = info["sequence"]
    algo = "legacy-unlabeled"
    sequence_text = sequence_blob
    env_positions = []
    for env_name in KNOWN_SEQUENCE_ENVS:
        idx = sequence_blob.find(env_name)
        if idx >= 0 and (idx == 0 or sequence_blob[idx - 1] == "_"):
            env_positions.append((idx, env_name))
    if env_positions:
        start_idx, _ = min(env_positions, key=lambda item: item[0])
        if start_idx == 0:
            algo = "legacy-unlabeled"
            sequence_text = sequence_blob
        else:
            algo = sequence_blob[: start_idx - 1]
            sequence_text = sequence_blob[start_idx:]
    info["algo"] = algo
    info["sequence"] = sequence_text.split("_to_")
    info["run_type"] = "sequence"
    return info


def parse_run_dir(folder_name):
    return parse_sequence_dir(folder_name) or parse_standard_dir(folder_name)


def load_standard_summary(metrics_path):
    final_avg10 = None
    best_avg10 = None
    final_episode_reward = None
    best_episode_reward = None
    last_step = None
    reward_points = []
    alpha_values = []
    reset_values = []
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
                    final_episode_reward = reward_value
                    best_episode_reward = reward_value if best_episode_reward is None else max(best_episode_reward, reward_value)
                if avg10_value is not None:
                    final_avg10 = avg10_value
                    best_avg10 = avg10_value if best_avg10 is None else max(best_avg10, avg10_value)
                if step is not None and avg10_value is not None:
                    reward_points.append((step, avg10_value))
            elif tag == "speed":
                sps_value = safe_float(row[2]) if len(row) > 2 else None
                if sps_value is not None:
                    speed_values.append(sps_value)
            elif tag == "sbp":
                total_sbp_rows += 1
                alpha_value = safe_float(row[4]) if len(row) > 4 else None
                reset_value = safe_float(row[8]) if len(row) > 8 else None
                trigger_mode = row[24].strip() if len(row) > 24 and row[24].strip() else "periodic"
                if alpha_value is not None:
                    alpha_values.append(alpha_value)
                if reset_value is not None:
                    reset_values.append(reset_value)
                trigger_counts[trigger_mode] += 1

    auc_raw, auc_norm = trapz_auc(reward_points)
    return {
        "final_avg10": final_avg10,
        "best_avg10": best_avg10,
        "final_episode_reward": final_episode_reward,
        "best_episode_reward": best_episode_reward,
        "auc_avg10_raw": auc_raw,
        "auc_avg10_norm": auc_norm,
        "mean_reset_rate": mean(reset_values),
        "mean_alpha_using": mean(alpha_values),
        "mean_sps": mean(speed_values),
        "sbp_trigger_count": total_sbp_rows,
        "event_trigger_count": trigger_counts.get("event", 0),
        "force_trigger_count": trigger_counts.get("force", 0),
        "periodic_trigger_count": trigger_counts.get("periodic", 0),
        "other_trigger_count": sum(count for key, count in trigger_counts.items() if key not in {"event", "force", "periodic"}),
        "last_logged_step": int(last_step) if last_step is not None else None,
        "reward_points": len(reward_points),
    }


def load_sequence_summary(metrics_path, eval_a_env=None):
    phase_summaries = []
    eval_points = []
    sbp_trigger_count = 0
    trigger_counts = Counter()

    with open(metrics_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2 or (row[0] == "step" and row[1] == "tag"):
                continue
            tag = row[1].strip()
            if tag == "phase_summary" and len(row) >= 9:
                phase_summaries.append(
                    {
                        "step": safe_float(row[0]),
                        "phase": safe_float(row[2]),
                        "train_env": row[3],
                        "eval_a_reward": safe_float(row[4]),
                        "eval_a_reference": safe_float(row[5]),
                        "retention_a": safe_float(row[6]),
                        "forgetting_a": safe_float(row[7]),
                        "elapsed_min": safe_float(row[8]),
                    }
                )
            elif tag == "eval" and len(row) >= 8 and (eval_a_env is None or row[4] == eval_a_env):
                eval_points.append((safe_float(row[0]), safe_float(row[5])))
            elif tag == "sbp":
                sbp_trigger_count += 1
                mode = row[4].strip() if len(row) > 4 and row[4].strip() else "unknown"
                trigger_counts[mode] += 1

    auc_raw, auc_norm = trapz_auc(eval_points)
    last_phase = phase_summaries[-1] if phase_summaries else {}
    return {
        "final_eval_a_reward": last_phase.get("eval_a_reward"),
        "final_retention_a": last_phase.get("retention_a"),
        "final_forgetting_a": last_phase.get("forgetting_a"),
        "auc_eval_reward_raw": auc_raw,
        "auc_eval_reward_norm": auc_norm,
        "sbp_trigger_count": sbp_trigger_count,
        "event_trigger_count": trigger_counts.get("event", 0),
        "force_trigger_count": trigger_counts.get("force", 0),
        "periodic_trigger_count": trigger_counts.get("periodic", 0),
        "phase_count": len(phase_summaries),
    }


def discover_runs(runs_dir):
    standard_runs = []
    sequence_runs = []
    if not os.path.isdir(runs_dir):
        return standard_runs, sequence_runs

    for entry in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        parsed = parse_run_dir(entry)
        if not parsed:
            continue
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        if parsed["run_type"] == "standard":
            run = dict(parsed)
            run["run_dir_name"] = entry
            run["run_dir"] = run_dir
            run["group"] = infer_experiment_group(run["algo"])
            run.update(load_standard_summary(metrics_path))
            standard_runs.append(run)
        else:
            run = dict(parsed)
            run["run_dir_name"] = entry
            run["run_dir"] = run_dir
            eval_a_env = run["sequence"][0] if run.get("sequence") else None
            run.update(load_sequence_summary(metrics_path, eval_a_env=eval_a_env))
            sequence_runs.append(run)

    standard_runs.sort(key=lambda item: (item["group"], item["env"], item["algo"], item["steps"], str(item["seed"])))
    sequence_runs.sort(key=lambda item: (item["steps"], str(item["seed"]), item["run_dir_name"]))
    return standard_runs, sequence_runs


def build_standard_run_rows(runs):
    rows = []
    for index, run in enumerate(runs, start=1):
        rows.append(
            {
                "run_index": f"R{index:03d}",
                "group": run["group"],
                "env": run["env"],
                "algo": run["algo"],
                "seed": run["seed"],
                "steps": run["steps"],
                "final_avg10": format_float(run["final_avg10"]),
                "best_avg10": format_float(run["best_avg10"]),
                "auc_avg10_raw": format_float(run["auc_avg10_raw"], digits=2),
                "auc_avg10_norm": format_float(run["auc_avg10_norm"], digits=3),
                "mean_reset_rate": format_float(run["mean_reset_rate"], digits=5),
                "mean_alpha_using": format_float(run["mean_alpha_using"], digits=5),
                "sbp_trigger_count": run["sbp_trigger_count"],
                "event_trigger_count": run["event_trigger_count"],
                "force_trigger_count": run["force_trigger_count"],
                "periodic_trigger_count": run["periodic_trigger_count"],
                "mean_sps": format_float(run["mean_sps"], digits=2),
                "last_logged_step": run["last_logged_step"] if run["last_logged_step"] is not None else "N/A",
                "reward_points": run["reward_points"],
                "run_dir_name": run["run_dir_name"],
            }
        )
    return rows


def build_standard_aggregate_rows(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = (run["group"], run["env"], run["algo"], run["steps"])
        grouped[key].append(run)

    rows = []
    for (group, env, algo, steps), items in sorted(grouped.items()):
        final_vals = [item["final_avg10"] for item in items]
        best_vals = [item["best_avg10"] for item in items]
        auc_vals = [item["auc_avg10_norm"] for item in items]
        reset_vals = [item["mean_reset_rate"] for item in items]
        alpha_vals = [item["mean_alpha_using"] for item in items]
        sps_vals = [item["mean_sps"] for item in items]
        trigger_vals = [item["sbp_trigger_count"] for item in items]
        rows.append(
            {
                "group": group,
                "env": env,
                "algo": algo,
                "steps": steps,
                "num_runs": len(items),
                "seeds": ",".join(sorted(str(item["seed"]) for item in items)),
                "final_avg10_mean_std": format_mean_std(final_vals, digits=3),
                "final_avg10_bootstrap95": format_interval(bootstrap_ci(final_vals), digits=3),
                "best_avg10_mean_std": format_mean_std(best_vals, digits=3),
                "best_avg10_bootstrap95": format_interval(bootstrap_ci(best_vals), digits=3),
                "auc_avg10_norm_mean_std": format_mean_std(auc_vals, digits=3),
                "auc_avg10_norm_bootstrap95": format_interval(bootstrap_ci(auc_vals), digits=3),
                "mean_reset_rate_mean_std": format_mean_std(reset_vals, digits=5),
                "mean_alpha_using_mean_std": format_mean_std(alpha_vals, digits=5),
                "mean_sps_mean_std": format_mean_std(sps_vals, digits=2),
                "sbp_trigger_count_mean_std": format_mean_std(trigger_vals, digits=2),
                "event_trigger_total": sum(item["event_trigger_count"] for item in items),
                "force_trigger_total": sum(item["force_trigger_count"] for item in items),
                "periodic_trigger_total": sum(item["periodic_trigger_count"] for item in items),
            }
        )
    return rows


def build_sequence_rows(runs):
    rows = []
    for index, run in enumerate(runs, start=1):
        rows.append(
            {
                "sequence_index": f"S{index:03d}",
                "algo": run["algo"],
                "sequence": " -> ".join(run["sequence"]),
                "seed": run["seed"],
                "steps_per_phase": run["steps"],
                "phase_count": run["phase_count"],
                "final_eval_a_reward": format_float(run["final_eval_a_reward"]),
                "final_retention_a": format_float(run["final_retention_a"], digits=6),
                "final_forgetting_a": format_float(run["final_forgetting_a"]),
                "auc_eval_reward_norm": format_float(run["auc_eval_reward_norm"], digits=3),
                "sbp_trigger_count": run["sbp_trigger_count"],
                "event_trigger_count": run["event_trigger_count"],
                "force_trigger_count": run["force_trigger_count"],
                "periodic_trigger_count": run["periodic_trigger_count"],
                "run_dir_name": run["run_dir_name"],
            }
        )
    return rows


def build_sequence_aggregate_rows(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = (run["algo"], tuple(run["sequence"]), run["steps"])
        grouped[key].append(run)

    rows = []
    for (algo, sequence, steps), items in sorted(grouped.items()):
        final_vals = [item["final_eval_a_reward"] for item in items]
        retention_vals = [item["final_retention_a"] for item in items]
        forgetting_vals = [item["final_forgetting_a"] for item in items]
        auc_vals = [item["auc_eval_reward_norm"] for item in items]
        trigger_vals = [item["sbp_trigger_count"] for item in items]
        rows.append(
            {
                "algo": algo,
                "sequence": " -> ".join(sequence),
                "steps_per_phase": steps,
                "num_runs": len(items),
                "seeds": ",".join(sorted(str(item["seed"]) for item in items)),
                "final_eval_a_reward_mean_std": format_mean_std(final_vals, digits=3),
                "final_eval_a_reward_bootstrap95": format_interval(bootstrap_ci(final_vals), digits=3),
                "final_retention_a_mean_std": format_mean_std(retention_vals, digits=6),
                "final_retention_a_bootstrap95": format_interval(bootstrap_ci(retention_vals), digits=6),
                "final_forgetting_a_mean_std": format_mean_std(forgetting_vals, digits=3),
                "final_forgetting_a_bootstrap95": format_interval(bootstrap_ci(forgetting_vals), digits=3),
                "auc_eval_reward_norm_mean_std": format_mean_std(auc_vals, digits=3),
                "auc_eval_reward_norm_bootstrap95": format_interval(bootstrap_ci(auc_vals), digits=3),
                "sbp_trigger_count_mean_std": format_mean_std(trigger_vals, digits=2),
                "event_trigger_total": sum(item["event_trigger_count"] for item in items),
                "force_trigger_total": sum(item["force_trigger_count"] for item in items),
                "periodic_trigger_total": sum(item["periodic_trigger_count"] for item in items),
            }
        )
    return rows


def build_pairwise_rows(standard_runs, sequence_runs):
    rows = []

    def add_standard_rows(reference_algo, candidate_algo, metric, better_direction):
        grouped = defaultdict(dict)
        for run in standard_runs:
            key = (run["env"], run["steps"], run["seed"])
            grouped[key][run["algo"]] = run
        bucketed = defaultdict(list)
        for (env, steps, seed), algo_runs in grouped.items():
            ref = algo_runs.get(reference_algo)
            cand = algo_runs.get(candidate_algo)
            if ref and cand:
                bucketed[(env, steps)].append((seed, ref.get(metric), cand.get(metric)))
        for (env, steps), pairs in sorted(bucketed.items()):
            ref_values = [pair[1] for pair in pairs]
            cand_values = [pair[2] for pair in pairs]
            p_value, pair_count, mean_delta = paired_sign_flip_pvalue(ref_values, cand_values)
            if pair_count < 2:
                continue
            if mean_delta is None:
                better_algo = "N/A"
            elif better_direction == "higher":
                better_algo = candidate_algo if mean_delta > 0 else reference_algo
            else:
                better_algo = candidate_algo if mean_delta < 0 else reference_algo
            rows.append(
                {
                    "scope": "standard",
                    "context": env,
                    "steps": steps,
                    "metric": metric,
                    "reference_algo": reference_algo,
                    "candidate_algo": candidate_algo,
                    "pairs": pair_count,
                    "reference_mean": format_float(mean(ref_values)),
                    "candidate_mean": format_float(mean(cand_values)),
                    "mean_delta_candidate_minus_reference": format_float(mean_delta),
                    "paired_signflip_pvalue": format_float(p_value, digits=4),
                    "better_algo": better_algo,
                }
            )

    def add_sequence_rows(reference_algo, candidate_algo, metric, better_direction):
        grouped = defaultdict(dict)
        for run in sequence_runs:
            key = (" -> ".join(run["sequence"]), run["steps"], run["seed"])
            grouped[key][run["algo"]] = run
        bucketed = defaultdict(list)
        for (sequence_name, steps, seed), algo_runs in grouped.items():
            ref = algo_runs.get(reference_algo)
            cand = algo_runs.get(candidate_algo)
            if ref and cand:
                bucketed[(sequence_name, steps)].append((seed, ref.get(metric), cand.get(metric)))
        for (sequence_name, steps), pairs in sorted(bucketed.items()):
            ref_values = [pair[1] for pair in pairs]
            cand_values = [pair[2] for pair in pairs]
            p_value, pair_count, mean_delta = paired_sign_flip_pvalue(ref_values, cand_values)
            if pair_count < 2:
                continue
            if mean_delta is None:
                better_algo = "N/A"
            elif better_direction == "higher":
                better_algo = candidate_algo if mean_delta > 0 else reference_algo
            else:
                better_algo = candidate_algo if mean_delta < 0 else reference_algo
            rows.append(
                {
                    "scope": "sequence",
                    "context": sequence_name,
                    "steps": steps,
                    "metric": metric,
                    "reference_algo": reference_algo,
                    "candidate_algo": candidate_algo,
                    "pairs": pair_count,
                    "reference_mean": format_float(mean(ref_values)),
                    "candidate_mean": format_float(mean(cand_values)),
                    "mean_delta_candidate_minus_reference": format_float(mean_delta),
                    "paired_signflip_pvalue": format_float(p_value, digits=4),
                    "better_algo": better_algo,
                }
            )

    for reference_algo, candidate_algo, metric, direction in STANDARD_PAIRWISE_SPECS:
        add_standard_rows(reference_algo, candidate_algo, metric, better_direction=direction)
    for reference_algo, candidate_algo, metric, direction in SEQUENCE_PAIRWISE_SPECS:
        add_sequence_rows(reference_algo, candidate_algo, metric, better_direction=direction)
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, title, rows, fieldnames):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"# {title}\n\n")
        if not rows:
            file.write("No rows found.\n")
            return
        file.write("| " + " | ".join(fieldnames) + " |\n")
        file.write("|" + " --- |" * len(fieldnames) + "\n")
        for row in rows:
            file.write("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |\n")


def write_overview(path, standard_runs, sequence_runs, aggregate_rows):
    group_counter = Counter(run["group"] for run in standard_runs)
    env_counter = Counter(run["env"] for run in standard_runs)
    best_by_env = {}
    for row in aggregate_rows:
        score_text = row["final_avg10_mean_std"]
        score = safe_float(score_text.split("+/-")[0].strip())
        if score is None:
            continue
        current = best_by_env.get(row["env"])
        if current is None or score > current[0]:
            best_by_env[row["env"]] = (score, row["algo"], row["group"])

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Formal Statistics Overview\n\n")
        file.write(f"- Standard runs indexed: {len(standard_runs)}\n")
        file.write(f"- Sequence runs indexed: {len(sequence_runs)}\n")
        file.write(f"- Experiment groups: {', '.join(f'{k}={v}' for k, v in sorted(group_counter.items()))}\n")
        file.write(f"- Environments: {', '.join(f'{k}={v}' for k, v in sorted(env_counter.items()))}\n\n")
        file.write("## Best Final Avg10 By Environment\n\n")
        if not best_by_env:
            file.write("No valid aggregate rows found.\n")
        else:
            for env, (score, algo, group) in sorted(best_by_env.items()):
                file.write(f"- {env}: `{algo}` ({group}), final avg10 mean = {score:.3f}\n")
        file.write("\n## Sequence Coverage\n\n")
        if not sequence_runs:
            file.write("No sequence runs found.\n")
        else:
            coverage = Counter((" -> ".join(run["sequence"]), run["algo"], run["steps"]) for run in sequence_runs)
            for (sequence_name, algo, steps), count in sorted(coverage.items()):
                file.write(f"- {sequence_name} | algo=`{algo}` | steps/phase={steps} | runs={count}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate formal paper-style statistics from p3o_runs.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory containing run folders.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for formal statistics outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    standard_runs, sequence_runs = discover_runs(args.runs_dir)
    standard_run_rows = build_standard_run_rows(standard_runs)
    standard_aggregate_rows = build_standard_aggregate_rows(standard_runs)
    sequence_rows = build_sequence_rows(sequence_runs)
    sequence_aggregate_rows = build_sequence_aggregate_rows(sequence_runs)
    pairwise_rows = build_pairwise_rows(standard_runs, sequence_runs)

    standard_run_fields = [
        "run_index", "group", "env", "algo", "seed", "steps", "final_avg10", "best_avg10",
        "auc_avg10_raw", "auc_avg10_norm", "mean_reset_rate", "mean_alpha_using", "sbp_trigger_count",
        "event_trigger_count", "force_trigger_count", "periodic_trigger_count", "mean_sps", "last_logged_step",
        "reward_points", "run_dir_name",
    ]
    standard_aggregate_fields = [
        "group", "env", "algo", "steps", "num_runs", "seeds", "final_avg10_mean_std",
        "final_avg10_bootstrap95", "best_avg10_mean_std", "best_avg10_bootstrap95",
        "auc_avg10_norm_mean_std", "auc_avg10_norm_bootstrap95", "mean_reset_rate_mean_std",
        "mean_alpha_using_mean_std", "mean_sps_mean_std", "sbp_trigger_count_mean_std",
        "event_trigger_total", "force_trigger_total", "periodic_trigger_total",
    ]
    sequence_fields = [
        "sequence_index", "algo", "sequence", "seed", "steps_per_phase", "phase_count", "final_eval_a_reward",
        "final_retention_a", "final_forgetting_a", "auc_eval_reward_norm", "sbp_trigger_count",
        "event_trigger_count", "force_trigger_count", "periodic_trigger_count", "run_dir_name",
    ]
    sequence_aggregate_fields = [
        "algo", "sequence", "steps_per_phase", "num_runs", "seeds",
        "final_eval_a_reward_mean_std", "final_eval_a_reward_bootstrap95",
        "final_retention_a_mean_std", "final_retention_a_bootstrap95",
        "final_forgetting_a_mean_std", "final_forgetting_a_bootstrap95",
        "auc_eval_reward_norm_mean_std", "auc_eval_reward_norm_bootstrap95",
        "sbp_trigger_count_mean_std", "event_trigger_total", "force_trigger_total", "periodic_trigger_total",
    ]
    pairwise_fields = [
        "scope", "context", "steps", "metric", "reference_algo", "candidate_algo", "pairs",
        "reference_mean", "candidate_mean", "mean_delta_candidate_minus_reference",
        "paired_signflip_pvalue", "better_algo",
    ]

    outputs = [
        ("formal_standard_run_stats.csv", standard_run_rows, standard_run_fields),
        ("formal_standard_aggregate_stats.csv", standard_aggregate_rows, standard_aggregate_fields),
        ("formal_sequence_stats.csv", sequence_rows, sequence_fields),
        ("formal_sequence_aggregate_stats.csv", sequence_aggregate_rows, sequence_aggregate_fields),
        ("formal_pairwise_tests.csv", pairwise_rows, pairwise_fields),
    ]
    for filename, rows, fields in outputs:
        write_csv(os.path.join(args.out_dir, filename), rows, fields)
        write_markdown(os.path.join(args.out_dir, filename.replace(".csv", ".md")), filename.replace("_", " ").replace(".csv", "").title(), rows, fields)

    write_overview(
        os.path.join(args.out_dir, "formal_stats_overview.md"),
        standard_runs,
        sequence_runs,
        standard_aggregate_rows,
    )

    print(f"[saved] {os.path.join(args.out_dir, 'formal_standard_run_stats.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'formal_standard_aggregate_stats.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'formal_sequence_stats.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'formal_sequence_aggregate_stats.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'formal_pairwise_tests.csv')}")
    print(f"[saved] {os.path.join(args.out_dir, 'formal_stats_overview.md')}")


if __name__ == "__main__":
    main()
