import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
OUT_DIR = os.path.join(os.getcwd(), "sequence_ready")
PREFIX = "sequence_"
ENV_NAMES = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]
ALGO_ORDER = ["PPO", "P3O", "P3O-ClosedLoopFull", "P3O-ClosedLoopMemory"]
ALGO_LABELS = {
    "PPO": "PPO",
    "P3O": "Stay Hungry / P3O",
    "P3O-ClosedLoopFull": "ClosedLoop-Full",
    "P3O-ClosedLoopMemory": "ClosedLoop-Memory",
    "legacy": "Legacy",
}
SEQUENCE_LABELS = {
    "Hopper-v4 -> Walker2d-v4 -> Hopper-v4": "Seq1: Hopper -> Walker2d -> Hopper",
    "Walker2d-v4 -> HalfCheetah-v4 -> Walker2d-v4": "Seq2: Walker2d -> HalfCheetah -> Walker2d",
}
COLORS = {
    "PPO": "#4E79A7",
    "P3O": "#9C755F",
    "P3O-ClosedLoopFull": "#17BECF",
    "P3O-ClosedLoopMemory": "#E15759",
    "legacy": "#7F7F7F",
}


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return mu, math.sqrt(var)


def parse_sequence_dir(folder):
    if not folder.startswith(PREFIX) or "_seed" not in folder or "_steps" not in folder:
        return None
    try:
        tail = folder[len(PREFIX) :]
        seq_part, rest = tail.rsplit("_seed", 1)
        seed, steps = rest.split("_steps", 1)

        first_env_idx = None
        first_env = None
        for env_name in ENV_NAMES:
            marker = env_name + "_to_"
            idx = seq_part.find(marker)
            if idx >= 0 and (first_env_idx is None or idx < first_env_idx):
                first_env_idx = idx
                first_env = env_name
            if seq_part == env_name and first_env_idx is None:
                first_env_idx = 0
                first_env = env_name

        if first_env_idx is None:
            return None

        algo = seq_part[:first_env_idx].rstrip("_") or "legacy"
        seq_text = seq_part[first_env_idx:] if first_env is not None else seq_part
        sequence = seq_text.split("_to_")
        return {
            "sequence": sequence,
            "sequence_key": " -> ".join(sequence),
            "seed": int(seed),
            "steps": int(steps),
            "algo": algo,
        }
    except Exception:
        return None


def load_runs():
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs

    for entry in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(run_dir):
            continue
        meta = parse_sequence_dir(entry)
        if not meta:
            continue
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        rows = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2 or (row[0] == "step" and row[1] == "tag"):
                    continue
                rows.append(row)
        if rows:
            runs.append({"dir": run_dir, "name": entry, "meta": meta, "rows": rows})
    return runs


def select_runs_for_plots(runs):
    if not runs:
        return []

    best_step_by_sequence = defaultdict(int)
    for run in runs:
        seq_key = run["meta"]["sequence_key"]
        best_step_by_sequence[seq_key] = max(best_step_by_sequence[seq_key], run["meta"]["steps"])

    selected = []
    for run in runs:
        seq_key = run["meta"]["sequence_key"]
        if run["meta"]["steps"] == best_step_by_sequence[seq_key]:
            selected.append(run)

    selected.sort(
        key=lambda r: (
            tuple(r["meta"]["sequence"]),
            ALGO_ORDER.index(r["meta"]["algo"]) if r["meta"]["algo"] in ALGO_ORDER else 999,
            r["meta"]["seed"],
        )
    )
    return selected


def extract_phase_boundaries(rows):
    bounds = []
    for row in rows:
        if row[1] != "phase_start" or len(row) < 5:
            continue
        bounds.append(
            {
                "step": safe_float(row[0]) or 0.0,
                "phase": int(float(row[2])),
                "env": row[3],
                "phase_steps": safe_float(row[4]) or 0.0,
            }
        )
    return bounds


def extract_eval_series(rows):
    series = defaultdict(list)
    for row in rows:
        if row[1] != "eval" or len(row) < 8:
            continue
        step = safe_float(row[0])
        eval_env = row[4]
        reward = safe_float(row[5])
        retention = safe_float(row[6])
        forgetting = safe_float(row[7])
        if step is None or reward is None:
            continue
        series[eval_env].append(
            {
                "step": step,
                "reward": reward,
                "retention": retention,
                "forgetting": forgetting,
                "phase": int(float(row[2])),
                "train_env": row[3],
            }
        )
    return series


def extract_phase_summaries(rows):
    items = []
    for row in rows:
        if row[1] != "phase_summary" or len(row) < 9:
            continue
        items.append(
            {
                "step": safe_float(row[0]),
                "phase": int(float(row[2])),
                "train_env": row[3],
                "eval_a_reward": safe_float(row[4]),
                "eval_a_ref": safe_float(row[5]),
                "retention_a": safe_float(row[6]),
                "forgetting_a": safe_float(row[7]),
                "elapsed_min": safe_float(row[8]),
            }
        )
    return items


def build_run_records(selected_runs):
    records = []
    for run in selected_runs:
        phase_bounds = extract_phase_boundaries(run["rows"])
        eval_series = extract_eval_series(run["rows"])
        phase_summaries = extract_phase_summaries(run["rows"])
        records.append(
            {
                "dir": run["dir"],
                "name": run["name"],
                "meta": run["meta"],
                "phase_bounds": phase_bounds,
                "eval_series": eval_series,
                "phase_summaries": phase_summaries,
            }
        )
    return records


def save_summary_csv(run_records):
    out_path = os.path.join(OUT_DIR, "sequence_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_dir",
                "algo",
                "sequence",
                "steps_per_phase",
                "seed",
                "phase",
                "train_env",
                "eval_a_reward",
                "eval_a_reference",
                "retention_a",
                "forgetting_a",
                "elapsed_min",
            ]
        )
        for run in run_records:
            meta = run["meta"]
            for item in run["phase_summaries"]:
                writer.writerow(
                    [
                        run["name"],
                        meta["algo"],
                        meta["sequence_key"],
                        meta["steps"],
                        meta["seed"],
                        item["phase"],
                        item["train_env"],
                        item["eval_a_reward"],
                        item["eval_a_ref"],
                        item["retention_a"],
                        item["forgetting_a"],
                        item["elapsed_min"],
                    ]
                )
    return out_path


def aggregate_anchor_curves(run_records):
    grouped = defaultdict(lambda: defaultdict(list))
    phase_bounds = {}

    for run in run_records:
        seq_key = run["meta"]["sequence_key"]
        algo = run["meta"]["algo"]
        anchor_env = run["meta"]["sequence"][0]
        points = run["eval_series"].get(anchor_env, [])
        for point in points:
            grouped[seq_key][algo].append(point)
        if seq_key not in phase_bounds and run["phase_bounds"]:
            phase_bounds[seq_key] = run["phase_bounds"]

    curves = defaultdict(dict)
    for seq_key, algo_points in grouped.items():
        for algo, points in algo_points.items():
            by_step = defaultdict(list)
            for point in points:
                by_step[point["step"]].append(point["reward"])
            ordered_steps = sorted(by_step.keys())
            curves[seq_key][algo] = {
                "steps": ordered_steps,
                "mean": [mean_std(by_step[step])[0] for step in ordered_steps],
                "std": [mean_std(by_step[step])[1] or 0.0 for step in ordered_steps],
            }
    return curves, phase_bounds


def plot_eval_curves(run_records):
    curves, phase_bounds = aggregate_anchor_curves(run_records)
    sequences = sorted(curves.keys())
    if not sequences:
        return None

    fig, axes = plt.subplots(len(sequences), 1, figsize=(11.5, 4.4 * len(sequences)), sharex=False)
    if len(sequences) == 1:
        axes = [axes]

    for ax, seq_key in zip(axes, sequences):
        plotted = False
        for algo in ALGO_ORDER + [algo for algo in sorted(curves[seq_key].keys()) if algo not in ALGO_ORDER]:
            if algo not in curves[seq_key]:
                continue
            series = curves[seq_key][algo]
            x = np.array(series["steps"], dtype=float) / 1000.0
            y = np.array(series["mean"], dtype=float)
            s = np.array(series["std"], dtype=float)
            color = COLORS.get(algo, "#7F7F7F")
            label = ALGO_LABELS.get(algo, algo)
            ax.plot(x, y, linewidth=2.2, color=color, label=label)
            if np.any(s > 0):
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.16, linewidth=0)
            plotted = True

        for bound in phase_bounds.get(seq_key, [])[1:]:
            ax.axvline(bound["step"] / 1000.0, color="#BBBBBB", linestyle="--", linewidth=1.0, alpha=0.8)

        ymax = ax.get_ylim()[1]
        for bound in phase_bounds.get(seq_key, []):
            center = (bound["step"] + bound["phase_steps"] / 2.0) / 1000.0
            ax.text(
                center,
                ymax,
                f"P{bound['phase']}\n{bound['env']}",
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

        anchor_env = seq_key.split(" -> ")[0]
        ax.set_title(f"{seq_key} | Eval on Anchor Env ({anchor_env})")
        ax.set_ylabel("Eval Reward")
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(frameon=True, ncol=2, fontsize=9)

    axes[-1].set_xlabel("Global Training Steps (x1e3)")
    fig.suptitle("Sequence Evaluation Curves (Mean +/- Std Across Seeds)", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(OUT_DIR, "sequence_eval_curves.png")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def aggregate_phase_metrics(run_records):
    grouped = defaultdict(lambda: defaultdict(lambda: {"retention": [], "forgetting": [], "eval_a": []}))
    for run in run_records:
        seq_key = run["meta"]["sequence_key"]
        algo = run["meta"]["algo"]
        if not run["phase_summaries"]:
            continue
        final_phase = run["phase_summaries"][-1]
        grouped[seq_key][algo]["retention"].append(final_phase["retention_a"])
        grouped[seq_key][algo]["forgetting"].append(final_phase["forgetting_a"])
        grouped[seq_key][algo]["eval_a"].append(final_phase["eval_a_reward"])
    return grouped


def plot_retention_forgetting(run_records):
    grouped = aggregate_phase_metrics(run_records)
    sequences = sorted(grouped.keys())
    if not sequences:
        return None

    fig, axes = plt.subplots(len(sequences), 2, figsize=(13.8, 5.1 * len(sequences)))
    if len(sequences) == 1:
        axes = np.array([axes])

    for row_idx, seq_key in enumerate(sequences):
        algos = [algo for algo in ALGO_ORDER if algo in grouped[seq_key]]
        algos += [algo for algo in sorted(grouped[seq_key].keys()) if algo not in algos]
        labels = [ALGO_LABELS.get(algo, algo) for algo in algos]
        x = np.arange(len(algos))

        retention_means = []
        retention_stds = []
        forgetting_means = []
        forgetting_stds = []
        for algo in algos:
            mu, sd = mean_std(grouped[seq_key][algo]["retention"])
            retention_means.append(mu or 0.0)
            retention_stds.append(sd or 0.0)
            mu, sd = mean_std(grouped[seq_key][algo]["forgetting"])
            forgetting_means.append(mu or 0.0)
            forgetting_stds.append(sd or 0.0)

        ax_ret = axes[row_idx, 0]
        ax_for = axes[row_idx, 1]
        bar_colors = [COLORS.get(algo, "#7F7F7F") for algo in algos]
        seq_title = SEQUENCE_LABELS.get(seq_key, seq_key)

        ax_ret.bar(x, retention_means, yerr=retention_stds, color=bar_colors, alpha=0.9, capsize=4)
        ax_for.bar(x, forgetting_means, yerr=forgetting_stds, color=bar_colors, alpha=0.9, capsize=4)

        ax_ret.set_title(f"{seq_title}\nFinal Retention on Anchor Env", fontsize=11.5, pad=10)
        ax_for.set_title(f"{seq_title}\nFinal Forgetting on Anchor Env", fontsize=11.5, pad=10)
        ax_ret.set_ylabel("Retention Ratio", fontsize=11)
        ax_for.set_ylabel("Forgetting", fontsize=11)

        ax_ret.set_xticks(x)
        ax_ret.set_xticklabels(labels, rotation=13, ha="right", fontsize=10)
        ax_for.set_xticks(x)
        ax_for.set_xticklabels(labels, rotation=13, ha="right", fontsize=10)

        ax_ret.grid(axis="y", alpha=0.2)
        ax_for.grid(axis="y", alpha=0.2)
        ax_ret.tick_params(axis="y", labelsize=10)
        ax_for.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Sequence Retention / Forgetting Comparison\n(Mean +/- Std Across Seeds)",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.10, hspace=0.34, wspace=0.14)
    out_path = os.path.join(OUT_DIR, "sequence_retention_forgetting.png")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No sequence runs found.")
        return

    selected_runs = select_runs_for_plots(runs)
    run_records = build_run_records(selected_runs)
    csv_path = save_summary_csv(run_records)
    curve_path = plot_eval_curves(run_records)
    bar_path = plot_retention_forgetting(run_records)

    seq_names = sorted({record["meta"]["sequence_key"] for record in run_records})
    algo_names = sorted({record["meta"]["algo"] for record in run_records})

    print("Saved sequence plots to:", OUT_DIR)
    print("Selected sequences:", "; ".join(seq_names))
    print("Algorithms shown:", ", ".join(algo_names))
    print("Summary CSV:", csv_path)
    print("Eval curves:", curve_path)
    print("Retention/Forgetting:", bar_path)


if __name__ == "__main__":
    main()
