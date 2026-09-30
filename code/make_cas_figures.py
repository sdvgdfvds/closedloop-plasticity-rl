import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
OUT_DIR = os.path.join(os.getcwd(), "cas_ready")

ALGO_PRIORITY = [
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

CORE_ALGOS = [
    "PPO",
    "PPO+Cycle",
    "P3O",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill",
    "P3O-ClosedLoopAlpha",
    "P3O-ClosedLoopFull",
]

SEQUENCE_ALGOS = [
    "PPO",
    "P3O",
    "P3O-ClosedLoopFull",
    "P3O-ClosedLoopMemory",
]

ALGO_LABELS = {
    "PPO": "PPO",
    "PPO+Cycle": "PPO+Cycle",
    "P3O": "Stay Hungry / P3O",
    "P3O-dynamic": "Dyn Alpha",
    "P3O-dyn+AdaReset": "Dyn+AR",
    "P3O-dyn+HardDistill": "Dyn+HD",
    "P3O-dyn+AdaReset+HardDistill": "Dyn+AR+HD",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill": "Evt+AR+HD",
    "P3O-ClosedLoopAlpha": "ClosedLoop-Alpha",
    "P3O-ClosedLoopFull": "ClosedLoop-Full",
    "P3O-ClosedLoopMemory": "ClosedLoop-Memory",
}

COLORS = {
    "PPO": "#4E79A7",
    "PPO+Cycle": "#59A14F",
    "P3O": "#9C755F",
    "P3O-dynamic": "#F28E2B",
    "P3O-dyn+AdaReset": "#76B7B2",
    "P3O-dyn+HardDistill": "#B07AA1",
    "P3O-dyn+AdaReset+HardDistill": "#E15759",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill": "#EDC948",
    "P3O-ClosedLoopAlpha": "#AF7AA1",
    "P3O-ClosedLoopFull": "#17BECF",
    "P3O-ClosedLoopMemory": "#FF9DA7",
}

ENV_ORDER = ["Hopper-v4", "Walker2d-v4", "Ant-v4", "HalfCheetah-v4", "Humanoid-v4"]
SEQUENCE_PREFIX = "sequence_"
LEGACY_SEQUENCE_ALGO = "Legacy"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def ordered_envs(envs):
    prioritized = [env for env in ENV_ORDER if env in envs]
    rest = sorted(env for env in envs if env not in ENV_ORDER)
    return prioritized + rest


def parse_standard_folder(folder):
    if folder.startswith(SEQUENCE_PREFIX):
        return None
    if "_seed" not in folder or "_steps" not in folder:
        return None
    try:
        prefix, tail = folder.rsplit("_seed", 1)
        seed, steps = tail.split("_steps", 1)
        env, algo = prefix.split("_", 1)
        return {"env": env, "algo": algo, "seed": seed, "steps": int(steps)}
    except Exception:
        return None


def parse_sequence_folder(folder):
    if not folder.startswith(SEQUENCE_PREFIX) or "_seed" not in folder or "_steps" not in folder:
        return None
    try:
        tail = folder[len(SEQUENCE_PREFIX):]
        seq_part, rest = tail.rsplit("_seed", 1)
        seed, steps = rest.split("_steps", 1)

        algo = LEGACY_SEQUENCE_ALGO
        sequence_text = seq_part
        for env in ENV_ORDER:
            marker = f"_{env}"
            if marker in seq_part:
                prefix, suffix = seq_part.split(marker, 1)
                if prefix:
                    algo = prefix
                    sequence_text = f"{env}{suffix}"
                    break

        return {
            "algo": algo,
            "sequence": sequence_text.split("_to_"),
            "seed": seed,
            "steps": int(steps),
        }
    except Exception:
        return None


def parse_reward_row(row):
    if len(row) < 2 or row[1] != "reward":
        return None
    if len(row) >= 8:
        return {
            "step": safe_float(row[0]),
            "avg10": safe_float(row[4]),
        }
    return None


def parse_phase_summary_row(row):
    if len(row) < 9 or row[1] != "phase_summary":
        return None
    return {
        "step": safe_float(row[0]),
        "phase": int(float(row[2])),
        "train_env": row[3],
        "eval_a_reward": safe_float(row[4]),
        "eval_a_reference": safe_float(row[5]),
        "retention_a": safe_float(row[6]),
        "forgetting_a": safe_float(row[7]),
        "elapsed_min": safe_float(row[8]),
    }


def load_standard_runs(runs_dir):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for entry in sorted(os.listdir(runs_dir)):
        root = os.path.join(runs_dir, entry)
        if not os.path.isdir(root):
            continue
        meta = parse_standard_folder(entry)
        if not meta:
            continue
        metrics_path = os.path.join(root, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        rewards = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2 or (row[0] == "step" and row[1] == "tag"):
                    continue
                parsed = parse_reward_row(row)
                if parsed and parsed["step"] is not None and parsed["avg10"] is not None:
                    rewards.append(parsed)
        if rewards:
            rewards.sort(key=lambda item: item["step"])
            runs.append({**meta, "root": root, "rewards": rewards})
    return runs


def load_sequence_runs(runs_dir):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for entry in sorted(os.listdir(runs_dir)):
        root = os.path.join(runs_dir, entry)
        if not os.path.isdir(root):
            continue
        meta = parse_sequence_folder(entry)
        if not meta:
            continue
        metrics_path = os.path.join(root, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        phase_rows = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2 or (row[0] == "step" and row[1] == "tag"):
                    continue
                parsed_phase = parse_phase_summary_row(row)
                if parsed_phase and parsed_phase["step"] is not None:
                    phase_rows.append(parsed_phase)
        if phase_rows:
            runs.append({**meta, "root": root, "phase_rows": phase_rows})
    return runs


def choose_step_count(runs, explicit_steps=None):
    if explicit_steps is not None:
        return explicit_steps
    if not runs:
        return None
    score_by_steps = defaultdict(lambda: {"algos": set(), "env_algos": set(), "runs": 0})
    for run in runs:
        info = score_by_steps[run["steps"]]
        info["algos"].add(run["algo"])
        info["env_algos"].add((run["env"], run["algo"]))
        info["runs"] += 1
    ranked = sorted(
        score_by_steps.items(),
        key=lambda item: (len(item[1]["algos"]), len(item[1]["env_algos"]), item[1]["runs"], item[0]),
    )
    return ranked[-1][0]


def choose_algorithms(runs, steps):
    available = {run["algo"] for run in runs if run["steps"] == steps}
    chosen = [algo for algo in CORE_ALGOS if algo in available]
    if len(chosen) < 4:
        chosen = [algo for algo in ALGO_PRIORITY if algo in available]
    return chosen


def aggregate_curve(runs, env, algo, steps, step_gap=5000):
    selected = []
    for run in runs:
        if run["env"] != env or run["algo"] != algo or run["steps"] != steps:
            continue
        points = [(item["step"], item["avg10"]) for item in run["rewards"]]
        if points:
            selected.append(points)
    if not selected:
        return None, None, None

    max_step = max(points[-1][0] for points in selected)
    grid = np.arange(0, max_step + step_gap, step_gap)
    values = []
    for points in selected:
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        interp = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        values.append(interp)
    arr = np.array(values, dtype=np.float64)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def build_summary(runs, steps):
    summary = defaultdict(lambda: defaultdict(list))
    auc_summary = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run["steps"] != steps:
            continue
        if not run["rewards"]:
            continue
        final_val = run["rewards"][-1]["avg10"]
        summary[run["env"]][run["algo"]].append(final_val)
        xs = np.array([item["step"] for item in run["rewards"]], dtype=np.float64)
        ys = np.array([item["avg10"] for item in run["rewards"]], dtype=np.float64)
        if len(xs) >= 2:
            auc = float(np.trapezoid(ys, xs) / max(xs[-1], 1.0))
        else:
            auc = float(ys[-1])
        auc_summary[run["env"]][run["algo"]].append(auc)
    return summary, auc_summary


def save_summary_csv(summary, auc_summary, algos):
    envs = ordered_envs(summary.keys())
    csv_path = os.path.join(OUT_DIR, "cas_summary_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "algo", "final_avg10_mean", "final_avg10_std", "auc_mean", "auc_std"])
        for env in envs:
            for algo in algos:
                final_mean, final_std = mean_std(summary[env].get(algo, []))
                auc_mean, auc_std = mean_std(auc_summary[env].get(algo, []))
                writer.writerow(
                    [
                        env,
                        algo,
                        "" if final_mean is None else round(final_mean, 4),
                        "" if final_std is None else round(final_std, 4),
                        "" if auc_mean is None else round(auc_mean, 4),
                        "" if auc_std is None else round(auc_std, 4),
                    ]
                )
    return csv_path


def plot_main_reward_curves(runs, steps, algos):
    envs = ordered_envs({run["env"] for run in runs if run["steps"] == steps})
    if not envs or not algos:
        return None
    cols = 2 if len(envs) > 1 else 1
    rows = int(math.ceil(len(envs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.4 * cols, 4.6 * rows), squeeze=False)
    axes_flat = axes.flatten()
    legend_handles = {}
    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        for algo in algos:
            grid, mean_vals, std_vals = aggregate_curve(runs, env, algo, steps)
            if grid is None:
                continue
            color = COLORS.get(algo)
            label = ALGO_LABELS.get(algo, algo)
            (line,) = ax.plot(grid / 1_000_000, mean_vals, linewidth=2.2, color=color, label=label)
            legend_handles.setdefault(algo, line)
            if np.isfinite(std_vals).any():
                ax.fill_between(
                    grid / 1_000_000,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color,
                    alpha=0.14,
                )
        ax.set_title(env)
        ax.set_xlabel("Training Steps (Millions)")
        ax.set_ylabel("Avg10 Reward")
        ax.grid(alpha=0.25)
    for idx in range(len(envs), len(axes_flat)):
        axes_flat[idx].axis("off")
    if legend_handles:
        ordered_handles = [legend_handles[algo] for algo in algos if algo in legend_handles]
        ordered_labels = [ALGO_LABELS.get(algo, algo) for algo in algos if algo in legend_handles]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=min(3, max(1, len(ordered_handles))),
            frameon=True,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=10,
        )
    fig.suptitle(
        f"Main Performance Curves at {steps:,} Steps (mean +/- std over available seeds)",
        fontsize=15,
        y=1.03,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    path = os.path.join(OUT_DIR, "cas_main_reward_curves.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison_bars(summary, algos):
    envs = ordered_envs(summary.keys())
    if not envs or not algos:
        return None
    x = np.arange(len(envs))
    width = min(0.82 / max(len(algos), 1), 0.16)

    fig, ax = plt.subplots(figsize=(max(10, 2.0 * len(envs) + 7), 5.4))
    for idx, algo in enumerate(algos):
        means = []
        stds = []
        for env in envs:
            mu, sigma = mean_std(summary[env].get(algo, []))
            means.append(0.0 if mu is None else mu)
            stds.append(0.0 if sigma is None else sigma)
        offset = (idx - (len(algos) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=COLORS.get(algo),
            alpha=0.92,
            label=ALGO_LABELS.get(algo, algo),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Final Avg10 Reward")
    ax.set_title("Final Performance Comparison Across Environments (mean +/- std)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2, fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cas_algo_comparison_bars.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_table_figure(summary, auc_summary, algos):
    envs = ordered_envs(summary.keys())
    if not envs or not algos:
        return None

    headers = ["Env"] + [ALGO_LABELS.get(algo, algo) for algo in algos]
    rows = []
    for env in envs:
        row = [env]
        for algo in algos:
            mu, sigma = mean_std(summary[env].get(algo, []))
            auc_mu, _ = mean_std(auc_summary[env].get(algo, []))
            if mu is None:
                row.append("N/A")
            elif sigma is None or sigma == 0:
                row.append(f"{mu:.1f}\nAUC {auc_mu:.1f}" if auc_mu is not None else f"{mu:.1f}")
            else:
                row.append(f"{mu:.1f} +/- {sigma:.1f}\nAUC {auc_mu:.1f}" if auc_mu is not None else f"{mu:.1f} +/- {sigma:.1f}")
        rows.append(row)

    fig_w = max(12.0, 1.65 * len(headers))
    fig_h = max(3.8, 0.95 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.95)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#E9EEF6")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#F6F7F9")
            cell.set_text_props(weight="bold")
    ax.set_title("Performance Table: Final Avg10 Reward and Normalized AUC", fontsize=14, pad=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cas_performance_table.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def choose_sequence_groups(sequence_runs):
    if not sequence_runs:
        return []
    by_sequence = defaultdict(list)
    for run in sequence_runs:
        if len(run["sequence"]) < 3:
            continue
        by_sequence[tuple(run["sequence"])].append(run)

    selected = []
    for sequence_names, runs in by_sequence.items():
        best_steps = max(run["steps"] for run in runs)
        chosen_runs = [run for run in runs if run["steps"] == best_steps]
        selected.append(((sequence_names, best_steps), chosen_runs))

    return sorted(
        selected,
        key=lambda item: (" -> ".join(item[0][0]), item[0][1]),
    )


def summarize_sequence_groups(groups):
    summary = {}
    for (sequence_names, sequence_steps), runs in groups:
        named_algos = {run["algo"] for run in runs if run["algo"] != LEGACY_SEQUENCE_ALGO}
        algo_summary = defaultdict(lambda: {"retention": [], "forgetting": [], "reward": []})
        for run in runs:
            if named_algos and run["algo"] == LEGACY_SEQUENCE_ALGO:
                continue
            if not run["phase_rows"]:
                continue
            last_phase = sorted(run["phase_rows"], key=lambda item: item["phase"])[-1]
            algo_summary[run["algo"]]["retention"].append(last_phase["retention_a"])
            algo_summary[run["algo"]]["forgetting"].append(last_phase["forgetting_a"])
            algo_summary[run["algo"]]["reward"].append(last_phase["eval_a_reward"])
        summary[(sequence_names, sequence_steps)] = algo_summary
    return summary


def save_sequence_csv(sequence_summary):
    csv_path = os.path.join(OUT_DIR, "cas_sequence_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "steps_per_phase",
                "algo",
                "num_runs",
                "final_retention_mean",
                "final_retention_std",
                "final_forgetting_mean",
                "final_forgetting_std",
                "final_eval_a_reward_mean",
                "final_eval_a_reward_std",
            ]
        )
        for (sequence_names, sequence_steps), algo_summary in sorted(sequence_summary.items()):
            sequence_name = " -> ".join(sequence_names)
            for algo in [algo for algo in SEQUENCE_ALGOS if algo in algo_summary]:
                r_mu, r_std = mean_std(algo_summary[algo]["retention"])
                f_mu, f_std = mean_std(algo_summary[algo]["forgetting"])
                e_mu, e_std = mean_std(algo_summary[algo]["reward"])
                writer.writerow(
                    [
                        sequence_name,
                        sequence_steps,
                        algo,
                        len(algo_summary[algo]["retention"]),
                        "" if r_mu is None else round(r_mu, 6),
                        "" if r_std is None else round(r_std, 6),
                        "" if f_mu is None else round(f_mu, 6),
                        "" if f_std is None else round(f_std, 6),
                        "" if e_mu is None else round(e_mu, 6),
                        "" if e_std is None else round(e_std, 6),
                    ]
                )
    return csv_path


def plot_sequence_summary(sequence_summary):
    if not sequence_summary:
        return None
    ordered_groups = sorted(sequence_summary.items(), key=lambda item: (" -> ".join(item[0][0]), item[0][1]))
    fig, axes = plt.subplots(len(ordered_groups), 3, figsize=(15.0, 4.5 * len(ordered_groups)), squeeze=False)
    metric_specs = [
        ("retention", "Final Retention on A"),
        ("forgetting", "Final Forgetting on A"),
        ("reward", "Final Eval Reward on A"),
    ]

    for row_idx, ((sequence_names, sequence_steps), algo_summary) in enumerate(ordered_groups):
        algos = [algo for algo in SEQUENCE_ALGOS if algo in algo_summary]
        x = np.arange(len(algos))
        subtitle = f"{' -> '.join(sequence_names)} | steps/phase={sequence_steps:,}"
        for col_idx, (metric_key, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            means = []
            stds = []
            for algo in algos:
                mu, sigma = mean_std(algo_summary[algo][metric_key])
                means.append(0.0 if mu is None else mu)
                stds.append(0.0 if sigma is None else sigma)
            colors = [COLORS.get(algo, "#999999") for algo in algos]
            ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.92)
            ax.set_xticks(x)
            ax.set_xticklabels([ALGO_LABELS.get(algo, algo) for algo in algos], rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(subtitle)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Sequence Comparison Across Tasks (mean +/- std over available seeds)", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    path = os.path.join(OUT_DIR, "cas_sequence_summary.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_overview(steps, algos, curve_path, bar_path, table_path, sequence_path, summary_csv, sequence_csv, sequence_summary):
    md_path = os.path.join(OUT_DIR, "cas_figure_overview.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CAS Figure Overview\n\n")
        f.write(f"- Main comparison step budget: `{steps}`\n")
        f.write(f"- Algorithms shown: {', '.join(ALGO_LABELS.get(algo, algo) for algo in algos)}\n")
        f.write("- Main plots summarize all available runs at the selected step budget.\n")
        f.write(f"- Reward curves: `{curve_path}`\n")
        f.write(f"- Comparison bars: `{bar_path}`\n")
        f.write(f"- Table figure: `{table_path}`\n")
        if sequence_path:
            f.write(f"- Sequence summary: `{sequence_path}`\n")
            for (sequence_names, sequence_steps), algo_summary in sorted(sequence_summary.items()):
                visible_algos = [ALGO_LABELS.get(algo, algo) for algo in SEQUENCE_ALGOS if algo in algo_summary]
                f.write(
                    f"- Sequence included: `{' -> '.join(sequence_names)}` at `{sequence_steps}` steps/phase; algorithms: {', '.join(visible_algos)}\n"
                )
        f.write(f"- Summary CSV: `{summary_csv}`\n")
        if sequence_csv:
            f.write(f"- Sequence CSV: `{sequence_csv}`\n")
    return md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate higher-strength paper figures from p3o_runs.")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--steps", type=int, default=None, help="Optional explicit training step budget to visualize.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global OUT_DIR
    OUT_DIR = args.out_dir

    standard_runs = load_standard_runs(args.runs_dir)
    if not standard_runs:
        print("No standard experiment runs found in", args.runs_dir)
        return

    steps = choose_step_count(standard_runs, explicit_steps=args.steps)
    algos = choose_algorithms(standard_runs, steps)
    summary, auc_summary = build_summary(standard_runs, steps)

    curve_path = plot_main_reward_curves(standard_runs, steps, algos)
    bar_path = plot_comparison_bars(summary, algos)
    table_path = plot_table_figure(summary, auc_summary, algos)
    summary_csv = save_summary_csv(summary, auc_summary, algos)

    sequence_runs = load_sequence_runs(args.runs_dir)
    sequence_summary = {}
    sequence_path = None
    sequence_csv = None
    chosen_sequences = choose_sequence_groups(sequence_runs)
    if chosen_sequences:
        sequence_summary = summarize_sequence_groups(chosen_sequences)
        sequence_path = plot_sequence_summary(sequence_summary)
        sequence_csv = save_sequence_csv(sequence_summary)

    overview_path = write_overview(
        steps,
        algos,
        curve_path,
        bar_path,
        table_path,
        sequence_path,
        summary_csv,
        sequence_csv,
        sequence_summary,
    )

    print("Saved CAS figures to:", args.out_dir)
    print("Chosen step budget:", steps)
    print("Algorithms:", ", ".join(algos))
    print("Reward curves:", curve_path)
    print("Comparison bars:", bar_path)
    print("Table figure:", table_path)
    if sequence_path:
        print("Sequence summary:", sequence_path)
    print("Summary CSV:", summary_csv)
    if sequence_csv:
        print("Sequence CSV:", sequence_csv)
    print("Overview:", overview_path)


if __name__ == "__main__":
    main()
