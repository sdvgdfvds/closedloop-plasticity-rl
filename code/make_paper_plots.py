import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


# ====== Paths ======
RUNS_DIR = r"C:\Users\33277\Desktop\p3o_runs"
OUT_DIR = os.path.join(RUNS_DIR, "paper_ready")


# ====== Filters ======
ALLOWED_ALGOS = {"PPO", "PPO+Cycle", "P3O-static", "P3O-dynamic"}
ALLOWED_STEPS = {500000}
ALLOWED_ENVS = None


# ====== Plot config ======
STEP_GAP_REWARD = 2000
STEP_GAP_LOSS = 5000
STEP_GAP_ALPHA = 10000
SMOOTH_WINDOW = 3

ENV_ORDER = [
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "HalfCheetah-v4",
    "Humanoid-v4",
]
ALGO_ORDER = ["PPO", "PPO+Cycle", "P3O-static", "P3O-dynamic"]
ALGO_COLORS = {
    "PPO": "#1f77b4",
    "PPO+Cycle": "#17becf",
    "P3O-static": "#ff7f0e",
    "P3O-dynamic": "#2ca02c",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 320,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    }
)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def moving_average(values, window):
    if window <= 1 or not values:
        return values
    out = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = [v for v in values[left : i + 1] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def parse_folder_name(folder_name):
    # Example: Hopper-v4_P3O-dynamic_seed0_steps500000
    parts = folder_name.split("_")
    if len(parts) < 4:
        return None, None, None, None
    env = parts[0]
    algo = parts[1]
    seed = parts[2].replace("seed", "")
    steps = None
    if parts[3].startswith("steps"):
        try:
            steps = int(parts[3].replace("steps", ""))
        except Exception:
            steps = None
    return env, algo, seed, steps


def parse_row(row):
    if not row or len(row) < 2:
        return None
    if row[0] == "step" and row[1] == "tag":
        return None
    tag = row[1]

    def g(i):
        return row[i].strip() if i < len(row) else ""

    if tag == "reward" and len(row) >= 8:
        return {
            "step": g(0),
            "tag": g(1),
            "episode": g(2),
            "reward": g(3),
            "avg10": g(4),
            "algo": g(5),
            "env": g(6),
            "seed": g(7),
        }
    if tag == "loss" and len(row) >= 7:
        return {
            "step": g(0),
            "tag": g(1),
            "actor_loss": g(2),
            "critic_loss": g(3),
            "algo": g(4),
            "env": g(5),
            "seed": g(6),
        }
    if tag == "sbp" and len(row) >= 8:
        return {
            "step": g(0),
            "tag": g(1),
            "alpha_static": g(2),
            "alpha_dynamic": g(3),
            "alpha_using": g(4),
            "algo": g(5),
            "env": g(6),
            "seed": g(7),
        }
    if tag == "speed" and len(row) >= 7:
        return {
            "step": g(0),
            "tag": g(1),
            "sps": g(2),
            "elapsed_min": g(3),
            "algo": g(4),
            "env": g(5),
            "seed": g(6),
        }
    return None


def load_runs():
    runs = []
    for root, _, files in os.walk(RUNS_DIR):
        if "metrics.csv" not in files:
            continue

        folder = os.path.basename(root)
        env_f, algo_f, seed_f, steps_f = parse_folder_name(folder)
        if ALLOWED_ALGOS and algo_f not in ALLOWED_ALGOS:
            continue
        if ALLOWED_STEPS and steps_f not in ALLOWED_STEPS:
            continue
        if ALLOWED_ENVS and env_f not in ALLOWED_ENVS:
            continue

        rows = []
        metrics_path = os.path.join(root, "metrics.csv")
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                parsed = parse_row(row)
                if parsed:
                    rows.append(parsed)

        if not rows:
            continue

        runs.append(
            {
                "env": rows[0].get("env") or env_f,
                "algo": rows[0].get("algo") or algo_f,
                "seed": str(rows[0].get("seed") or seed_f or "0"),
                "rows": rows,
            }
        )
    return runs


def resample_points(points, step_gap):
    if not points:
        return [], []
    points.sort(key=lambda x: x[0])
    max_step = points[-1][0]
    grid = list(range(0, max_step + 1, step_gap))
    vals = []
    idx = 0
    last_val = None
    for s in grid:
        while idx < len(points) and points[idx][0] <= s:
            last_val = points[idx][1]
            idx += 1
        vals.append(last_val)
    return grid, vals


def aggregate_series(runs, tag, value_key, step_gap):
    data = defaultdict(lambda: defaultdict(list))
    for run in runs:
        points = []
        for row in run["rows"]:
            if row.get("tag") != tag:
                continue
            step = safe_float(row.get("step"))
            val = safe_float(row.get(value_key))
            if step is None or val is None:
                continue
            points.append((int(step), val))
        steps, vals = resample_points(points, step_gap)
        if steps:
            data[run["env"]][run["algo"]].append((steps, vals))
    return data


def stats_from_runs(series_list):
    if not series_list:
        return None
    steps = series_list[0][0]
    matrix = []
    for _, vals in series_list:
        matrix.append([v if v is not None else float("nan") for v in vals])

    n_steps = len(matrix[0])
    mean, std, ci95 = [], [], []
    for j in range(n_steps):
        col = [row[j] for row in matrix if row[j] == row[j]]
        if not col:
            mean.append(float("nan"))
            std.append(float("nan"))
            ci95.append(float("nan"))
            continue
        m = sum(col) / len(col)
        v = sum((x - m) ** 2 for x in col) / len(col)
        s = math.sqrt(v)
        ci = 1.96 * s / math.sqrt(len(col)) if len(col) > 1 else 0.0
        mean.append(m)
        std.append(s)
        ci95.append(ci)
    return steps, mean, std, ci95


def ordered_envs(data):
    envs = list(data.keys())
    prioritized = [e for e in ENV_ORDER if e in envs]
    remaining = sorted([e for e in envs if e not in prioritized])
    return prioritized + remaining


def plot_grid(data, title_prefix, y_label, out_name, use_ci=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    envs = ordered_envs(data)
    if not envs:
        return None

    cols = 2 if len(envs) > 1 else 1
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.2 * cols, 3.5 * rows), squeeze=False)

    for i, env in enumerate(envs):
        ax = axes[i // cols][i % cols]
        for algo in ALGO_ORDER:
            series_list = data[env].get(algo, [])
            if not series_list:
                continue
            result = stats_from_runs(series_list)
            if result is None:
                continue
            steps, mean, _, ci95 = result
            mean = moving_average(mean, SMOOTH_WINDOW)
            ci95 = moving_average(ci95, SMOOTH_WINDOW)
            color = ALGO_COLORS.get(algo, None)
            ax.plot(steps, mean, label=algo, color=color)
            if use_ci:
                upper = [
                    m + c if (m == m and c == c) else float("nan")
                    for m, c in zip(mean, ci95)
                ]
                lower = [
                    m - c if (m == m and c == c) else float("nan")
                    for m, c in zip(mean, ci95)
                ]
                ax.fill_between(steps, lower, upper, alpha=0.12, color=color)
        ax.set_title(env)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.28)

    for k in range(i + 1, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    # Keep title and legend on separate rows to avoid overlap in Word exports.
    fig.suptitle(title_prefix, y=0.992, fontsize=13)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.958),
            ncol=min(4, len(labels)),
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def build_summary_records(runs):
    records = []
    for run in runs:
        last_avg10 = None
        speed_values = []
        for row in run["rows"]:
            if row.get("tag") == "reward":
                last_avg10 = safe_float(row.get("avg10"))
            if row.get("tag") == "speed":
                sps = safe_float(row.get("sps"))
                if sps is not None:
                    speed_values.append(sps)
        avg_sps = sum(speed_values) / len(speed_values) if speed_values else None
        records.append(
            {
                "env": run["env"],
                "algo": run["algo"],
                "seed": run["seed"],
                "final_avg10": last_avg10,
                "avg_sps": avg_sps,
            }
        )
    return records


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_main_table(summary_records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in summary_records:
        v = safe_float(row.get("final_avg10"))
        if v is None:
            continue
        grouped[row["env"]][row["algo"]].append(v)

    table_rows = []
    for env in ordered_envs(grouped):
        ppo_vals = grouped[env].get("PPO", [])
        ppo_mean = sum(ppo_vals) / len(ppo_vals) if ppo_vals else None
        for algo in ALGO_ORDER:
            vals = grouped[env].get(algo, [])
            if vals:
                n = len(vals)
                mean = sum(vals) / n
                var = sum((x - mean) ** 2 for x in vals) / n
                std = math.sqrt(var)
                ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
                lo = mean - ci95
                hi = mean + ci95
                improve = None
                if ppo_mean is not None and abs(ppo_mean) > 1e-8:
                    improve = 100.0 * (mean - ppo_mean) / abs(ppo_mean)
                table_rows.append(
                    {
                        "env": env,
                        "algo": algo,
                        "n_seeds": n,
                        "mean": round(mean, 3),
                        "ci95_low": round(lo, 3),
                        "ci95_high": round(hi, 3),
                        "mean_ci95": f"{mean:.2f} [{lo:.2f}, {hi:.2f}]",
                        "improve_over_ppo_pct": (
                            f"{improve:.1f}%" if improve is not None else "N/A"
                        ),
                    }
                )
            else:
                table_rows.append(
                    {
                        "env": env,
                        "algo": algo,
                        "n_seeds": 0,
                        "mean": "",
                        "ci95_low": "",
                        "ci95_high": "",
                        "mean_ci95": "N/A",
                        "improve_over_ppo_pct": "N/A",
                    }
                )
    return table_rows


def save_main_table_files(table_rows):
    csv_path = os.path.join(OUT_DIR, "table_main_95ci.csv")
    save_csv(
        csv_path,
        table_rows,
        [
            "env",
            "algo",
            "n_seeds",
            "mean",
            "ci95_low",
            "ci95_high",
            "mean_ci95",
            "improve_over_ppo_pct",
        ],
    )

    # compact markdown (per env as a row)
    by_env = defaultdict(dict)
    for row in table_rows:
        by_env[row["env"]][row["algo"]] = row
    md_path = os.path.join(OUT_DIR, "table_main_95ci.md")
    with open(md_path, "w", encoding="utf-8") as f:
        header = [
            "Env",
            "PPO",
            "PPO+Cycle",
            "P3O-static",
            "P3O-dynamic",
            "P3O-dynamic vs PPO",
        ]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + " --- |" * len(header) + "\n")
        for env in ordered_envs(by_env):
            ppo = by_env[env].get("PPO", {}).get("mean_ci95", "N/A")
            cycle = by_env[env].get("PPO+Cycle", {}).get("mean_ci95", "N/A")
            st = by_env[env].get("P3O-static", {}).get("mean_ci95", "N/A")
            dy = by_env[env].get("P3O-dynamic", {}).get("mean_ci95", "N/A")
            imp = by_env[env].get("P3O-dynamic", {}).get("improve_over_ppo_pct", "N/A")
            f.write(f"| {env} | {ppo} | {cycle} | {st} | {dy} | {imp} |\n")
    return csv_path, md_path


def plot_summary_bar_with_ci(table_rows):
    grouped = defaultdict(dict)
    for row in table_rows:
        grouped[row["env"]][row["algo"]] = row
    envs = ordered_envs(grouped)
    if not envs:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = list(range(len(envs)))
    width = 0.18
    for i, algo in enumerate(ALGO_ORDER):
        means = []
        ci95 = []
        for env in envs:
            row = grouped[env].get(algo)
            m = safe_float(row.get("mean")) if row else None
            lo = safe_float(row.get("ci95_low")) if row else None
            hi = safe_float(row.get("ci95_high")) if row else None
            if m is None:
                means.append(0.0)
                ci95.append(0.0)
            else:
                means.append(m)
                ci95.append(max(abs(m - lo), abs(hi - m)))
        shift = [xi + (i - 1.5) * width for xi in x]
        ax.bar(
            shift,
            means,
            width=width,
            yerr=ci95,
            color=ALGO_COLORS.get(algo, None),
            alpha=0.87,
            label=algo,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=0)
    ax.set_ylabel("Final Avg Reward (mean with 95% CI)")
    ax.set_title("Final Performance Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "figure_summary_bar_95ci.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def write_caption_template():
    caption_path = os.path.join(OUT_DIR, "figure_captions_cn.txt")
    text = """图注模板（可直接贴到论文）

图1 不同算法在多个环境上的学习曲线（Reward, mean with 95% CI）。横轴为环境交互步数，纵轴为最近10回合平均奖励。
图2 动态与静态 alpha 调度对比。静态 alpha 为常数，动态 alpha 随训练进度变化。
图3 Actor Loss 曲线（mean with 95% CI），用于比较训练稳定性与收敛行为。
图4 最终性能柱状图（mean with 95% CI）。每组柱对应一个环境，不同颜色表示不同算法。
表1 多环境最终性能汇总（mean [95% CI]），并报告相对 PPO 的提升百分比。

注1 所有结果均在相同训练步数下比较。
注2 如出现 N/A，表示该配置尚未运行或缺失对应日志。
"""
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(text)
    return caption_path


def main():
    if not os.path.isdir(RUNS_DIR):
        print("RUNS_DIR not found:", RUNS_DIR)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    runs = load_runs()
    if not runs:
        print("No metrics.csv found under:", RUNS_DIR)
        return

    reward = aggregate_series(runs, tag="reward", value_key="avg10", step_gap=STEP_GAP_REWARD)
    loss = aggregate_series(runs, tag="loss", value_key="actor_loss", step_gap=STEP_GAP_LOSS)
    alpha = aggregate_series(runs, tag="sbp", value_key="alpha_using", step_gap=STEP_GAP_ALPHA)
    summary = build_summary_records(runs)
    table_rows = build_main_table(summary)

    p1 = plot_grid(
        reward,
        title_prefix="Performance in MuJoCo Environments",
        y_label="Average Return (last 10 episodes)",
        out_name="figure_performance_grid.png",
        use_ci=True,
    )
    p2 = plot_grid(
        alpha,
        title_prefix="Dynamic vs Static Alpha Schedule",
        y_label="Alpha",
        out_name="figure_alpha_grid.png",
        use_ci=False,
    )
    p3 = plot_grid(
        loss,
        title_prefix="Actor Loss in MuJoCo Environments",
        y_label="Actor Loss",
        out_name="figure_actor_loss_grid.png",
        use_ci=True,
    )
    csv_path, md_path = save_main_table_files(table_rows)
    p4 = plot_summary_bar_with_ci(table_rows)
    cap_path = write_caption_template()

    print("Saved to:", OUT_DIR)
    print("Figure1:", p1)
    print("Figure2:", p2)
    print("Figure3:", p3)
    print("Figure4:", p4)
    print("Table CSV:", csv_path)
    print("Table MD:", md_path)
    print("Captions:", cap_path)


if __name__ == "__main__":
    main()
