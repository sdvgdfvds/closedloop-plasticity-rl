import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
OUT_DIR = os.environ.get("P3O_ABLATION_OUT", os.path.join(os.getcwd(), "ablation_ready"))
ALGO_ORDER = ["PPO", "PPO+Cycle", "P3O-static", "P3O-dynamic"]
ALGO_COLORS = {
    "PPO": "#1f77b4",
    "PPO+Cycle": "#17becf",
    "P3O-static": "#ff7f0e",
    "P3O-dynamic": "#2ca02c",
}
ALLOWED_STEPS = {500000}
ENV_ORDER = ["Hopper-v4", "Walker2d-v4", "Ant-v4", "HalfCheetah-v4", "Humanoid-v4"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)


# ---------------------------
# Parsing
# ---------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_folder_name(folder_name):
    # Expected: <env>_<algo>_seed<seed>_steps<steps>
    if "_seed" not in folder_name or "_steps" not in folder_name:
        return None
    try:
        prefix, tail = folder_name.rsplit("_seed", 1)
        seed_str, steps_str = tail.split("_steps", 1)
        env, algo = prefix.split("_", 1)
        return {
            "env": env,
            "algo": algo,
            "seed": seed_str,
            "steps": int(steps_str),
        }
    except Exception:
        return None


def parse_row(row):
    if not row or len(row) < 2:
        return None
    if row[0] == "step" and row[1] == "tag":
        return None

    tag = row[1].strip()

    def g(i):
        return row[i].strip() if i < len(row) else ""

    if tag == "reward" and len(row) >= 8:
        return {
            "step": g(0),
            "tag": "reward",
            "reward": g(3),
            "avg10": g(4),
            "algo": g(5),
            "env": g(6),
            "seed": g(7),
        }
    if tag == "loss" and len(row) >= 7:
        return {
            "step": g(0),
            "tag": "loss",
            "actor_loss": g(2),
            "critic_loss": g(3),
            "algo": g(4),
            "env": g(5),
            "seed": g(6),
        }
    if tag == "speed" and len(row) >= 7:
        return {
            "step": g(0),
            "tag": "speed",
            "sps": g(2),
            "elapsed_min": g(3),
            "algo": g(4),
            "env": g(5),
            "seed": g(6),
        }
    if tag == "sbp" and len(row) >= 8:
        return {
            "step": g(0),
            "tag": "sbp",
            "alpha_static": g(2),
            "alpha_dynamic": g(3),
            "alpha_using": g(4),
            "algo": g(5),
            "env": g(6),
            "seed": g(7),
        }
    return None


def load_runs():
    runs = []
    for root, _, files in os.walk(RUNS_DIR):
        if "metrics.csv" not in files:
            continue

        folder = os.path.basename(root)
        meta = parse_folder_name(folder)
        if not meta:
            continue
        if ALLOWED_STEPS and meta["steps"] not in ALLOWED_STEPS:
            continue
        if meta["algo"] not in ALGO_ORDER:
            continue

        metrics_path = os.path.join(root, "metrics.csv")
        rows = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                parsed = parse_row(row)
                if parsed:
                    rows.append(parsed)
        if not rows:
            continue

        run = {
            "env": rows[0].get("env") or meta["env"],
            "algo": rows[0].get("algo") or meta["algo"],
            "seed": str(rows[0].get("seed") or meta["seed"]),
            "steps": meta["steps"],
            "rows": rows,
        }
        run.update(extract_run_metrics(run))
        runs.append(run)
    return runs


def extract_run_metrics(run):
    reward_points = []
    speed_values = []
    alpha_points = []

    for r in run["rows"]:
        step = safe_float(r.get("step"))
        if step is None:
            continue
        step = int(step)

        if r.get("tag") == "reward":
            avg10 = safe_float(r.get("avg10"))
            if avg10 is not None:
                reward_points.append((step, avg10))

        if r.get("tag") == "speed":
            sps = safe_float(r.get("sps"))
            if sps is not None:
                speed_values.append(sps)

        if r.get("tag") == "sbp":
            alpha_using = safe_float(r.get("alpha_using"))
            if alpha_using is not None:
                alpha_points.append((step, alpha_using))

    reward_points.sort(key=lambda x: x[0])
    alpha_points.sort(key=lambda x: x[0])

    final_avg10 = reward_points[-1][1] if reward_points else None
    best_avg10 = max([v for _, v in reward_points], default=None)
    avg_sps = sum(speed_values) / len(speed_values) if speed_values else None

    # Mean reward over training trajectory (normalized AUC)
    mean_reward_over_time = None
    if len(reward_points) >= 2:
        area = 0.0
        for i in range(1, len(reward_points)):
            x0, y0 = reward_points[i - 1]
            x1, y1 = reward_points[i]
            area += 0.5 * (y0 + y1) * (x1 - x0)
        duration = reward_points[-1][0] - reward_points[0][0]
        if duration > 0:
            mean_reward_over_time = area / duration

    return {
        "reward_points": reward_points,
        "alpha_points": alpha_points,
        "final_avg10": final_avg10,
        "best_avg10": best_avg10,
        "avg_sps": avg_sps,
        "mean_reward_over_time": mean_reward_over_time,
    }


# ---------------------------
# Aggregation
# ---------------------------
def ordered_envs(envs):
    pri = [e for e in ENV_ORDER if e in envs]
    rest = sorted([e for e in envs if e not in pri])
    return pri + rest


def calc_stats(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def fmt_mean_std(stat):
    if not stat:
        return "N/A"
    return f"{stat['mean']:.2f} +/- {stat['std']:.2f}"


def build_grouped_stats(runs, key):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        grouped[r["env"]][r["algo"]].append(r.get(key))
    out = defaultdict(dict)
    for env, algo_map in grouped.items():
        for algo, vals in algo_map.items():
            out[env][algo] = calc_stats(vals)
    return out


def build_component_table(runs):
    final_stats = build_grouped_stats(runs, "final_avg10")
    rows = []
    for env in ordered_envs(final_stats.keys()):
        ppo = final_stats[env].get("PPO")
        cyc = final_stats[env].get("PPO+Cycle")
        stt = final_stats[env].get("P3O-static")
        dyn = final_stats[env].get("P3O-dynamic")

        def delta(a, b):
            if not a or not b:
                return "N/A"
            return f"{(b['mean'] - a['mean']):+.2f}"

        rows.append(
            {
                "env": env,
                "ppo": fmt_mean_std(ppo),
                "ppo_cycle": fmt_mean_std(cyc),
                "p3o_static": fmt_mean_std(stt),
                "p3o_dynamic": fmt_mean_std(dyn),
                "delta_cycle_vs_ppo": delta(ppo, cyc),
                "delta_static_vs_cycle": delta(cyc, stt),
                "delta_dynamic_vs_static": delta(stt, dyn),
            }
        )
    return rows, final_stats


def build_alpha_table(runs):
    # paired comparison by same env + seed
    lookup = defaultdict(dict)
    for r in runs:
        lookup[(r["env"], r["seed"])][r["algo"]] = r

    pairs = defaultdict(list)
    for (env, _seed), algo_map in lookup.items():
        a = algo_map.get("P3O-static")
        b = algo_map.get("P3O-dynamic")
        if a and b and a.get("final_avg10") is not None and b.get("final_avg10") is not None:
            pairs[env].append((a["final_avg10"], b["final_avg10"]))

    rows = []
    for env in ordered_envs(pairs.keys()):
        arr = pairs[env]
        if not arr:
            continue
        st = [x for x, _ in arr]
        dy = [y for _, y in arr]
        diff = [y - x for x, y in arr]
        wins = sum(1 for d in diff if d > 0)
        n = len(diff)

        st_s = calc_stats(st)
        dy_s = calc_stats(dy)
        d_s = calc_stats(diff)

        rel = "N/A"
        if st_s and abs(st_s["mean"]) > 1e-8:
            rel = f"{(100.0 * d_s['mean'] / abs(st_s['mean'])):+.1f}%"

        rows.append(
            {
                "env": env,
                "p3o_static": fmt_mean_std(st_s),
                "p3o_dynamic": fmt_mean_std(dy_s),
                "dynamic_minus_static": f"{d_s['mean']:+.2f} +/- {d_s['std']:.2f}",
                "dynamic_minus_static_mean": round(d_s["mean"], 6),
                "dynamic_minus_static_std": round(d_s["std"], 6),
                "relative_gain": rel,
                "win_rate": f"{wins}/{n}",
            }
        )
    return rows


def first_step_reach(points, threshold):
    if not points:
        return None
    for s, v in points:
        if v >= threshold:
            return s
    return None


def build_efficiency_table(runs, final_stats):
    # threshold = 80% of best mean final performance in each env
    thresholds = {}
    for env, algo_map in final_stats.items():
        means = [st["mean"] for st in algo_map.values() if st]
        if means:
            thresholds[env] = 0.8 * max(means)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        env = r["env"]
        algo = r["algo"]
        thr = thresholds.get(env)
        if thr is None:
            continue
        step_hit = first_step_reach(r.get("reward_points", []), thr)
        grouped[env][algo].append(
            {
                "step_hit": step_hit,
                "mean_reward_over_time": r.get("mean_reward_over_time"),
                "avg_sps": r.get("avg_sps"),
            }
        )

    rows = []
    for env in ordered_envs(grouped.keys()):
        thr = thresholds[env]
        for algo in ALGO_ORDER:
            items = grouped[env].get(algo, [])
            step_vals = [it["step_hit"] for it in items if it["step_hit"] is not None]
            mean_reward_vals = [it["mean_reward_over_time"] for it in items if it["mean_reward_over_time"] is not None]
            sps_vals = [it["avg_sps"] for it in items if it["avg_sps"] is not None]

            step_stat = calc_stats(step_vals)
            reward_stat = calc_stats(mean_reward_vals)
            sps_stat = calc_stats(sps_vals)

            rows.append(
                {
                    "env": env,
                    "algo": algo,
                    "threshold_avg10": f"{thr:.2f}",
                    "steps_to_threshold": (f"{step_stat['mean']:.0f}" if step_stat else "N/A"),
                    "success_rate": f"{len(step_vals)}/{len(items)}" if items else "0/0",
                    "mean_reward_over_time": (f"{reward_stat['mean']:.2f}" if reward_stat else "N/A"),
                    "mean_sps": (f"{sps_stat['mean']:.1f}" if sps_stat else "N/A"),
                }
            )
    return rows


# ---------------------------
# Save helpers
# ---------------------------
def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_md(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + " --- |" * len(fieldnames) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in fieldnames) + " |\n")


def save_table_png(path, title, rows, fieldnames):
    if not rows:
        return None
    fig_w = max(10, 2.2 * len(fieldnames))
    fig_h = max(2.8, 0.55 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell_text = [[str(r.get(k, "")) for k in fieldnames] for r in rows]
    tbl = ax.table(cellText=cell_text, colLabels=fieldnames, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    ax.set_title(title, pad=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path


# ---------------------------
# Plot helpers
# ---------------------------
def plot_component_bar(final_stats):
    envs = ordered_envs(final_stats.keys())
    if not envs:
        return None

    x = list(range(len(envs)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    for i, algo in enumerate(ALGO_ORDER):
        means = []
        stds = []
        for env in envs:
            st = final_stats[env].get(algo)
            means.append(st["mean"] if st else 0.0)
            stds.append(st["std"] if st else 0.0)
        shift = [xi + (i - 1.5) * width for xi in x]
        ax.bar(
            shift,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=ALGO_COLORS.get(algo),
            alpha=0.9,
            label=algo,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Final Avg10 Reward")
    ax.set_title("Ablation Components: Final Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_component_bar.png")
    plt.savefig(out)
    plt.close(fig)
    return out


def aggregate_line(runs, env, algo, step_gap=5000):
    series = [r for r in runs if r["env"] == env and r["algo"] == algo]
    if not series:
        return None, None

    max_step = max((pts[-1][0] for pts in [s["reward_points"] for s in series] if pts), default=0)
    if max_step <= 0:
        return None, None

    grid = list(range(0, max_step + 1, step_gap))
    all_vals = []
    for s in series:
        pts = s["reward_points"]
        if not pts:
            continue
        idx = 0
        last_v = None
        vals = []
        for g in grid:
            while idx < len(pts) and pts[idx][0] <= g:
                last_v = pts[idx][1]
                idx += 1
            vals.append(last_v if last_v is not None else float("nan"))
        all_vals.append(vals)

    if not all_vals:
        return None, None

    means = []
    for j in range(len(grid)):
        col = [row[j] for row in all_vals if row[j] == row[j]]
        means.append(sum(col) / len(col) if col else float("nan"))
    return grid, means


def plot_alpha_vs_static_curves(runs):
    envs = ordered_envs({r["env"] for r in runs})
    if not envs:
        return None
    cols = 2 if len(envs) > 1 else 1
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.8 * rows), squeeze=False)

    for i, env in enumerate(envs):
        ax = axes[i // cols][i % cols]
        for algo in ["P3O-static", "P3O-dynamic"]:
            x, y = aggregate_line(runs, env, algo, step_gap=5000)
            if not x:
                continue
            ax.plot([v / 1_000_000 for v in x], y, label=algo, color=ALGO_COLORS.get(algo), linewidth=2.2)

        ax.set_title(env)
        ax.set_xlabel("Environment Steps (Million)")
        ax.set_ylabel("Avg10 Reward")
        ax.grid(alpha=0.25)

    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True)
    fig.suptitle("Ablation: Dynamic Alpha vs Static Alpha", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(OUT_DIR, "ablation_alpha_curve.png")
    plt.savefig(out)
    plt.close(fig)
    return out


def plot_alpha_gap_bar(alpha_rows):
    if not alpha_rows:
        return None

    envs = [r["env"] for r in alpha_rows]
    means = [safe_float(r.get("dynamic_minus_static_mean")) or 0.0 for r in alpha_rows]
    stds = [safe_float(r.get("dynamic_minus_static_std")) or 0.0 for r in alpha_rows]
    colors = ["#2ca02c" if m >= 0 else "#d62728" for m in means]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = list(range(len(envs)))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.9, capsize=4)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Dynamic - Static (Final Avg10 Reward)")
    ax.set_title("Ablation: Dynamic Alpha Gain")
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, means):
        y = bar.get_height()
        offset = 2 if y >= 0 else -2
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{val:+.1f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_alpha_gain_bar.png")
    plt.savefig(out)
    plt.close(fig)
    return out


def write_caption_file():
    lines = [
        "Figure A1. Ablation components comparison (PPO, PPO+Cycle, P3O-static, P3O-dynamic) with std error bars.",
        "Figure A2. Dynamic-alpha vs static-alpha reward curves across training steps.",
        "Figure A3. Dynamic-alpha gain bar (dynamic minus static final Avg10 reward, with std).",
        "Table A1. Component ablation: final Avg10 reward (mean +/- std) and incremental deltas.",
        "Table A2. Alpha ablation: static vs dynamic paired comparison and win rate.",
        "Table A3. Efficiency ablation: steps-to-threshold, mean reward-over-time, and SPS.",
    ]
    out = os.path.join(OUT_DIR, "ablation_captions.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def export_paper_bundle():
    mapping = {
        "ablation_component_bar.png": "Fig5_ablation_components.png",
        "ablation_alpha_curve.png": "Fig6_alpha_static_vs_dynamic_curve.png",
        "ablation_alpha_gain_bar.png": "Fig7_alpha_gain_bar.png",
        "ablation_component_table.png": "Table5_component_ablation.png",
        "ablation_alpha_table.png": "Table6_alpha_ablation.png",
        "ablation_efficiency_table.png": "Table7_efficiency_ablation.png",
    }
    exported = []
    for src_name, dst_name in mapping.items():
        src = os.path.join(OUT_DIR, src_name)
        dst = os.path.join(OUT_DIR, dst_name)
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            exported.append(dst)
    return exported


def write_insert_guide():
    out = os.path.join(OUT_DIR, "ablation_insert_guide_cn.md")
    text = """# 消融图表替换清单（可直接放论文）

1. 图5：`Fig5_ablation_components.png`
注释：组件消融结果。比较 PPO、PPO+Cycle、P3O-static、P3O-dynamic 在各环境的最终性能（Final Avg10 Reward），误差线为标准差。

2. 图6：`Fig6_alpha_static_vs_dynamic_curve.png`
注释：动态 α 与静态 α（α=0.4）在训练过程中的奖励曲线对比。横轴为环境交互步数（百万），纵轴为 Avg10 Reward。

3. 图7：`Fig7_alpha_gain_bar.png`
注释：动态 α 相对静态 α 的最终性能增益（Dynamic - Static）。柱体高于0表示动态 α 更优，低于0表示静态 α 更优。

4. 表5：`Table5_component_ablation.png`
注释：组件消融表。给出各方法最终性能均值±标准差及逐步增量（Cycle 相对 PPO、Static 相对 Cycle、Dynamic 相对 Static）。

5. 表6：`Table6_alpha_ablation.png`
注释：α 消融表。给出静态/动态 α 的成对比较、相对增益和胜率（win rate）。

6. 表7：`Table7_efficiency_ablation.png`
注释：训练效率消融。给出达到阈值所需步数、训练过程平均奖励、平均 SPS。阈值定义为该环境最佳最终性能的 80%。
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No valid runs found under:", RUNS_DIR)
        return

    component_rows, final_stats = build_component_table(runs)
    alpha_rows = build_alpha_table(runs)
    efficiency_rows = build_efficiency_table(runs, final_stats)

    c_fields = [
        "env",
        "ppo",
        "ppo_cycle",
        "p3o_static",
        "p3o_dynamic",
        "delta_cycle_vs_ppo",
        "delta_static_vs_cycle",
        "delta_dynamic_vs_static",
    ]
    a_fields = [
        "env",
        "p3o_static",
        "p3o_dynamic",
        "dynamic_minus_static",
        "relative_gain",
        "win_rate",
    ]
    e_fields = [
        "env",
        "algo",
        "threshold_avg10",
        "steps_to_threshold",
        "success_rate",
        "mean_reward_over_time",
        "mean_sps",
    ]

    save_csv(os.path.join(OUT_DIR, "ablation_component_table.csv"), component_rows, c_fields)
    save_md(os.path.join(OUT_DIR, "ablation_component_table.md"), component_rows, c_fields)
    save_table_png(
        os.path.join(OUT_DIR, "ablation_component_table.png"),
        "Table A1. Component Ablation",
        component_rows,
        c_fields,
    )

    save_csv(os.path.join(OUT_DIR, "ablation_alpha_table.csv"), alpha_rows, a_fields)
    save_md(os.path.join(OUT_DIR, "ablation_alpha_table.md"), alpha_rows, a_fields)
    save_table_png(
        os.path.join(OUT_DIR, "ablation_alpha_table.png"),
        "Table A2. Dynamic vs Static Alpha",
        alpha_rows,
        a_fields,
    )

    save_csv(os.path.join(OUT_DIR, "ablation_efficiency_table.csv"), efficiency_rows, e_fields)
    save_md(os.path.join(OUT_DIR, "ablation_efficiency_table.md"), efficiency_rows, e_fields)
    save_table_png(
        os.path.join(OUT_DIR, "ablation_efficiency_table.png"),
        "Table A3. Efficiency Ablation",
        efficiency_rows,
        e_fields,
    )

    fig1 = plot_component_bar(final_stats)
    fig2 = plot_alpha_vs_static_curves(runs)
    fig3 = plot_alpha_gap_bar(alpha_rows)
    cap = write_caption_file()
    exported = export_paper_bundle()
    guide = write_insert_guide()

    print("Saved ablation package to:", OUT_DIR)
    print("Figure:", fig1)
    print("Figure:", fig2)
    print("Figure:", fig3)
    print("Table: ablation_component_table.(csv/md/png)")
    print("Table: ablation_alpha_table.(csv/md/png)")
    print("Table: ablation_efficiency_table.(csv/md/png)")
    print("Captions:", cap)
    print("Paper bundle files:", len(exported))
    print("Insert guide:", guide)


if __name__ == "__main__":
    main()
