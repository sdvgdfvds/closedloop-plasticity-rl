import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
OUT_DIR = os.path.join(os.getcwd(), "innovation_ready")
ALGO_ORDER = [
    "P3O-dynamic",
    "P3O-dyn+AdaReset",
    "P3O-dyn+HardDistill",
    "P3O-dyn+AdaReset+HardDistill",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill",
    "P3O-ClosedLoopAlpha",
    "P3O-ClosedLoopFull",
]
ALGO_LABELS = {
    "P3O-dynamic": "Dyn",
    "P3O-dyn+AdaReset": "Dyn+AR",
    "P3O-dyn+HardDistill": "Dyn+HD",
    "P3O-dyn+AdaReset+HardDistill": "Dyn+AR+HD",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill": "Dyn+Evt+AR+HD",
    "P3O-ClosedLoopAlpha": "ClosedLoop-A",
    "P3O-ClosedLoopFull": "ClosedLoop-Full",
}
COLORS = {
    "P3O-dynamic": "#1f77b4",
    "P3O-dyn+AdaReset": "#ff7f0e",
    "P3O-dyn+HardDistill": "#9467bd",
    "P3O-dyn+AdaReset+HardDistill": "#2ca02c",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill": "#d62728",
    "P3O-ClosedLoopAlpha": "#8c564b",
    "P3O-ClosedLoopFull": "#17becf",
}
_STEP_FILTER_RAW = os.environ.get("P3O_STEP_FILTER", "500000").strip()
STEP_FILTER = {int(x) for x in _STEP_FILTER_RAW.split(",") if x.strip()} if _STEP_FILTER_RAW else set()
ENV_ORDER = ["Hopper-v4", "Walker2d-v4", "Ant-v4", "HalfCheetah-v4", "Humanoid-v4"]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_folder(folder):
    if "_seed" not in folder or "_steps" not in folder:
        return None
    try:
        prefix, tail = folder.rsplit("_seed", 1)
        seed, steps = tail.split("_steps", 1)
        env, algo = prefix.split("_", 1)
        return {"env": env, "algo": algo, "seed": seed, "steps": int(steps)}
    except Exception:
        return None


def parse_row(row):
    if not row or len(row) < 2:
        return None
    if row[0] == "step" and row[1] == "tag":
        return None
    tag = row[1]

    def g(i):
        return row[i].strip() if i < len(row) else ""

    if tag == "reward" and len(row) >= 8:
        return {"step": g(0), "tag": tag, "avg10": g(4), "algo": g(5), "env": g(6), "seed": g(7)}
    if tag == "speed" and len(row) >= 7:
        return {"step": g(0), "tag": tag, "sps": g(2), "algo": g(4), "env": g(5), "seed": g(6)}
    if tag == "sbp" and len(row) >= 8:
        # extended fields may exist from index>=8
        return {
            "step": g(0),
            "tag": tag,
            "alpha_using": g(4),
            "algo": g(5),
            "env": g(6),
            "seed": g(7),
            "reset_rate": g(8),
        }
    return None


def load_runs():
    runs = []
    for root, _, files in os.walk(RUNS_DIR):
        if "metrics.csv" not in files:
            continue
        meta = parse_folder(os.path.basename(root))
        if not meta:
            continue
        if meta["algo"] not in ALGO_ORDER:
            continue
        if STEP_FILTER and meta["steps"] not in STEP_FILTER:
            continue

        rows = []
        with open(os.path.join(root, "metrics.csv"), "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                p = parse_row(row)
                if p:
                    rows.append(p)
        if not rows:
            continue
        runs.append({"env": meta["env"], "algo": meta["algo"], "seed": meta["seed"], "steps": meta["steps"], "rows": rows})
    return runs


def ordered_envs(envs):
    pri = [e for e in ENV_ORDER if e in envs]
    rest = sorted([e for e in envs if e not in pri])
    return pri + rest


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    m = sum(vals) / n
    v = sum((x - m) ** 2 for x in vals) / n
    s = math.sqrt(v)
    return m, s


def build_table(runs):
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        last_avg10 = None
        sps_vals = []
        for r in run["rows"]:
            if r["tag"] == "reward":
                v = safe_float(r.get("avg10"))
                if v is not None:
                    last_avg10 = v
            elif r["tag"] == "speed":
                sp = safe_float(r.get("sps"))
                if sp is not None:
                    sps_vals.append(sp)
        mean_sps = (sum(sps_vals) / len(sps_vals)) if sps_vals else None
        grouped[run["env"]][run["algo"]].append((last_avg10, mean_sps))

    rows = []
    for env in ordered_envs(grouped.keys()):
        row = {"env": env}
        for algo in ALGO_ORDER:
            vals = [x[0] for x in grouped[env].get(algo, []) if x[0] is not None]
            sps = [x[1] for x in grouped[env].get(algo, []) if x[1] is not None]
            st = stat(vals)
            st_sps = stat(sps)
            k = ALGO_LABELS[algo].lower().replace("+", "_plus_")
            row[f"{k}_reward"] = f"{st[0]:.2f} +/- {st[1]:.2f}" if st else "N/A"
            row[f"{k}_sps"] = f"{st_sps[0]:.1f}" if st_sps else "N/A"
        rows.append(row)
    return rows


def save_table(rows):
    fields = ["env"]
    pretty_fields = ["Env"]
    for algo in ALGO_ORDER:
        label = ALGO_LABELS[algo]
        key = label.lower().replace("+", "_plus_")
        fields.append(f"{key}_reward")
        pretty_fields.append(f"{label} Reward")
    for algo in ALGO_ORDER:
        label = ALGO_LABELS[algo]
        key = label.lower().replace("+", "_plus_")
        fields.append(f"{key}_sps")
        pretty_fields.append(f"{label} SPS")

    csv_path = os.path.join(OUT_DIR, "innovation_table.csv")
    md_path = os.path.join(OUT_DIR, "innovation_table.md")
    png_path = os.path.join(OUT_DIR, "innovation_table.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(pretty_fields) + " |\n")
        f.write("|" + " --- |" * len(pretty_fields) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")

    fig_w = max(15.5, 1.55 * len(fields))
    fig_h = max(3.2, 0.9 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell = [[str(r.get(k, "")) for k in fields] for r in rows]
    col_widths = [0.12] + [0.12] * len(ALGO_ORDER) + [0.09] * len(ALGO_ORDER)
    tbl = ax.table(
        cellText=cell,
        colLabels=pretty_fields,
        colWidths=col_widths,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax.set_title("Innovation Comparison Table")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    return csv_path, md_path, png_path


def aggregate_mean_curve(runs, env, algo, tag, key, step_gap=5000):
    selected = []
    for run in runs:
        if run["env"] != env or run["algo"] != algo:
            continue
        pts = []
        for r in run["rows"]:
            if r["tag"] != tag:
                continue
            s = safe_float(r.get("step"))
            v = safe_float(r.get(key))
            if s is None or v is None:
                continue
            pts.append((int(s), v))
        if pts:
            pts.sort(key=lambda x: x[0])
            selected.append(pts)
    if not selected:
        return None, None

    max_step = max(pts[-1][0] for pts in selected)
    grid = list(range(0, max_step + 1, step_gap))
    all_vals = []
    for pts in selected:
        idx = 0
        last = None
        vals = []
        for g in grid:
            while idx < len(pts) and pts[idx][0] <= g:
                last = pts[idx][1]
                idx += 1
            vals.append(last if last is not None else float("nan"))
        all_vals.append(vals)

    means = []
    for j in range(len(grid)):
        col = [row[j] for row in all_vals if row[j] == row[j]]
        means.append(sum(col) / len(col) if col else float("nan"))

    return grid, means


def plot_reward(runs):
    envs = ordered_envs({r["env"] for r in runs})
    if not envs:
        return None
    cols = 2 if len(envs) > 1 else 1
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.2 * cols, 4.2 * rows), squeeze=False)

    last_idx = -1
    for i, env in enumerate(envs):
        last_idx = i
        ax = axes[i // cols][i % cols]
        for algo in ALGO_ORDER:
            x, y = aggregate_mean_curve(runs, env, algo, "reward", "avg10", step_gap=5000)
            if x is None:
                continue
            ax.plot(
                [v / 1_000_000 for v in x],
                y,
                label=ALGO_LABELS[algo],
                color=COLORS.get(algo),
                linewidth=2.2,
            )
        ax.set_title(env)
        ax.set_xlabel("Env Steps (Million)")
        ax.set_ylabel("Avg10 Reward")
        ax.grid(alpha=0.25)

    for j in range(last_idx + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=min(len(l), 5), frameon=True)
    fig.suptitle("Innovation Reward Curves", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(OUT_DIR, "innovation_reward_curve.png")
    plt.savefig(path)
    plt.close(fig)
    return path


def plot_reset_rate(runs):
    envs = ordered_envs({r["env"] for r in runs})
    if not envs:
        return None
    cols = 2 if len(envs) > 1 else 1
    rows = (len(envs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.2 * cols, 4.2 * rows), squeeze=False)

    tracked_algos = ALGO_ORDER
    last_idx = -1
    for i, env in enumerate(envs):
        last_idx = i
        ax = axes[i // cols][i % cols]
        for algo in tracked_algos:
            x, y = aggregate_mean_curve(runs, env, algo, "sbp", "reset_rate", step_gap=10000)
            if x is None:
                continue
            ax.plot(
                [v / 1_000_000 for v in x],
                y,
                label=ALGO_LABELS[algo],
                color=COLORS.get(algo),
                linewidth=2.2,
            )
        ax.set_title(env)
        ax.set_xlabel("Env Steps (Million)")
        ax.set_ylabel("Reset Rate")
        ax.grid(alpha=0.25)

    for j in range(last_idx + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=min(len(l), 5), frameon=True)
    fig.suptitle("Adaptive Reset Rate Curves", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(OUT_DIR, "innovation_reset_rate_curve.png")
    plt.savefig(path)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No innovation runs found.")
        return

    table_rows = build_table(runs)
    t_csv, t_md, t_png = save_table(table_rows)
    f1 = plot_reward(runs)
    f2 = plot_reset_rate(runs)

    print("Saved to:", OUT_DIR)
    print("Table CSV:", t_csv)
    print("Table MD:", t_md)
    print("Table PNG:", t_png)
    print("Figure reward:", f1)
    print("Figure reset:", f2)


if __name__ == "__main__":
    main()
