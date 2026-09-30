import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")
OUT_DIR = os.path.join(os.getcwd(), "innovation_ready", "mechanism_plots")
ALGO_PRIORITY = [
    "P3O-ClosedLoopFull",
    "P3O-ClosedLoopAlpha",
    "P3O-dyn+EvtSBP+AdaReset+HardDistill",
    "P3O-dyn+AdaReset+HardDistill",
    "P3O-dynamic",
]
COLORS = {
    "reward": "#1f77b4",
    "alpha": "#d62728",
    "alpha_dual": "#ff9896",
    "reset": "#2ca02c",
    "signal": "#9467bd",
    "event": "#ff7f0e",
    "force": "#8c564b",
}


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


def parse_sbp(row):
    if len(row) < 25 or row[1] != "sbp":
        return None
    return {
        "step": safe_float(row[0]),
        "alpha_using": safe_float(row[4]),
        "algo": row[5],
        "env": row[6],
        "seed": row[7],
        "reset_rate": safe_float(row[8]),
        "combined_signal": safe_float(row[19]) if len(row) > 19 else None,
        "trigger_mode": row[24] if len(row) > 24 else "unknown",
        "observed_kl": safe_float(row[25]) if len(row) > 25 else None,
        "target_kl": safe_float(row[26]) if len(row) > 26 else None,
        "alpha_dual_state": safe_float(row[27]) if len(row) > 27 else None,
        "alpha_dual_gap": safe_float(row[28]) if len(row) > 28 else None,
        "alpha_base": safe_float(row[29]) if len(row) > 29 else None,
        "pre_forward_kl": safe_float(row[30]) if len(row) > 30 else None,
        "pre_backward_kl": safe_float(row[31]) if len(row) > 31 else None,
        "pre_symmetric_kl": safe_float(row[32]) if len(row) > 32 else None,
        "post_forward_kl": safe_float(row[33]) if len(row) > 33 else None,
        "post_backward_kl": safe_float(row[34]) if len(row) > 34 else None,
        "post_symmetric_kl": safe_float(row[35]) if len(row) > 35 else None,
        "selective_reset": safe_float(row[36]) if len(row) > 36 else None,
        "reset_importance": safe_float(row[37]) if len(row) > 37 else None,
        "preserve_importance": safe_float(row[38]) if len(row) > 38 else None,
        "reset_count": safe_float(row[39]) if len(row) > 39 else None,
    }


def parse_reward(row):
    if len(row) < 5 or row[1] != "reward":
        return None
    if len(row) >= 8:
        return {
            "step": safe_float(row[0]),
            "episode_reward": safe_float(row[3]),
            "avg10": safe_float(row[4]),
            "algo": row[5] if len(row) > 5 else None,
            "env": row[6] if len(row) > 6 else None,
            "seed": row[7] if len(row) > 7 else None,
        }
    if len(row) >= 6:
        return {
            "step": safe_float(row[0]),
            "episode_reward": safe_float(row[4]) if len(row) > 4 else None,
            "avg10": safe_float(row[5]) if len(row) > 5 else None,
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
        rewards = []
        sbp = []
        with open(os.path.join(root, "metrics.csv"), "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                r = parse_reward(row)
                if r and r.get("step") is not None:
                    rewards.append(r)
                s = parse_sbp(row)
                if s and s.get("step") is not None:
                    sbp.append(s)
        if rewards or sbp:
            runs.append({**meta, "root": root, "rewards": rewards, "sbp": sbp})
    return runs


def choose_focus_runs(runs):
    if not runs:
        return []
    by_algo = defaultdict(list)
    for run in runs:
        by_algo[run["algo"]].append(run)
    for algo in ALGO_PRIORITY:
        if algo in by_algo:
            chosen = by_algo[algo]
            chosen.sort(key=lambda x: (x["env"], x["seed"]))
            return chosen
    return sorted(runs, key=lambda x: (x["algo"], x["env"], x["seed"]))


def plot_alignment(run):
    rewards = run["rewards"]
    sbp = run["sbp"]
    if not rewards and not sbp:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True)
    reward_x = [r["step"] / 1000 for r in rewards if r.get("avg10") is not None]
    reward_y = [r["avg10"] for r in rewards if r.get("avg10") is not None]
    if reward_x:
        axes[0].plot(reward_x, reward_y, color=COLORS["reward"], linewidth=2.2)
    axes[0].set_ylabel("Avg10 Reward")
    axes[0].grid(alpha=0.25)
    axes[0].set_title(f"Mechanism Alignment: {run['env']} | {run['algo']} | seed {run['seed']}")

    sbp_x = [x["step"] / 1000 for x in sbp if x.get("alpha_using") is not None]
    alpha_y = [x["alpha_using"] for x in sbp if x.get("alpha_using") is not None]
    base_rows = [x for x in sbp if x.get("alpha_base") is not None]
    dual_rows = [x for x in sbp if x.get("alpha_dual_state") is not None]
    base_x = [x["step"] / 1000 for x in base_rows]
    base_y = [x["alpha_base"] for x in base_rows]
    dual_x = [x["step"] / 1000 for x in dual_rows]
    dual_y = [x["alpha_dual_state"] for x in dual_rows]
    if base_x and base_y:
        axes[1].plot(base_x, base_y, color="#7f7f7f", linewidth=1.6, linestyle=":", label="alpha_base")
    if sbp_x and alpha_y:
        axes[1].plot(sbp_x, alpha_y, color=COLORS["alpha"], linewidth=2.0, label="alpha_using")
    if dual_x and dual_y:
        axes[1].plot(dual_x, dual_y, color=COLORS["alpha_dual"], linewidth=1.8, linestyle="--", label="alpha_dual")
    observed_rows = [x for x in sbp if x.get("observed_kl") is not None]
    if observed_rows:
        obs_x = [x["step"] / 1000 for x in observed_rows]
        obs_y = [x["observed_kl"] for x in observed_rows]
        ax2 = axes[1].twinx()
        ax2.plot(obs_x, obs_y, color="#7f7f7f", linewidth=1.5, alpha=0.75, label="observed_kl")
        target = observed_rows[0].get("target_kl")
        if target is not None:
            ax2.axhline(target, color="#7f7f7f", linestyle=":", linewidth=1.2, label="target_kl")
        ax2.set_ylabel("KL")
    axes[1].set_ylabel("Alpha")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="upper right")

    reset_rows = [x for x in sbp if x.get("reset_rate") is not None]
    signal_rows = [x for x in sbp if x.get("combined_signal") is not None]
    if reset_rows:
        axes[2].plot([x["step"] / 1000 for x in reset_rows], [x["reset_rate"] for x in reset_rows], color=COLORS["reset"], linewidth=2.0, label="reset_rate")
    if signal_rows:
        ax3 = axes[2].twinx()
        ax3.plot([x["step"] / 1000 for x in signal_rows], [x["combined_signal"] for x in signal_rows], color=COLORS["signal"], linewidth=1.8, linestyle="--", label="combined_signal")
        ax3.axhline(0.35, color=COLORS["signal"], linestyle=":", linewidth=1.1)
        post_kl_rows = [x for x in sbp if x.get("post_symmetric_kl") is not None]
        if post_kl_rows:
            ax3.plot(
                [x["step"] / 1000 for x in post_kl_rows],
                [x["post_symmetric_kl"] for x in post_kl_rows],
                color="#bc80bd",
                linewidth=1.5,
                alpha=0.8,
                label="post_kl",
            )
        ax3.set_ylabel("Signal")
    axes[2].set_ylabel("Reset Rate")
    axes[2].grid(alpha=0.25)

    mode_to_y = {"event": 1, "force": 2, "periodic": 3}
    mode_to_color = {"event": COLORS["event"], "force": COLORS["force"], "periodic": "#17becf"}
    plotted = set()
    for row in sbp:
        mode = row.get("trigger_mode", "unknown")
        y = mode_to_y.get(mode, 0)
        color = mode_to_color.get(mode, "#7f7f7f")
        label = mode if mode not in plotted else None
        axes[3].scatter(row["step"] / 1000, y, color=color, s=26, label=label)
        plotted.add(mode)
    imp_rows = [x for x in sbp if x.get("reset_importance") is not None and x.get("preserve_importance") is not None]
    if imp_rows:
        gap_ax = axes[3].twinx()
        gap_ax.plot(
            [x["step"] / 1000 for x in imp_rows],
            [x["preserve_importance"] - x["reset_importance"] for x in imp_rows],
            color="#8c564b",
            linewidth=1.6,
            alpha=0.75,
            label="importance_gap",
        )
        gap_ax.set_ylabel("Importance Gap")
    axes[3].set_xlabel("Training Step (x1e3)")
    axes[3].set_ylabel("Trigger")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(["other", "event", "force", "periodic"])
    axes[3].grid(alpha=0.20)
    if plotted:
        axes[3].legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(
        OUT_DIR,
        f"mechanism_alignment_{run['env']}_{run['algo']}_seed{run['seed']}_steps{run['steps']}.png",
    )
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_trigger_summary(runs):
    if not runs:
        return None
    counts = defaultdict(Counter)
    for run in runs:
        for row in run["sbp"]:
            counts[run["algo"]][row.get("trigger_mode", "unknown")] += 1
    algos = list(counts.keys())
    if not algos:
        return None
    modes = ["event", "force", "periodic", "unknown"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bottoms = [0] * len(algos)
    for mode in modes:
        vals = [counts[a][mode] for a in algos]
        ax.bar(algos, vals, bottom=bottoms, label=mode)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("SBP Trigger Count")
    ax.set_title("SBP Trigger Composition Across Algorithms")
    ax.tick_params(axis="x", rotation=18)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "mechanism_trigger_summary.png")
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_mean_signal_vs_reward(runs):
    groups = defaultdict(lambda: {"reward": [], "signal": []})
    for run in runs:
        reward_vals = [r["avg10"] for r in run["rewards"] if r.get("avg10") is not None]
        signal_vals = [s["combined_signal"] for s in run["sbp"] if s.get("combined_signal") is not None]
        if reward_vals:
            groups[run["algo"]]["reward"].append(reward_vals[-1])
        if signal_vals:
            groups[run["algo"]]["signal"].append(sum(signal_vals) / len(signal_vals))
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for algo, vals in groups.items():
        if not vals["reward"] or not vals["signal"]:
            continue
        x = sum(vals["signal"]) / len(vals["signal"])
        y = sum(vals["reward"]) / len(vals["reward"])
        ax.scatter(x, y, s=90, label=algo)
        ax.text(x, y, algo, fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Mean Combined Signal")
    ax.set_ylabel("Final Avg10 Reward")
    ax.set_title("Mechanism Signal vs Final Performance")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "mechanism_signal_vs_reward.png")
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs found in", RUNS_DIR)
        return
    focus_runs = choose_focus_runs(runs)
    saved = []
    for run in focus_runs[:4]:
        path = plot_alignment(run)
        if path:
            saved.append(path)
    saved.append(plot_trigger_summary(runs))
    saved.append(plot_mean_signal_vs_reward(runs))
    print("Saved mechanism plots to:", OUT_DIR)
    for path in saved:
        if path:
            print(path)


if __name__ == "__main__":
    main()
