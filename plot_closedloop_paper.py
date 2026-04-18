"""Generate publication-quality figures & tables for the ClosedLoop paper.

Outputs PNG (300 dpi) + PDF + CSV to {RUNS_DIR}/closedloop_figures/.
Requires: matplotlib, numpy, scipy.
"""
import csv
import os
import glob
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

# ─── Paths ───
RUNS_DIR = r"C:\Users\33277\Desktop\p3o_runs"
FIG_DIR = os.path.join(RUNS_DIR, "closedloop_figures")
os.makedirs(FIG_DIR, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]
SEEDS = list(range(5))
STEPS = 500_000

# ─── Algorithm groups ───
MAIN_ALGOS = {
    "PPO":        {"dir": "PPO",       "label": "PPO",       "color": "#1f77b4", "ls": "-"},
    "PPO+Cycle":  {"dir": "PPO+Cycle", "label": "PPO+Cycle", "color": "#aec7e8", "ls": "--"},
    "P3O":        {"dir": "P3O",       "label": "P3O",       "color": "#ff7f0e", "ls": "-"},
    "EvtSBP":     {"dir": "P3O-dyn+EvtSBP+AdaReset+HardDistill",
                   "label": "EvtSBP+AdaReset+HardDistill", "color": "#2ca02c", "ls": "--"},
    "CLAlpha":    {"dir": "P3O-ClosedLoopAlpha",
                   "label": "ClosedLoopAlpha", "color": "#9467bd", "ls": "-"},
    "CLFull":     {"dir": "P3O-ClosedLoopFull",
                   "label": "ClosedLoopFull",  "color": "#d62728", "ls": "-"},
}

ABLATION_ALGOS = {
    "Dynamic":              {"dir": "P3O-dynamic",     "label": "Dynamic"},
    "Dynamic+AdaReset":     {"dir": "P3O-dyn+AdaReset","label": "Dyn+AdaReset"},
    "Dynamic+HardDistill":  {"dir": "P3O-dyn+HardDistill","label": "Dyn+HardDistill"},
    "EvtSBP":               {"dir": "P3O-dyn+EvtSBP+AdaReset+HardDistill",
                              "label": "EvtSBP+AdaReset+HardDistill"},
    "CLAlpha":              {"dir": "P3O-ClosedLoopAlpha","label": "ClosedLoopAlpha"},
    "CLFull":               {"dir": "P3O-ClosedLoopFull", "label": "ClosedLoopFull"},
}

SEQ_ALGOS = {
    "PPO":      {"dir": "PPO",                  "label": "PPO",              "color": "#1f77b4"},
    "P3O":      {"dir": "P3O",                  "label": "P3O",              "color": "#ff7f0e"},
    "CLFull":   {"dir": "P3O-ClosedLoopFull",   "label": "ClosedLoopFull",   "color": "#d62728"},
    "CLMemory": {"dir": "P3O-ClosedLoopMemory", "label": "ClosedLoopMemory", "color": "#2ca02c"},
}

# ─── Matplotlib global style ───
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

FULL_W = 7.0


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_reward_curve(algo_dir, env, seed, steps=STEPS):
    """Load (step[], avg10[]) from metrics.csv for single-task."""
    path = os.path.join(RUNS_DIR,
                        f"{env}_{algo_dir}_seed{seed}_steps{steps}",
                        "metrics.csv")
    if not os.path.exists(path):
        return None, None
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[1] == "reward":
                xs.append(int(row[0]))
                ys.append(float(row[4]))  # avg10
    if not xs:
        return None, None
    return np.array(xs), np.array(ys)


def load_final_reward(algo_dir, env, seed, steps=STEPS):
    """Return final avg10 value."""
    xs, ys = load_reward_curve(algo_dir, env, seed, steps)
    if ys is None or len(ys) == 0:
        return None
    return float(ys[-1])


def load_sequence_phase_summary(algo_dir, seq_str, seed, steps_per_phase=80000):
    """Load phase_summary rows from sequence run.
    Returns list of dicts with keys: phase, env, current_a, reference_a, retention, forgetting.
    """
    path = os.path.join(RUNS_DIR,
                        f"sequence_{algo_dir}_{seq_str}_seed{seed}_steps{steps_per_phase}",
                        "metrics.csv")
    if not os.path.exists(path):
        return None
    summaries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 9 and row[1] == "phase_summary":
                summaries.append({
                    "step": int(row[0]),
                    "phase": int(row[2]),
                    "env": row[3],
                    "current_a": float(row[4]),
                    "reference_a": float(row[5]),
                    "retention": float(row[6]),
                    "forgetting": float(row[7]),
                })
    return summaries if summaries else None


def load_sequence_eval(algo_dir, seq_str, seed, steps_per_phase=80000):
    """Load eval rows: step, phase, train_env, eval_env, reward, retention, forgetting."""
    path = os.path.join(RUNS_DIR,
                        f"sequence_{algo_dir}_{seq_str}_seed{seed}_steps{steps_per_phase}",
                        "metrics.csv")
    if not os.path.exists(path):
        return None
    evals = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 8 and row[1] == "eval":
                evals.append({
                    "step": int(row[0]),
                    "phase": int(row[2]),
                    "train_env": row[3],
                    "eval_env": row[4],
                    "reward": float(row[5]),
                    "retention": float(row[6]),
                    "forgetting": float(row[7]),
                })
    return evals if evals else None


def load_sbp_alpha(algo_dir, env, seed, steps=STEPS):
    """Load alpha trajectory from sbp log rows."""
    path = os.path.join(RUNS_DIR,
                        f"{env}_{algo_dir}_seed{seed}_steps{steps}",
                        "metrics.csv")
    if not os.path.exists(path):
        return None, None
    xs, alphas = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[1] == "sbp":
                xs.append(int(row[0]))
                alphas.append(float(row[4]))  # alpha_using
    if not xs:
        return None, None
    return np.array(xs), np.array(alphas)


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def smooth(y, window=20):
    if len(y) <= window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def interp_to_common(steps_list, vals_list, n_points=200):
    """Interpolate multiple (steps, vals) to common x-axis."""
    all_max = max(s[-1] for s in steps_list if len(s) > 0)
    common_x = np.linspace(0, all_max, n_points)
    interped = []
    for sx, vy in zip(steps_list, vals_list):
        interped.append(np.interp(common_x, sx, vy))
    return common_x, np.array(interped)


def sign_flip_test(a, b, n_perm=10000):
    """Paired sign-flip permutation test. Returns p-value (two-sided)."""
    diff = np.array(a) - np.array(b)
    n = len(diff)
    obs = np.abs(np.mean(diff))
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if np.abs(np.mean(diff * signs)) >= obs:
            count += 1
    return count / n_perm


def welch_ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return p


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def save_csv(rows, name):
    path = os.path.join(FIG_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"  Saved {name}.csv")


# ═══════════════════════════════════════════════════════════════
# Fig 2: Single-task learning curves (2x2, 4 envs)
# ═══════════════════════════════════════════════════════════════

def plot_learning_curves():
    print("Fig2: Learning curves...")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.0))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        for key, info in MAIN_ALGOS.items():
            steps_list, vals_list = [], []
            for seed in SEEDS:
                xs, ys = load_reward_curve(info["dir"], env, seed)
                if xs is not None:
                    steps_list.append(xs)
                    vals_list.append(smooth(ys, window=30))
            if not steps_list:
                continue
            common_x, interped = interp_to_common(steps_list, vals_list, n_points=300)
            mean = interped.mean(axis=0)
            std = interped.std(axis=0)
            ax.plot(common_x / 1000, mean, label=info["label"],
                    color=info["color"], ls=info["ls"], linewidth=1.2)
            ax.fill_between(common_x / 1000, mean - std, mean + std,
                            alpha=0.15, color=info["color"])

        ax.set_title(env.replace("-v4", ""))
        ax.set_xlabel("Steps (k)")
        ax.set_ylabel("Avg Return (10 ep)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.8, ncol=1, fontsize=6)

    fig.tight_layout()
    save_fig(fig, "fig02_learning_curves")


# ═══════════════════════════════════════════════════════════════
# Fig 3: Main sequence eval curves (Hopper→Walker2d→Hopper)
# ═══════════════════════════════════════════════════════════════

def plot_main_sequence():
    print("Fig3: Main sequence eval curves...")
    seq_str = "Hopper-v4_to_Walker2d-v4_to_Hopper-v4"
    first_env = "Hopper-v4"

    fig, ax = plt.subplots(1, 1, figsize=(FULL_W, 3.5))

    for key, info in SEQ_ALGOS.items():
        all_steps = []
        all_rewards = []
        n_seeds = 20 if key in ("CLFull", "CLMemory") else 10
        for seed in range(n_seeds):
            evals = load_sequence_eval(info["dir"], seq_str, seed)
            if evals is None:
                continue
            # Extract eval of first_env across all phases
            xs = [e["step"] for e in evals if e["eval_env"] == first_env]
            ys = [e["reward"] for e in evals if e["eval_env"] == first_env]
            if xs:
                all_steps.append(np.array(xs))
                all_rewards.append(np.array(ys))
        if not all_steps:
            continue
        common_x, interped = interp_to_common(all_steps, all_rewards, n_points=100)
        mean = interped.mean(axis=0)
        std = interped.std(axis=0)
        ax.plot(common_x / 1000, mean, label=info["label"],
                color=info["color"], linewidth=1.5)
        ax.fill_between(common_x / 1000, mean - std, mean + std,
                        alpha=0.15, color=info["color"])

    # Phase boundaries
    for boundary in [80, 160]:
        ax.axvline(boundary, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(40, ax.get_ylim()[1]*0.95, "Phase 1\nHopper", ha="center", fontsize=6, color="gray")
    ax.text(120, ax.get_ylim()[1]*0.95, "Phase 2\nWalker2d", ha="center", fontsize=6, color="gray")
    ax.text(200, ax.get_ylim()[1]*0.95, "Phase 3\nHopper", ha="center", fontsize=6, color="gray")

    ax.set_xlabel("Total Steps (k)")
    ax.set_ylabel("Hopper-v4 Eval Return")
    ax.set_title("Hopper→Walker2d→Hopper (Eval on Hopper)")
    ax.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, "fig03_main_sequence")


# ═══════════════════════════════════════════════════════════════
# Fig 4: Retention & Forgetting bar chart (main sequence)
# ═══════════════════════════════════════════════════════════════

def plot_retention_forgetting():
    print("Fig4: Retention & forgetting bars...")
    seq_str = "Hopper-v4_to_Walker2d-v4_to_Hopper-v4"

    algo_data = {}
    for key, info in SEQ_ALGOS.items():
        n_seeds = 20 if key in ("CLFull", "CLMemory") else 10
        revisits, retentions, forgettings = [], [], []
        for seed in range(n_seeds):
            summaries = load_sequence_phase_summary(info["dir"], seq_str, seed)
            if summaries is None:
                continue
            phase3 = [s for s in summaries if s["phase"] == 3]
            if phase3:
                revisits.append(phase3[0]["current_a"])
                retentions.append(phase3[0]["retention"])
                forgettings.append(phase3[0]["forgetting"])
        algo_data[key] = {
            "revisit": np.array(revisits),
            "retention": np.array(retentions),
            "forgetting": np.array(forgettings),
            "label": info["label"],
            "color": info["color"],
        }

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0))

    metrics = [("revisit", "Revisit Reward"), ("retention", "Retention"),
               ("forgetting", "Forgetting")]
    keys_order = list(SEQ_ALGOS.keys())

    for ax, (metric, ylabel) in zip(axes, metrics):
        x_pos = np.arange(len(keys_order))
        for i, key in enumerate(keys_order):
            d = algo_data[key]
            vals = d[metric]
            if len(vals) == 0:
                continue
            bar = ax.bar(i, np.mean(vals), yerr=np.std(vals),
                         color=d["color"], capsize=3, width=0.6, alpha=0.85,
                         error_kw={"linewidth": 0.8})
        ax.set_xticks(x_pos)
        ax.set_xticklabels([algo_data[k]["label"] for k in keys_order],
                           rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)

    fig.suptitle("Hopper→Walker2d→Hopper: Phase 3 Summary", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig04_retention_forgetting")


# ═══════════════════════════════════════════════════════════════
# Fig 5: Ablation bar chart (Hopper + Walker2d)
# ═══════════════════════════════════════════════════════════════

def plot_ablation_bars():
    print("Fig5: Ablation bars...")
    abl_envs = ["Hopper-v4", "Walker2d-v4"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(ABLATION_ALGOS)))

    for ax, env in zip(axes, abl_envs):
        names, means, stds = [], [], []
        for i, (key, info) in enumerate(ABLATION_ALGOS.items()):
            vals = []
            for seed in SEEDS:
                v = load_final_reward(info["dir"], env, seed)
                if v is not None:
                    vals.append(v)
            if vals:
                names.append(info["label"])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                names.append(info["label"])
                means.append(0)
                stds.append(0)

        x = np.arange(len(names))
        ax.barh(x, means, xerr=stds, color=colors[:len(names)],
                capsize=3, height=0.6, alpha=0.85, error_kw={"linewidth": 0.8})
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Final Avg Return")
        ax.set_title(env.replace("-v4", ""))
        ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "fig05_ablation_bars")


# ═══════════════════════════════════════════════════════════════
# Fig 6: Alpha trajectory
# ═══════════════════════════════════════════════════════════════

def plot_alpha_trajectory():
    print("Fig6: Alpha trajectory...")
    alpha_algos = {
        "P3O": {"dir": "P3O-dynamic", "label": "P3O (dynamic schedule)", "color": "#ff7f0e"},
        "CLAlpha": {"dir": "P3O-ClosedLoopAlpha", "label": "ClosedLoopAlpha", "color": "#9467bd"},
        "CLFull": {"dir": "P3O-ClosedLoopFull", "label": "ClosedLoopFull", "color": "#d62728"},
    }
    envs_alpha = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8))

    for ax, env in zip(axes, envs_alpha):
        for key, info in alpha_algos.items():
            all_xs, all_ys = [], []
            for seed in SEEDS:
                xs, ys = load_sbp_alpha(info["dir"], env, seed)
                if xs is not None:
                    all_xs.append(xs)
                    all_ys.append(ys)
            if not all_xs:
                continue
            # Just plot seed 0 for clarity, with others as thin lines
            for i, (sx, sy) in enumerate(zip(all_xs, all_ys)):
                if i == 0:
                    ax.plot(sx / 1000, sy, label=info["label"],
                            color=info["color"], linewidth=1.2)
                else:
                    ax.plot(sx / 1000, sy, color=info["color"],
                            linewidth=0.3, alpha=0.3)

        ax.set_xlabel("Steps (k)")
        ax.set_ylabel("α")
        ax.set_title(env.replace("-v4", ""))
        ax.set_ylim(0, 0.75)
        if env == envs_alpha[0]:
            ax.legend(fontsize=5.5, loc="upper right")

    fig.tight_layout()
    save_fig(fig, "fig06_alpha_trajectory")


# ═══════════════════════════════════════════════════════════════
# Fig 7: Supplementary sequence eval curves
# ═══════════════════════════════════════════════════════════════

def plot_supp_sequence():
    print("Fig7: Supplementary sequence...")
    seq_str = "Walker2d-v4_to_HalfCheetah-v4_to_Walker2d-v4"
    first_env = "Walker2d-v4"

    fig, ax = plt.subplots(1, 1, figsize=(FULL_W, 3.5))

    for key, info in SEQ_ALGOS.items():
        all_steps, all_rewards = [], []
        # Try up to 20 seeds
        for seed in range(20):
            evals = load_sequence_eval(info["dir"], seq_str, seed)
            if evals is None:
                continue
            xs = [e["step"] for e in evals if e["eval_env"] == first_env]
            ys = [e["reward"] for e in evals if e["eval_env"] == first_env]
            if xs:
                all_steps.append(np.array(xs))
                all_rewards.append(np.array(ys))
        if not all_steps:
            continue
        n = len(all_steps)
        common_x, interped = interp_to_common(all_steps, all_rewards, n_points=100)
        mean = interped.mean(axis=0)
        std = interped.std(axis=0)
        ax.plot(common_x / 1000, mean,
                label=f"{info['label']} (n={n})",
                color=info["color"], linewidth=1.5)
        ax.fill_between(common_x / 1000, mean - std, mean + std,
                        alpha=0.15, color=info["color"])

    for boundary in [80, 160]:
        ax.axvline(boundary, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(40, ax.get_ylim()[1]*0.95, "Phase 1\nWalker2d", ha="center", fontsize=6, color="gray")
    ax.text(120, ax.get_ylim()[1]*0.95, "Phase 2\nHalfCheetah", ha="center", fontsize=6, color="gray")
    ax.text(200, ax.get_ylim()[1]*0.95, "Phase 3\nWalker2d", ha="center", fontsize=6, color="gray")

    ax.set_xlabel("Total Steps (k)")
    ax.set_ylabel("Walker2d-v4 Eval Return")
    ax.set_title("Walker2d→HalfCheetah→Walker2d (Eval on Walker2d)")
    ax.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, "fig07_supp_sequence")


# ═══════════════════════════════════════════════════════════════
# Table 1: Single-task results (6 algos × 4 envs)
# ═══════════════════════════════════════════════════════════════

def make_table1():
    print("Table1: Single-task results...")
    header = ["Algorithm"] + [e.replace("-v4", "") for e in ENVS]
    rows = [header]

    for key, info in MAIN_ALGOS.items():
        row = [info["label"]]
        for env in ENVS:
            vals = []
            for seed in SEEDS:
                v = load_final_reward(info["dir"], env, seed)
                if v is not None:
                    vals.append(v)
            if vals:
                row.append(f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}")
            else:
                row.append("N/A")
        rows.append(row)

    # Print
    print("\n  === Table 1: Single-Task Final Performance ===")
    for row in rows:
        print("  " + " | ".join(f"{c:>28s}" for c in row))
    save_csv(rows, "table1_single_task")


# ═══════════════════════════════════════════════════════════════
# Table 2: Main continual learning sequence (20 seeds)
# ═══════════════════════════════════════════════════════════════

def make_table2():
    print("Table2: Main sequence results...")
    seq_str = "Hopper-v4_to_Walker2d-v4_to_Hopper-v4"

    header = ["Algorithm", "Seeds", "Revisit Reward", "Retention", "Forgetting"]
    rows = [header]
    all_data = {}

    for key, info in SEQ_ALGOS.items():
        n_seeds = 20 if key in ("CLFull", "CLMemory") else 10
        revisits, retentions, forgettings = [], [], []
        for seed in range(n_seeds):
            summaries = load_sequence_phase_summary(info["dir"], seq_str, seed)
            if summaries is None:
                continue
            phase3 = [s for s in summaries if s["phase"] == 3]
            if phase3:
                revisits.append(phase3[0]["current_a"])
                retentions.append(phase3[0]["retention"])
                forgettings.append(phase3[0]["forgetting"])

        all_data[key] = {"revisit": revisits, "retention": retentions, "forgetting": forgettings}
        n = len(revisits)
        row = [
            info["label"],
            str(n),
            f"{np.mean(revisits):.2f} +/- {np.std(revisits):.2f}" if revisits else "N/A",
            f"{np.mean(retentions):.3f} +/- {np.std(retentions):.3f}" if retentions else "N/A",
            f"{np.mean(forgettings):.2f} +/- {np.std(forgettings):.2f}" if forgettings else "N/A",
        ]
        rows.append(row)

    # p-values: CLMemory vs CLFull
    if all_data.get("CLMemory") and all_data.get("CLFull"):
        mem = all_data["CLMemory"]
        full = all_data["CLFull"]
        n_paired = min(len(mem["revisit"]), len(full["revisit"]))
        if n_paired >= 5:
            p_revisit = sign_flip_test(mem["revisit"][:n_paired], full["revisit"][:n_paired])
            p_retention = sign_flip_test(mem["retention"][:n_paired], full["retention"][:n_paired])
            p_forgetting = sign_flip_test(mem["forgetting"][:n_paired], full["forgetting"][:n_paired])
            rows.append(["p-value (Memory vs Full)", "",
                         f"p={p_revisit:.4f}", f"p={p_retention:.4f}", f"p={p_forgetting:.4f}"])

    print("\n  === Table 2: Main Sequence (Hopper->Walker2d->Hopper) ===")
    for row in rows:
        print("  " + " | ".join(f"{c:>25s}" for c in row))
    save_csv(rows, "table2_main_sequence")


# ═══════════════════════════════════════════════════════════════
# Table 3: Ablation (6 combos × 2 envs)
# ═══════════════════════════════════════════════════════════════

def make_table3():
    print("Table3: Ablation results...")
    abl_envs = ["Hopper-v4", "Walker2d-v4"]
    header = ["Module Combination"] + [e.replace("-v4", "") for e in abl_envs]
    rows = [header]

    for key, info in ABLATION_ALGOS.items():
        row = [info["label"]]
        for env in abl_envs:
            vals = []
            for seed in SEEDS:
                v = load_final_reward(info["dir"], env, seed)
                if v is not None:
                    vals.append(v)
            if vals:
                row.append(f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}")
            else:
                row.append("N/A")
        rows.append(row)

    print("\n  === Table 3: Ablation ===")
    for row in rows:
        print("  " + " | ".join(f"{c:>35s}" for c in row))
    save_csv(rows, "table3_ablation")


# ═══════════════════════════════════════════════════════════════
# Table 4: Supplementary sequence
# ═══════════════════════════════════════════════════════════════

def make_table4():
    print("Table4: Supplementary sequence...")
    seq_str = "Walker2d-v4_to_HalfCheetah-v4_to_Walker2d-v4"

    header = ["Algorithm", "Seeds", "Revisit Reward", "Retention", "Forgetting"]
    rows = [header]

    for key, info in SEQ_ALGOS.items():
        revisits, retentions, forgettings = [], [], []
        for seed in range(20):
            summaries = load_sequence_phase_summary(info["dir"], seq_str, seed)
            if summaries is None:
                continue
            phase3 = [s for s in summaries if s["phase"] == 3]
            if phase3:
                revisits.append(phase3[0]["current_a"])
                ret = phase3[0]["retention"]
                if abs(ret) > 100:  # clip unreasonable retention (ref_a near 0)
                    ret = float("nan")
                retentions.append(ret)
                forgettings.append(phase3[0]["forgetting"])
        retentions = [r for r in retentions if not np.isnan(r)]
        n = len(revisits)
        row = [
            info["label"],
            str(n),
            f"{np.mean(revisits):.2f} +/- {np.std(revisits):.2f}" if revisits else "N/A",
            f"{np.mean(retentions):.3f} +/- {np.std(retentions):.3f}" if retentions else "N/A",
            f"{np.mean(forgettings):.2f} +/- {np.std(forgettings):.2f}" if forgettings else "N/A",
        ]
        rows.append(row)

    # p-values: CLMemory vs CLFull
    all_data = {}
    for key, info in SEQ_ALGOS.items():
        revisits = []
        for seed in range(20):
            summaries = load_sequence_phase_summary(info["dir"], seq_str, seed)
            if summaries is None:
                continue
            phase3 = [s for s in summaries if s["phase"] == 3]
            if phase3:
                revisits.append(phase3[0]["current_a"])
        all_data[key] = revisits
    if len(all_data.get("CLMemory", [])) >= 10 and len(all_data.get("CLFull", [])) >= 10:
        n_paired = min(len(all_data["CLMemory"]), len(all_data["CLFull"]))
        p_revisit = sign_flip_test(all_data["CLMemory"][:n_paired], all_data["CLFull"][:n_paired])
        rows.append(["p-value (Memory vs Full)", "", f"p={p_revisit:.4f}", "", ""])

    print("\n  === Table 4: Supplementary Sequence (Walker2d->HalfCheetah->Walker2d) ===")
    for row in rows:
        print("  " + " | ".join(f"{c:>25s}" for c in row))
    save_csv(rows, "table4_supp_sequence")


# ═══════════════════════════════════════════════════════════════
# Fig8 + Table5: Progressive sequence ablation
# ═══════════════════════════════════════════════════════════════

SEQ_PROGRESSIVE = {
    "P3O":          {"dir": "P3O",                  "label": "P3O (B)\nn=10",
                     "color": "#ff7f0e", "seeds": list(range(10))},
    "B+Alpha":      {"dir": "Seq-B+Alpha",          "label": "B+\u03b1\nn=10",
                     "color": "#9467bd", "seeds": list(range(10))},
    "B+Alpha+Reset":{"dir": "Seq-B+Alpha+Reset",    "label": "B+\u03b1+R\nn=10",
                     "color": "#8c564b", "seeds": list(range(10))},
    "CLFull":       {"dir": "P3O-ClosedLoopFull",   "label": "CLFull\nn=20",
                     "color": "#d62728", "seeds": list(range(20))},
    "CLMemory":     {"dir": "P3O-ClosedLoopMemory", "label": "CLMemory\nn=20",
                     "color": "#2ca02c", "seeds": list(range(20))},
}


def _collect_seq_ablation_data():
    """Collect revisit/retention/forgetting for each progressive algo."""
    seq_str = "Hopper-v4_to_Walker2d-v4_to_Hopper-v4"
    data = {}
    for key, info in SEQ_PROGRESSIVE.items():
        revisits, retentions, forgettings = [], [], []
        for seed in info["seeds"]:
            summaries = load_sequence_phase_summary(info["dir"], seq_str, seed)
            if summaries is None:
                continue
            phase3 = [s for s in summaries if s["phase"] == 3]
            if not phase3:
                continue
            s = phase3[0]
            revisits.append(s["current_a"])
            ref = s["reference_a"]
            if ref > 1e-3:
                ret = s["current_a"] / ref
                if abs(ret) <= 100:
                    retentions.append(ret)
            forgettings.append(s["forgetting"])
        data[key] = {
            "revisit": np.array(revisits),
            "retention": np.array(retentions),
            "forgetting": np.array(forgettings),
            "label": info["label"],
            "color": info["color"],
        }
    return data


def plot_progressive_ablation():
    """Fig8: Progressive ablation bar chart for sequence."""
    print("Fig8: Progressive sequence ablation...")
    data = _collect_seq_ablation_data()

    keys = list(SEQ_PROGRESSIVE.keys())
    metrics = [
        ("revisit",   "Revisit Reward"),
        ("retention", "Retention Rate"),
        ("forgetting","Forgetting"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W + 1.0, 3.4))

    for ax_i, (metric, title) in enumerate(metrics):
        ax = axes[ax_i]
        x_pos = np.arange(len(keys))
        means, stds, colors, labels = [], [], [], []
        for k in keys:
            d = data[k]
            vals = d[metric]
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
            colors.append(d["color"])
            labels.append(d["label"])

        ax.bar(x_pos, means, yerr=stds, capsize=3,
               color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7, rotation=0, ha="center")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(title, fontsize=7)

        upper_candidates = [m + s for m, s in zip(means, stds)]
        lower_candidates = [m - s for m, s in zip(means, stds)]
        y_span = max(max(upper_candidates) - min(lower_candidates), 1.0)
        y_cursor = max(upper_candidates) + y_span * 0.10

        # Add significance marks only for statistically significant adjacent pairs.
        for i in range(len(keys) - 1):
            d1 = data[keys[i]][metric]
            d2 = data[keys[i+1]][metric]
            if len(d1) >= 5 and len(d2) >= 5:
                n_paired = min(len(d1), len(d2))
                p = sign_flip_test(d2[:n_paired], d1[:n_paired])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else None
                if sig is None:
                    continue
                bracket_h = y_span * 0.02
                text_offset = y_span * 0.015
                ax.plot([i, i, i+1, i+1],
                        [y_cursor - bracket_h, y_cursor, y_cursor, y_cursor - bracket_h],
                        lw=0.7, color="black")
                ax.text((i + i+1)/2, y_cursor + text_offset, sig,
                        ha="center", va="bottom", fontsize=6)
                y_cursor += y_span * 0.10

        ax.set_ylim(min(lower_candidates) - y_span * 0.08, y_cursor + y_span * 0.05)

    fig.tight_layout()
    save_fig(fig, "fig08_progressive_ablation")


def make_table5():
    """Table5: Progressive sequence ablation summary."""
    print("Table5: Progressive sequence ablation...")
    data = _collect_seq_ablation_data()

    keys = list(SEQ_PROGRESSIVE.keys())
    display_names = {
        "P3O": "P3O (B)",
        "B+Alpha": "B+Alpha",
        "B+Alpha+Reset": "B+Alpha+Reset",
        "CLFull": "B+A+R+Distill (ClosedLoopFull)",
        "CLMemory": "B+A+R+D+Memory (ClosedLoopMemory)",
    }

    rows = [["Algorithm", "Seeds", "Revisit Reward", "Retention", "Forgetting"]]
    for k in keys:
        d = data[k]
        n = len(d["revisit"])
        def fmt(arr):
            if len(arr) == 0:
                return "N/A"
            return f"{np.mean(arr):.2f} +/- {np.std(arr):.2f}"
        rows.append([
            display_names[k],
            str(n),
            fmt(d["revisit"]),
            fmt(d["retention"]),
            fmt(d["forgetting"]),
        ])

    # p-values for adjacent pairs
    for i in range(len(keys) - 1):
        d1_rev = data[keys[i]]["revisit"]
        d2_rev = data[keys[i+1]]["revisit"]
        if len(d1_rev) >= 5 and len(d2_rev) >= 5:
            n_paired = min(len(d1_rev), len(d2_rev))
            p = sign_flip_test(d2_rev[:n_paired], d1_rev[:n_paired])
            rows.append([f"p ({keys[i+1]} vs {keys[i]})", "", f"p={p:.4f}", "", ""])

    print(f"\n  === Table 5: Progressive Sequence Ablation (Hopper->Walker2d->Hopper) ===")
    for row in rows:
        print("  " + " | ".join(f"{c:>35s}" for c in row))
    save_csv(rows, "table5_seq_progressive_ablation")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Output dir: {FIG_DIR}\n")

    # Figures
    plot_learning_curves()
    plot_main_sequence()
    plot_retention_forgetting()
    plot_ablation_bars()
    plot_alpha_trajectory()
    plot_supp_sequence()
    plot_progressive_ablation()

    # Tables
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    make_table5()

    print(f"\nAll done! Files in: {FIG_DIR}")


if __name__ == "__main__":
    main()
