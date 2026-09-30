import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = r"C:\Users\33277\Desktop\p3o_runs"
OUT_DIR = os.path.join(RUNS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

files = glob.glob(os.path.join(RUNS_DIR, "**", "metrics.csv"), recursive=True)
if not files:
    print("No metrics.csv found under", RUNS_DIR)
    raise SystemExit(1)

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source"] = f
        dfs.append(df)
    except Exception as e:
        print("Skip:", f, e)

data = pd.concat(dfs, ignore_index=True)

for col in [
    "step",
    "episode",
    "reward",
    "avg10",
    "actor_loss",
    "critic_loss",
    "alpha_static",
    "alpha_dynamic",
    "alpha_using",
    "sps",
    "elapsed_min",
]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# 1) Reward curve
reward = data[data["tag"] == "reward"].dropna(subset=["step", "avg10"])
envs = sorted(reward["env"].dropna().unique())
algos = sorted(reward["algo"].dropna().unique())

for env in envs:
    plt.figure()
    for algo in algos:
        sub = reward[(reward["env"] == env) & (reward["algo"] == algo)]
        if sub.empty:
            continue
        agg = (
            sub.groupby("step")["avg10"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("step")
        )
        plt.plot(agg["step"], agg["mean"], label=algo)
        if agg["std"].notna().any():
            plt.fill_between(
                agg["step"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.15,
            )
    plt.xlabel("Environment Steps")
    plt.ylabel("Avg Reward (last 10)")
    plt.title(f"Reward Curve - {env}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"reward_{env}.png"), dpi=200)
    plt.close()

# 2) Loss curve
loss = data[data["tag"] == "loss"].dropna(subset=["step"])
for env in sorted(loss["env"].dropna().unique()):
    plt.figure()
    for algo in sorted(loss["algo"].dropna().unique()):
        sub = loss[(loss["env"] == env) & (loss["algo"] == algo)]
        if sub.empty:
            continue
        a = sub.groupby("step")["actor_loss"].mean().reset_index().sort_values("step")
        c = (
            sub.groupby("step")["critic_loss"]
            .mean()
            .reset_index()
            .sort_values("step")
        )
        plt.plot(a["step"], a["actor_loss"], label=f"{algo}-actor")
        plt.plot(c["step"], c["critic_loss"], label=f"{algo}-critic", linestyle="--")
    plt.xlabel("Environment Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {env}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"loss_{env}.png"), dpi=200)
    plt.close()

# 3) Alpha curve
alpha = data[data["tag"] == "sbp"].dropna(subset=["step"])
for env in sorted(alpha["env"].dropna().unique()):
    plt.figure()
    for algo in sorted(alpha["algo"].dropna().unique()):
        sub = alpha[(alpha["env"] == env) & (alpha["algo"] == algo)]
        if sub.empty:
            continue
        agg = (
            sub.groupby("step")["alpha_using"]
            .mean()
            .reset_index()
            .sort_values("step")
        )
        plt.plot(agg["step"], agg["alpha_using"], label=f"{algo}-using")
    plt.xlabel("Environment Steps")
    plt.ylabel("Alpha")
    plt.title(f"Alpha Curve - {env}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"alpha_{env}.png"), dpi=200)
    plt.close()

print("Done. Plots saved to:", OUT_DIR)
