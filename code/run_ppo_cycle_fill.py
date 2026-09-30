import csv
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim


HP = {
    "lr": 3e-4,
    "buffer_size": 8192,
    "batch_size": 256,
    "gamma": 0.99,
    "train_steps": 500_000,
    "epochs_per_update": 10,
    "clip_range": 0.2,
    "clip_grad_norm": 0.5,
    "hidden_size": 256,
    "hidden_layers": 3,
    "activation": nn.Tanh,
    "reset_rate": 0.01,
    "reset_frequency": 50000,
    "alpha_dkl": 0.4,
    "alpha_start": 0.6,
    "alpha_end": 0.2,
    "alpha_lambda": 4.0,
}

ENV_LIST = ["Hopper-v4", "Walker2d-v4"]
SEEDS = [0, 1]
RUNS_DIR = r"C:\Users\33277\Desktop\p3o_runs"
ALGO_NAME = "PPO+Cycle"
SBP_MODE = "cycle"  # only reset, no distill
ALPHA_MODE = "fixed"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ExperimentLogger:
    def __init__(self, run_dir: str, config: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        # Universal short header; parser ignores this row.
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "tag"])

    def log_speed(self, step, sps, elapsed_min, algo, env, seed):
        row = [step, "speed", round(sps, 2), round(elapsed_min, 2), algo, env, seed]
        self._append_row(row)

    def log_loss(self, step, actor_loss, critic_loss, algo, env, seed):
        row = [step, "loss", actor_loss, critic_loss, algo, env, seed]
        self._append_row(row)

    def log_sbp(self, step, alpha_static, alpha_dynamic, alpha_using, algo, env, seed):
        row = [
            step,
            "sbp",
            round(alpha_static, 4),
            round(alpha_dynamic, 4),
            round(alpha_using, 4),
            algo,
            env,
            seed,
        ]
        self._append_row(row)

    def log_reward(self, step, episode, reward, avg10, algo, env, seed):
        row = [step, "reward", episode, float(reward), round(avg10, 3), algo, env, seed]
        self._append_row(row)

    def _append_row(self, row):
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hp):
        super().__init__()
        self.action_bound = action_bound
        self.hp = hp
        act = hp["activation"]

        actor_layers = []
        in_dim = state_dim
        for _ in range(hp["hidden_layers"]):
            actor_layers.append(nn.Linear(in_dim, hp["hidden_size"]))
            actor_layers.append(act())
            in_dim = hp["hidden_size"]
        actor_layers.append(nn.Linear(hp["hidden_size"], action_dim))
        self.actor = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        critic_layers = []
        in_dim = state_dim
        for _ in range(hp["hidden_layers"]):
            critic_layers.append(nn.Linear(in_dim, hp["hidden_size"]))
            critic_layers.append(act())
            in_dim = hp["hidden_size"]
        critic_layers.append(nn.Linear(hp["hidden_size"], 1))
        self.critic = nn.Sequential(*critic_layers)

        self.reset_index = 0.0

    def get_action(self, state):
        mean = self.actor(state)
        mean = torch.tanh(mean) * self.action_bound
        std = torch.exp(self.log_std)
        return dist.Normal(mean, std)

    def get_value(self, state):
        return self.critic(state)

    def cycle_reset(self):
        self.eval()
        reset_p = self.hp["reset_rate"]
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                out_dim, _ = layer.weight.shape
                reset_start = int(self.reset_index * out_dim)
                reset_end = int((self.reset_index + reset_p) * out_dim)
                if reset_end > out_dim:
                    reset_end = out_dim
                    self.reset_index = 0.0
                if reset_start < reset_end:
                    nn.init.xavier_uniform_(layer.weight[reset_start:reset_end, :])
                    nn.init.zeros_(layer.bias[reset_start:reset_end])
        self.reset_index = (self.reset_index + reset_p) % 1.0
        self.train()


class Buffer:
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
        )

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen


def get_dynamic_alpha(progress, alpha_start, alpha_end, lam):
    p = min(max(progress, 0.0), 1.0)
    tanh_term = np.tanh(lam * (p - 0.5))
    return alpha_start * (1 - tanh_term) / 2 + alpha_end * (1 + tanh_term) / 2


def ppo_update(ac, buffer, hp, optimizer):
    states, actions, rewards, next_states, dones = buffer.get_batch()
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    with torch.no_grad():
        values = ac.get_value(states).squeeze()
        next_values = ac.get_value(next_states).squeeze()
        deltas = rewards + hp["gamma"] * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        adv = 0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + hp["gamma"] * 0.95 * adv * (1 - dones[t])
            advantages[t] = adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards + hp["gamma"] * next_values * (1 - dones)

    last_actor_loss = 0.0
    last_critic_loss = 0.0
    for _ in range(hp["epochs_per_update"]):
        pi = ac.get_action(states)
        log_probs = pi.log_prob(actions).sum(dim=1)
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - hp["clip_range"], 1 + hp["clip_range"]) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(ac.get_value(states).squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), hp["clip_grad_norm"])
        optimizer.step()

        last_actor_loss = actor_loss.item()
        last_critic_loss = critic_loss.item()
    return last_actor_loss, last_critic_loss


def train_one(env_name, seed):
    seed_everything(seed)
    hp = dict(HP)
    run_dir = os.path.join(
        RUNS_DIR, f"{env_name}_{ALGO_NAME}_seed{seed}_steps{hp['train_steps']}"
    )

    # Skip if already exists and non-empty
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 64:
        print(f"[skip] exists: {run_dir}")
        return

    env = gym.make(env_name)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    ac = ActorCritic(state_dim, action_dim, action_bound, hp)
    optimizer = optim.Adam(ac.parameters(), lr=hp["lr"])
    buffer = Buffer(hp["buffer_size"], hp["batch_size"])
    logger = ExperimentLogger(
        run_dir,
        {
            "env": env_name,
            "algo": ALGO_NAME,
            "seed": seed,
            "train_steps": hp["train_steps"],
            "sbp_mode": SBP_MODE,
            "alpha_mode": ALPHA_MODE,
            "hp": hp,
        },
    )

    total_steps = 0
    max_steps = int(hp["train_steps"])
    episode_reward = 0.0
    episode_idx = 0
    recent_rewards = deque(maxlen=10)

    t_start = time.time()
    last_log_steps = 0
    last_log_time = t_start

    while total_steps < max_steps:
        action_dist = ac.get_action(torch.FloatTensor(state))
        action = action_dist.sample().detach().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_steps += 1

        if total_steps - last_log_steps >= 10000:
            now = time.time()
            sps = (total_steps - last_log_steps) / max(now - last_log_time, 1e-6)
            logger.log_speed(
                total_steps, sps, (now - t_start) / 60.0, ALGO_NAME, env_name, seed
            )
            last_log_steps = total_steps
            last_log_time = now

        if buffer.is_full():
            actor_loss, critic_loss = ppo_update(ac, buffer, hp, optimizer)
            buffer.buffer.clear()
            logger.log_loss(
                total_steps, actor_loss, critic_loss, ALGO_NAME, env_name, seed
            )

        if SBP_MODE != "none" and total_steps % hp["reset_frequency"] == 0:
            ac.cycle_reset()
            progress = total_steps / max_steps
            static_alpha = hp["alpha_dkl"]
            dynamic_alpha = get_dynamic_alpha(
                progress, hp["alpha_start"], hp["alpha_end"], hp["alpha_lambda"]
            )
            alpha_using = static_alpha
            logger.log_sbp(
                total_steps,
                static_alpha,
                dynamic_alpha,
                alpha_using,
                ALGO_NAME,
                env_name,
                seed,
            )
            print(
                f"[{env_name} seed{seed}] step={total_steps} reset done | static={static_alpha:.3f} dynamic={dynamic_alpha:.3f}"
            )

        if done:
            episode_idx += 1
            recent_rewards.append(episode_reward)
            avg10 = sum(recent_rewards) / len(recent_rewards)
            logger.log_reward(
                total_steps,
                episode_idx,
                episode_reward,
                avg10,
                ALGO_NAME,
                env_name,
                seed,
            )
            episode_reward = 0.0
            state, _ = env.reset()

    env.close()
    print(f"[done] {run_dir}")


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    for env_name in ENV_LIST:
        for seed in SEEDS:
            train_one(env_name, seed)
    print("All PPO+Cycle fill runs finished.")


if __name__ == "__main__":
    main()
