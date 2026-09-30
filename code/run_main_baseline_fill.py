import argparse
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
    "gae_lambda": 0.95,
    "train_steps": 500_000,
    "epochs_per_update": 10,
    "clip_range": 0.2,
    "clip_grad_norm": 0.5,
    "hidden_size": 256,
    "hidden_layers": 3,
    "activation": nn.Tanh,
    "reset_rate": 0.01,
    "reset_frequency": 50_000,
    "alpha_dkl": 0.4,
    "alpha_start": 0.6,
    "alpha_end": 0.2,
    "alpha_lambda": 4.0,
    "distill_loss_bound": 0.01,
    "distill_max_steps": 80,
}

RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")

ALGO_SETTINGS = {
    "PPO": {"sbp_mode": "none", "alpha_mode": "fixed"},
    "PPO+Cycle": {"sbp_mode": "cycle", "alpha_mode": "fixed"},
    "P3O": {"sbp_mode": "cycle+distill", "alpha_mode": "dynamic"},
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentLogger:
    def __init__(self, run_dir: str, config: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "tag"])

    def _append(self, row):
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def log_speed(self, step, sps, elapsed_min, algo, env, seed):
        self._append([step, "speed", round(sps, 2), round(elapsed_min, 2), algo, env, seed])

    def log_loss(self, step, actor_loss, critic_loss, algo, env, seed):
        self._append([step, "loss", actor_loss, critic_loss, algo, env, seed])

    def log_sbp(self, step, alpha_static, alpha_dynamic, alpha_using, algo, env, seed):
        self._append(
            [
                step,
                "sbp",
                round(alpha_static, 4),
                round(alpha_dynamic, 4),
                round(alpha_using, 4),
                algo,
                env,
                seed,
            ]
        )

    def log_reward(self, step, episode, reward, avg10, algo, env, seed):
        self._append([step, "reward", episode, float(reward), round(avg10, 3), algo, env, seed])


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


class ReplayBuffer:
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen


def get_dynamic_alpha(progress, alpha_start, alpha_end, lam):
    p = min(max(progress, 0.0), 1.0)
    tanh_term = np.tanh(lam * (p - 0.5))
    return alpha_start * (1 - tanh_term) / 2 + alpha_end * (1 + tanh_term) / 2


def alpha_dkl_loss(pi_teacher, pi_student, alpha):
    kl_forward = dist.kl_divergence(pi_teacher, pi_student).mean()
    kl_backward = dist.kl_divergence(pi_student, pi_teacher).mean()
    return alpha * kl_forward + (1.0 - alpha) * kl_backward


def inner_distill(actor_critic, teacher, state_batch, alpha, hp):
    actor_critic.train()
    teacher.eval()
    optimizer = optim.Adam(actor_critic.parameters(), lr=hp["lr"])
    distill_loss = float("inf")
    steps = 0
    while distill_loss > hp["distill_loss_bound"] and steps < hp["distill_max_steps"]:
        optimizer.zero_grad()
        pi_student = actor_critic.get_action(state_batch)
        with torch.no_grad():
            pi_teacher = teacher.get_action(state_batch)
        loss = alpha_dkl_loss(pi_teacher, pi_student, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), hp["clip_grad_norm"])
        optimizer.step()
        distill_loss = loss.item()
        steps += 1
    return actor_critic


def ppo_update(actor_critic, buffer, hp, optimizer):
    states, actions, rewards, next_states, dones = buffer.get_batch()
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    with torch.no_grad():
        values = actor_critic.get_value(states).squeeze()
        next_values = actor_critic.get_value(next_states).squeeze()
        deltas = rewards + hp["gamma"] * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        adv = 0.0
        for idx in reversed(range(len(deltas))):
            adv = deltas[idx] + hp["gamma"] * hp["gae_lambda"] * adv * (1 - dones[idx])
            advantages[idx] = adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards + hp["gamma"] * next_values * (1 - dones)

    old_log_probs = actor_critic.get_action(states).log_prob(actions).sum(dim=1).detach()
    last_actor_loss = 0.0
    last_critic_loss = 0.0

    for _ in range(hp["epochs_per_update"]):
        pi = actor_critic.get_action(states)
        log_probs = pi.log_prob(actions).sum(dim=1)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - hp["clip_range"], 1 + hp["clip_range"]) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(actor_critic.get_value(states).squeeze(), returns)
        total_loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), hp["clip_grad_norm"])
        optimizer.step()

        last_actor_loss = actor_loss.item()
        last_critic_loss = critic_loss.item()

    return last_actor_loss, last_critic_loss


def train_one(env_name, algo_name, seed, steps, runs_dir, force=False):
    cfg = ALGO_SETTINGS[algo_name]
    hp = dict(HP)
    hp["train_steps"] = int(steps)

    seed_everything(seed)
    run_dir = os.path.join(runs_dir, f"{env_name}_{algo_name}_seed{seed}_steps{int(steps)}")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not force and os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 64:
        print(f"[skip] exists: {run_dir}")
        return run_dir

    env = gym.make(env_name)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor_critic = ActorCritic(state_dim, action_dim, action_bound, hp)
    optimizer = optim.Adam(actor_critic.parameters(), lr=hp["lr"])
    buffer = ReplayBuffer(hp["buffer_size"], hp["batch_size"])
    logger = ExperimentLogger(
        run_dir,
        {
            "env": env_name,
            "algo": algo_name,
            "seed": seed,
            "train_steps": steps,
            "sbp_mode": cfg["sbp_mode"],
            "alpha_mode": cfg["alpha_mode"],
            "hp": hp,
        },
    )

    total_steps = 0
    episode_reward = 0.0
    episode_idx = 0
    recent_rewards = deque(maxlen=10)
    t_start = time.time()
    last_log_steps = 0
    last_log_time = t_start

    while total_steps < hp["train_steps"]:
        action_dist = actor_critic.get_action(torch.FloatTensor(state))
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
            logger.log_speed(total_steps, sps, (now - t_start) / 60.0, algo_name, env_name, seed)
            last_log_steps = total_steps
            last_log_time = now

        if buffer.is_full():
            actor_loss, critic_loss = ppo_update(actor_critic, buffer, hp, optimizer)
            buffer.buffer.clear()
            logger.log_loss(total_steps, actor_loss, critic_loss, algo_name, env_name, seed)

        if cfg["sbp_mode"] != "none" and total_steps % hp["reset_frequency"] == 0:
            teacher = ActorCritic(state_dim, action_dim, action_bound, hp)
            teacher.load_state_dict(actor_critic.state_dict())

            if cfg["sbp_mode"] in {"cycle", "cycle+distill"}:
                actor_critic.cycle_reset()

            progress = total_steps / float(hp["train_steps"])
            static_alpha = hp["alpha_dkl"]
            dynamic_alpha = get_dynamic_alpha(progress, hp["alpha_start"], hp["alpha_end"], hp["alpha_lambda"])
            alpha = dynamic_alpha if cfg["alpha_mode"] == "dynamic" else static_alpha

            if cfg["sbp_mode"] == "cycle+distill":
                if len(buffer.buffer) >= hp["batch_size"]:
                    state_batch, _, _, _, _ = buffer.get_batch()
                    state_batch = torch.FloatTensor(state_batch)
                else:
                    samples = [env.observation_space.sample() for _ in range(hp["batch_size"])]
                    state_batch = torch.FloatTensor(np.array(samples))
                actor_critic = inner_distill(actor_critic, teacher, state_batch, alpha, hp)

            logger.log_sbp(total_steps, static_alpha, dynamic_alpha, alpha, algo_name, env_name, seed)

        if done:
            episode_idx += 1
            recent_rewards.append(episode_reward)
            avg10 = sum(recent_rewards) / len(recent_rewards)
            logger.log_reward(total_steps, episode_idx, episode_reward, avg10, algo_name, env_name, seed)
            episode_reward = 0.0
            state, _ = env.reset()

    env.close()
    print(f"[done] {run_dir}")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run main baseline fills for PPO / PPO+Cycle / P3O.")
    parser.add_argument("--algo", choices=sorted(ALGO_SETTINGS.keys()), required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=HP["train_steps"])
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.runs_dir, exist_ok=True)
    train_one(args.env, args.algo, args.seed, args.steps, args.runs_dir, force=args.force)


if __name__ == "__main__":
    main()
