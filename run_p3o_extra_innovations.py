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
    "alpha_td_k": 0.10,
    "alpha_min": 0.15,
    "alpha_max": 0.65,
    "alpha_eps": 1e-8,
    "alpha_dual_lr": 0.035,
    "alpha_target_kl": 0.025,
    "alpha_progress_blend": 0.55,
    "distill_loss_bound": 0.01,
    "distill_max_steps": 80,
    "hard_top_ratio": 0.3,
    "reset_rate_min": 0.002,
    "reset_rate_max": 0.03,
    "reset_k": 0.8,
    "td_ema_decay": 0.90,
    "td_ref_decay": 0.99,
    "metric_eps": 1e-8,
    "reset_td_weight": 1.0,
    "reset_kl_weight": 0.35,
    "reset_ent_weight": 0.20,
    "priority_td_weight": 0.70,
    "priority_kl_weight": 0.30,
    "priority_gamma": 1.50,
    "distill_mix_random": 0.20,
    "importance_act_weight": 0.70,
    "importance_weight_weight": 0.30,
    "sbp_warmup_steps": 20_000,
    "sbp_min_interval": 20_000,
    "sbp_force_interval": 80_000,
    "sbp_event_threshold": 0.35,
}

RUNS_DIR = os.environ.get("P3O_RUNS_DIR", r"C:\Users\33277\Desktop\p3o_runs")


ALGO_SETTINGS = {
    "P3O-dynamic": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "adaptive_reset": False,
        "hard_distill": False,
    },
    "P3O-dyn+AdaReset": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "adaptive_reset": True,
        "hard_distill": False,
    },
    "P3O-dyn+HardDistill": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "adaptive_reset": False,
        "hard_distill": True,
    },
    "P3O-dyn+AdaReset+HardDistill": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "event_trigger_sbp": False,
    },
    "P3O-dyn+EvtSBP+AdaReset+HardDistill": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "event_trigger_sbp": True,
    },
    "P3O-ClosedLoopAlpha": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "event_trigger_sbp": True,
    },
    "P3O-ClosedLoopFull": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "event_trigger_sbp": True,
        "selective_reset": True,
    },
    # ═══ 消融实验: 减法 (从 ClosedLoopFull 逐一移除) ═══
    "Abl-Full-NoDualAlpha": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": False,
        "adaptive_reset": True,
        "hard_distill": True,
        "event_trigger_sbp": True,
        "selective_reset": True,
    },
    "Abl-Full-NoEvtReset": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": False,
        "hard_distill": True,
        "event_trigger_sbp": False,
        "selective_reset": False,
    },
    "Abl-Full-NoHardDistill": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": False,
        "event_trigger_sbp": True,
        "selective_reset": True,
    },
    # ═══ 消融实验: 加法 (在 P3O-dynamic 上仅加一个模块) ═══
    "Abl-OnlyDualAlpha": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": False,
        "hard_distill": False,
        "event_trigger_sbp": False,
        "selective_reset": False,
    },
    "Abl-OnlyEvtReset": {
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": False,
        "adaptive_reset": True,
        "hard_distill": False,
        "event_trigger_sbp": True,
        "selective_reset": False,
    },
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ExperimentLogger:
    def __init__(self, run_dir: str, config: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        self._has_header = False
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    def log(self, *row):
        mode = "a" if self._has_header else "w"
        with open(self.metrics_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._has_header:
                writer.writerow(["step", "tag"])
                self._has_header = True
            writer.writerow(row)


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

    def cycle_reset(self, reset_p=None):
        self.eval()
        p = float(self.hp["reset_rate"] if reset_p is None else reset_p)
        p = float(np.clip(p, 1e-5, 1.0))
        for layer in self.actor:
            if not isinstance(layer, nn.Linear):
                continue
            out_dim, _ = layer.weight.shape
            reset_start = int(self.reset_index * out_dim)
            reset_end = int((self.reset_index + p) * out_dim)
            if reset_end > out_dim:
                reset_end = out_dim
                self.reset_index = 0.0
            if reset_start < reset_end:
                nn.init.xavier_uniform_(layer.weight[reset_start:reset_end, :])
                nn.init.zeros_(layer.bias[reset_start:reset_end])
        self.reset_index = (self.reset_index + p) % 1.0
        self.train()

    def _actor_linear_layers(self):
        return [layer for layer in self.actor if isinstance(layer, nn.Linear)]

    def compute_reset_importance(self, state_batch):
        linears = self._actor_linear_layers()
        if len(linears) <= 1:
            return []

        importance = []
        x = state_batch
        hidden_idx = -1
        waiting_activation = False
        for layer in self.actor:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                if hidden_idx + 1 < len(linears) - 1:
                    hidden_idx += 1
                    waiting_activation = True
                else:
                    waiting_activation = False
            elif waiting_activation:
                activation_score = x.detach().abs().mean(dim=0)
                next_linear = linears[hidden_idx + 1]
                outgoing_score = next_linear.weight.detach().abs().mean(dim=0)
                score = (
                    self.hp["importance_act_weight"] * activation_score
                    + self.hp["importance_weight_weight"] * outgoing_score
                )
                importance.append(score.cpu().numpy())
                waiting_activation = False
        return importance

    @staticmethod
    def _reset_rows(linear, row_idx):
        if len(row_idx) == 0:
            return
        row_idx = torch.as_tensor(row_idx, dtype=torch.long, device=linear.weight.device)
        new_weights = torch.empty(
            (len(row_idx), linear.weight.shape[1]),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        nn.init.xavier_uniform_(new_weights)
        linear.weight.data[row_idx, :] = new_weights
        linear.bias.data[row_idx] = 0.0

    @staticmethod
    def _reset_columns(linear, col_idx):
        if len(col_idx) == 0:
            return
        col_idx = torch.as_tensor(col_idx, dtype=torch.long, device=linear.weight.device)
        new_weights = torch.empty(
            (linear.weight.shape[0], len(col_idx)),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        nn.init.xavier_uniform_(new_weights)
        linear.weight.data[:, col_idx] = new_weights

    def selective_reset(self, state_batch, reset_p=None):
        self.eval()
        p = float(self.hp["reset_rate"] if reset_p is None else reset_p)
        p = float(np.clip(p, 1e-5, 1.0))
        importance = self.compute_reset_importance(state_batch)
        linears = self._actor_linear_layers()
        hidden_linears = linears[:-1]

        mean_selected = []
        mean_preserved = []
        reset_counts = []
        for idx, layer in enumerate(hidden_linears):
            out_dim = layer.weight.shape[0]
            reset_count = int(np.clip(round(p * out_dim), 1, out_dim))
            scores = importance[idx] if idx < len(importance) else None
            if scores is None or len(scores) != out_dim:
                chosen = np.arange(reset_count)
                preserved_scores = np.array([], dtype=np.float32)
                selected_scores = np.array([], dtype=np.float32)
            else:
                order = np.argsort(scores)
                chosen = order[:reset_count]
                selected_scores = scores[chosen]
                preserved_scores = scores[order[-reset_count:]]

            self._reset_rows(layer, chosen)
            self._reset_columns(linears[idx + 1], chosen)

            mean_selected.append(float(np.mean(selected_scores)) if selected_scores.size else 0.0)
            mean_preserved.append(float(np.mean(preserved_scores)) if preserved_scores.size else 0.0)
            reset_counts.append(int(reset_count))

        self.train()
        return {
            "mean_reset_importance": float(np.mean(mean_selected)) if mean_selected else 0.0,
            "mean_preserve_importance": float(np.mean(mean_preserved)) if mean_preserved else 0.0,
            "mean_reset_count": float(np.mean(reset_counts)) if reset_counts else 0.0,
        }


class ReplayBuffer:
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done, td_error):
        self.buffer.append((state, action, reward, next_state, done, td_error))

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def clear(self):
        self.buffer.clear()

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, ns, d, td = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
            np.array(td),
        )

    def sample_states_for_distill(
        self,
        batch_size,
        hard=False,
        top_ratio=0.3,
        priorities=None,
        random_mix=0.0,
    ):
        if not self.buffer:
            return None
        items = list(self.buffer)
        take = min(batch_size, len(items))

        if priorities is not None:
            probs = np.asarray(priorities, dtype=np.float64)
            probs = np.clip(probs, 1e-8, None)
            probs = probs / probs.sum()
            priority_take = max(1, int(round(take * (1.0 - random_mix))))
            replace = len(items) < priority_take
            sampled_idx = np.random.choice(len(items), size=priority_take, replace=replace, p=probs)
            picked = [items[i] for i in sampled_idx.tolist()]

            if len(picked) < take:
                remaining = take - len(picked)
                random_idx = np.random.choice(len(items), size=remaining, replace=len(items) < remaining)
                picked.extend(items[i] for i in random_idx.tolist())
            states = [x[0] for x in picked[:take]]
            return np.array(states)

        if not hard:
            picked = random.sample(items, take)
            states = [x[0] for x in picked]
            return np.array(states)

        sorted_buf = sorted(items, key=lambda x: x[5], reverse=True)
        top_k = max(1, int(len(sorted_buf) * top_ratio))
        top_pool = sorted_buf[:top_k]
        picked = random.sample(top_pool, min(take, len(top_pool)))
        if len(picked) < take:
            rest = random.sample(sorted_buf, min(take - len(picked), len(sorted_buf)))
            picked.extend(rest)
        states = [x[0] for x in picked]
        return np.array(states)


def alpha_dkl_loss(pi_tem, pi_theta, alpha):
    kl_forward = dist.kl_divergence(pi_tem, pi_theta).mean()
    kl_backward = dist.kl_divergence(pi_theta, pi_tem).mean()
    return alpha * kl_forward + (1.0 - alpha) * kl_backward


def get_dynamic_alpha(progress, alpha_start, alpha_end, lam):
    p = min(max(progress, 0.0), 1.0)
    tanh_term = np.tanh(lam * (p - 0.5))
    return alpha_start * (1 - tanh_term) / 2 + alpha_end * (1 + tanh_term) / 2


def get_adaptive_alpha(alpha_progress, td_ema, td_ref, hp):
    ratio = td_ema / (td_ref + hp["alpha_eps"])
    ratio = float(np.clip(ratio, 0.5, 2.0))  # 限制信号范围，防止极端环境下失控
    alpha = alpha_progress - hp["alpha_td_k"] * (ratio - 1.0)
    alpha = float(np.clip(alpha, hp["alpha_min"], hp["alpha_max"]))
    return alpha


def update_dual_alpha(alpha_prev, observed_kl, hp):
    target_kl = hp["alpha_target_kl"]
    gap = (observed_kl - target_kl) / max(target_kl, hp["alpha_eps"])
    alpha_next = alpha_prev + hp["alpha_dual_lr"] * gap
    alpha_next = float(np.clip(alpha_next, hp["alpha_min"], hp["alpha_max"]))
    return alpha_next, float(gap)


def blend_alpha(alpha_progress, alpha_dual, hp):
    blend = float(np.clip(hp["alpha_progress_blend"], 0.0, 1.0))
    alpha = blend * alpha_progress + (1.0 - blend) * alpha_dual
    return float(np.clip(alpha, hp["alpha_min"], hp["alpha_max"]))


def update_ema_pair(ema, ref, value, ema_decay, ref_decay):
    if ema is None or ref is None:
        return value, value
    ema = ema_decay * ema + (1.0 - ema_decay) * value
    ref = ref_decay * ref + (1.0 - ref_decay) * value
    return ema, ref


def estimate_td_error(ac, state, reward, next_state, done, gamma):
    with torch.no_grad():
        s = torch.FloatTensor(state)
        ns = torch.FloatTensor(next_state)
        v = ac.get_value(s).squeeze().item()
        nv = ac.get_value(ns).squeeze().item()
        td = reward + gamma * nv * (1 - float(done)) - v
        return float(abs(td))


def batch_update_td_errors(ac, buffer, gamma):
    """Batch-compute TD errors for all buffer items (replaces per-step estimation)."""
    if not buffer.buffer:
        return
    items = list(buffer.buffer)
    states = np.array([x[0] for x in items], dtype=np.float32)
    next_states = np.array([x[3] for x in items], dtype=np.float32)
    rewards = np.array([x[2] for x in items], dtype=np.float32)
    dones = np.array([x[4] for x in items], dtype=np.float32)
    with torch.no_grad():
        v = ac.get_value(torch.FloatTensor(states)).squeeze().numpy()
        nv = ac.get_value(torch.FloatTensor(next_states)).squeeze().numpy()
    td_errors = np.abs(rewards + gamma * nv * (1 - dones) - v)
    for i in range(len(items)):
        s, a, r, ns, d, _ = items[i]
        buffer.buffer[i] = (s, a, r, ns, d, float(td_errors[i]))


def compute_priority_scores(ac, ac_tem, buffer, hp):
    if not buffer.buffer:
        return None, None, None

    states = np.array([x[0] for x in buffer.buffer], dtype=np.float32)
    td_values = np.array([x[5] for x in buffer.buffer], dtype=np.float32)
    state_batch = torch.FloatTensor(states)
    with torch.no_grad():
        pi_new = ac.get_action(state_batch)
        pi_old = ac_tem.get_action(state_batch)
        kl_values = dist.kl_divergence(pi_old, pi_new).sum(dim=1).cpu().numpy()

    td_norm = td_values / (td_values.mean() + hp["metric_eps"])
    kl_norm = kl_values / (kl_values.mean() + hp["metric_eps"])
    scores = hp["priority_td_weight"] * td_norm + hp["priority_kl_weight"] * kl_norm
    scores = np.clip(scores, hp["metric_eps"], None)
    scores = np.power(scores, hp["priority_gamma"])
    return scores, td_values, kl_values


def compute_policy_divergence(ac_ref, ac_new, state_batch):
    with torch.no_grad():
        pi_ref = ac_ref.get_action(state_batch)
        pi_new = ac_new.get_action(state_batch)
        kl_forward = dist.kl_divergence(pi_ref, pi_new).sum(dim=1).mean().item()
        kl_backward = dist.kl_divergence(pi_new, pi_ref).sum(dim=1).mean().item()
    kl_symmetric = 0.5 * (kl_forward + kl_backward)
    return float(kl_forward), float(kl_backward), float(kl_symmetric)


def inner_distill(ac, ac_tem, state_batch, alpha, hp):
    ac.eval()
    ac_tem.eval()
    opt = optim.Adam(ac.parameters(), lr=hp["lr"])
    loss_val = float("inf")
    steps = 0
    max_steps = int(hp.get("distill_max_steps", 80))
    while loss_val > hp["distill_loss_bound"] and steps < max_steps:
        opt.zero_grad()
        pi_new = ac.get_action(state_batch)
        pi_old = ac_tem.get_action(state_batch)
        loss = alpha_dkl_loss(pi_old, pi_new, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), hp["clip_grad_norm"])
        opt.step()
        loss_val = float(loss.item())
        steps += 1
    ac.train()
    return ac, loss_val, steps


def ppo_update(ac, buffer, hp, optimizer):
    states, actions, rewards, next_states, dones, _ = buffer.get_batch()
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    with torch.no_grad():
        old_pi = ac.get_action(states)
        old_log_probs = old_pi.log_prob(actions).sum(dim=1)
        values = ac.get_value(states).squeeze()
        next_values = ac.get_value(next_states).squeeze()
        deltas = rewards + hp["gamma"] * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        adv = 0.0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + hp["gamma"] * 0.95 * adv * (1 - dones[t])
            advantages[t] = adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards + hp["gamma"] * next_values * (1 - dones)

    last_actor = 0.0
    last_critic = 0.0
    for _ in range(hp["epochs_per_update"]):
        pi = ac.get_action(states)
        log_probs = pi.log_prob(actions).sum(dim=1)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - hp["clip_range"], 1 + hp["clip_range"]) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(ac.get_value(states).squeeze(), returns)
        total_loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), hp["clip_grad_norm"])
        optimizer.step()

        last_actor = float(actor_loss.item())
        last_critic = float(critic_loss.item())

    mean_abs_td = float(torch.mean(torch.abs(deltas)).item())
    with torch.no_grad():
        new_pi = ac.get_action(states)
        mean_entropy = float(new_pi.entropy().sum(dim=1).mean().item())
        mean_policy_shift = float(
            dist.kl_divergence(old_pi, new_pi).sum(dim=1).mean().item()
        )
    return last_actor, last_critic, mean_abs_td, mean_entropy, mean_policy_shift


def compute_adaptive_reset_rate(base_r, td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, hp):
    td_signal = np.clip(td_ema / (td_ref + hp["metric_eps"]) - 1.0, -1.0, 1.0)
    kl_signal = np.clip(kl_ema / (kl_ref + hp["metric_eps"]) - 1.0, -1.0, 1.0)
    ent_signal = np.clip(ent_ref / (ent_ema + hp["metric_eps"]) - 1.0, -1.0, 1.0)
    combined_signal = (
        hp["reset_td_weight"] * td_signal
        + hp["reset_kl_weight"] * kl_signal
        + hp["reset_ent_weight"] * ent_signal
    )
    rate = base_r * (1.0 + hp["reset_k"] * combined_signal)
    rate = float(np.clip(rate, hp["reset_rate_min"], hp["reset_rate_max"]))
    return rate, td_signal, kl_signal, ent_signal, combined_signal


def compute_sbp_event_signal(td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, hp):
    td_signal = td_ema / (td_ref + hp["metric_eps"]) - 1.0
    kl_signal = kl_ema / (kl_ref + hp["metric_eps"]) - 1.0
    ent_signal = ent_ref / (ent_ema + hp["metric_eps"]) - 1.0
    combined_signal = (
        hp["reset_td_weight"] * td_signal
        + hp["reset_kl_weight"] * kl_signal
        + hp["reset_ent_weight"] * ent_signal
    )
    return td_signal, kl_signal, ent_signal, combined_signal


def should_trigger_sbp(total_steps, last_sbp_step, td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, cfg, hp):
    since_last = total_steps - last_sbp_step
    if total_steps < hp["sbp_warmup_steps"]:
        return False, "warmup", 0.0, 0.0, 0.0, 0.0
    if since_last < hp["sbp_min_interval"]:
        return False, "cooldown", 0.0, 0.0, 0.0, 0.0

    td_signal, kl_signal, ent_signal, combined_signal = compute_sbp_event_signal(
        td_ema if td_ema is not None else 1.0,
        td_ref if td_ref is not None else 1.0,
        kl_ema if kl_ema is not None else 1.0,
        kl_ref if kl_ref is not None else 1.0,
        ent_ema if ent_ema is not None else 1.0,
        ent_ref if ent_ref is not None else 1.0,
        hp,
    )

    if cfg.get("event_trigger_sbp", False):
        if combined_signal >= hp["sbp_event_threshold"]:
            return True, "event", td_signal, kl_signal, ent_signal, combined_signal
        if since_last >= hp["sbp_force_interval"]:
            return True, "force", td_signal, kl_signal, ent_signal, combined_signal
        return False, "waiting", td_signal, kl_signal, ent_signal, combined_signal

    if since_last >= hp["reset_frequency"]:
        return True, "periodic", td_signal, kl_signal, ent_signal, combined_signal
    return False, "periodic_wait", td_signal, kl_signal, ent_signal, combined_signal


def train_one(env_name, seed, algo_name, cfg, hp):
    seed_everything(seed)
    hp = dict(hp)

    run_dir = os.path.join(RUNS_DIR, f"{env_name}_{algo_name}_seed{seed}_steps{int(hp['train_steps'])}")
    os.makedirs(run_dir, exist_ok=True)
    logger = ExperimentLogger(
        run_dir,
        {
            "env": env_name,
            "algo": algo_name,
            "seed": seed,
            "hp": hp,
            "settings": cfg,
        },
    )

    env = gym.make(env_name)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    ac = ActorCritic(state_dim, action_dim, action_bound, hp)
    optimizer = optim.Adam(ac.parameters(), lr=hp["lr"])
    buffer = ReplayBuffer(hp["buffer_size"], hp["batch_size"])

    total_steps = 0
    episode_idx = 0
    episode_reward = 0.0
    recent_rewards = deque(maxlen=10)

    td_ema = None
    td_ref = None
    ent_ema = None
    ent_ref = None
    kl_ema = None
    kl_ref = None

    start_time = time.time()
    last_speed_time = start_time
    last_speed_step = 0
    last_sbp_step = 0
    alpha_dual_state = float(np.clip(hp["alpha_start"], hp["alpha_min"], hp["alpha_max"]))

    while total_steps < int(hp["train_steps"]):
        action_dist = ac.get_action(torch.FloatTensor(state))
        action = action_dist.sample().detach().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, next_state, done, 0.0)

        state = next_state
        episode_reward += reward
        total_steps += 1

        if total_steps - last_speed_step >= 10_000:
            now = time.time()
            sps = (total_steps - last_speed_step) / max(1e-6, (now - last_speed_time))
            logger.log(total_steps, "speed", round(sps, 2), round((now - start_time) / 60, 2), algo_name, env_name, seed)
            last_speed_time = now
            last_speed_step = total_steps

        need_sbp_check = False
        if buffer.is_full():
            batch_update_td_errors(ac, buffer, hp["gamma"])
            actor_loss, critic_loss, mean_abs_td, mean_entropy, mean_policy_shift = ppo_update(
                ac, buffer, hp, optimizer
            )
            td_ema, td_ref = update_ema_pair(
                td_ema, td_ref, mean_abs_td, hp["td_ema_decay"], hp["td_ref_decay"]
            )
            ent_ema, ent_ref = update_ema_pair(
                ent_ema, ent_ref, mean_entropy, hp["td_ema_decay"], hp["td_ref_decay"]
            )
            kl_ema, kl_ref = update_ema_pair(
                kl_ema, kl_ref, mean_policy_shift, hp["td_ema_decay"], hp["td_ref_decay"]
            )
            logger.log(total_steps, "loss", actor_loss, critic_loss, algo_name, env_name, seed)
            need_sbp_check = True

        trigger_sbp = False
        trigger_mode = "skip"
        td_signal = kl_signal = ent_signal = combined_signal = 0.0
        if need_sbp_check:
            trigger_sbp, trigger_mode, td_signal, kl_signal, ent_signal, combined_signal = should_trigger_sbp(
                total_steps,
                last_sbp_step,
                td_ema,
                td_ref,
                kl_ema,
                kl_ref,
                ent_ema,
                ent_ref,
                cfg,
                hp,
            )

        if trigger_sbp:
            progress = total_steps / float(hp["train_steps"])
            alpha_static = hp["alpha_dkl"]
            alpha_dynamic = get_dynamic_alpha(progress, hp["alpha_start"], hp["alpha_end"], hp["alpha_lambda"])
            alpha_base = alpha_dynamic if cfg["alpha_mode"] == "dynamic" else alpha_static
            alpha_td_ema = td_ema if td_ema is not None else 1.0
            alpha_td_ref = td_ref if td_ref is not None else max(alpha_td_ema, 1.0)
            if cfg.get("adaptive_alpha", False) and cfg["alpha_mode"] == "dynamic":
                alpha_base = get_adaptive_alpha(alpha_dynamic, alpha_td_ema, alpha_td_ref, hp)

            current_reset_rate = hp["reset_rate"]
            if cfg["adaptive_reset"]:
                current_reset_rate, td_signal, kl_signal, ent_signal, combined_signal = (
                    compute_adaptive_reset_rate(
                        hp["reset_rate"],
                        td_ema if td_ema is not None else 1.0,
                        td_ref if td_ref is not None else 1.0,
                        kl_ema if kl_ema is not None else 1.0,
                        kl_ref if kl_ref is not None else 1.0,
                        ent_ema if ent_ema is not None else 1.0,
                        ent_ref if ent_ref is not None else 1.0,
                        hp,
                    )
                )

            # ── Module coordination: α低→策略稳定→降低重置/蒸馏强度 ──
            if cfg.get("dual_alpha", False):
                coord = alpha_dual_state / hp["alpha_max"]  # 0~1
                coord = float(np.clip(coord, 0.2, 1.0))     # 最低保留20%
                current_reset_rate *= coord

            reset_states_np = buffer.sample_states_for_distill(hp["batch_size"], hard=False, top_ratio=hp["hard_top_ratio"])
            if reset_states_np is None:
                reset_states_np = np.array([env.observation_space.sample() for _ in range(hp["batch_size"])])
            reset_state_batch = torch.FloatTensor(reset_states_np)

            ac_tem = ActorCritic(state_dim, action_dim, action_bound, hp)
            ac_tem.load_state_dict(ac.state_dict())

            reset_stats = {
                "mean_reset_importance": 0.0,
                "mean_preserve_importance": 0.0,
                "mean_reset_count": 0.0,
            }
            if cfg.get("selective_reset", False):
                reset_stats = ac.selective_reset(reset_state_batch, current_reset_rate)
            else:
                ac.cycle_reset(current_reset_rate)

            pre_forward_kl, pre_backward_kl, pre_symmetric_kl = compute_policy_divergence(
                ac_tem, ac, reset_state_batch
            )

            observed_kl = pre_symmetric_kl
            alpha_dual_gap = 0.0
            alpha_using = alpha_base
            if cfg.get("dual_alpha", False):
                alpha_dual_state, alpha_dual_gap = update_dual_alpha(alpha_dual_state, observed_kl, hp)
                alpha_using = blend_alpha(alpha_base, alpha_dual_state, hp)
            else:
                alpha_dual_state = alpha_using

            hard = cfg["hard_distill"]
            priorities = None
            mean_priority = 0.0
            mean_priority_kl = 0.0
            if hard:
                priorities, _, kl_values = compute_priority_scores(ac, ac_tem, buffer, hp)
                if priorities is not None:
                    mean_priority = float(np.mean(priorities))
                if kl_values is not None:
                    mean_priority_kl = float(np.mean(kl_values))
            states_np = buffer.sample_states_for_distill(
                hp["batch_size"],
                hard=hard,
                top_ratio=hp["hard_top_ratio"],
                priorities=priorities,
                random_mix=hp["distill_mix_random"],
            )
            if states_np is None:
                states_np = np.array([env.observation_space.sample() for _ in range(hp["batch_size"])])
            state_batch = torch.FloatTensor(states_np)

            # ── Module coordination: 蒸馏步数也随 α 协调 ──
            distill_hp = dict(hp)
            if cfg.get("dual_alpha", False):
                coord = alpha_using / hp["alpha_max"]
                coord = float(np.clip(coord, 0.3, 1.0))
                distill_hp["distill_max_steps"] = max(10, int(hp["distill_max_steps"] * coord))
            ac, distill_loss, distill_steps = inner_distill(ac, ac_tem, state_batch, alpha_using, distill_hp)
            post_forward_kl, post_backward_kl, post_symmetric_kl = compute_policy_divergence(
                ac_tem, ac, state_batch
            )

            logger.log(
                total_steps,
                "sbp",
                round(alpha_static, 4),
                round(alpha_dynamic, 4),
                round(alpha_using, 4),
                algo_name,
                env_name,
                seed,
                round(current_reset_rate, 5),
                int(hard),
                round(td_ema if td_ema is not None else 0.0, 6),
                round(td_ref if td_ref is not None else 0.0, 6),
                round(kl_ema if kl_ema is not None else 0.0, 6),
                round(kl_ref if kl_ref is not None else 0.0, 6),
                round(ent_ema if ent_ema is not None else 0.0, 6),
                round(ent_ref if ent_ref is not None else 0.0, 6),
                round(td_signal, 6),
                round(kl_signal, 6),
                round(ent_signal, 6),
                round(combined_signal, 6),
                round(mean_priority, 6),
                round(mean_priority_kl, 6),
                round(distill_loss, 6),
                distill_steps,
                trigger_mode,
                round(observed_kl, 6),
                round(hp["alpha_target_kl"], 6),
                round(alpha_dual_state, 6),
                round(alpha_dual_gap, 6),
                round(alpha_base, 6),
                round(pre_forward_kl, 6),
                round(pre_backward_kl, 6),
                round(pre_symmetric_kl, 6),
                round(post_forward_kl, 6),
                round(post_backward_kl, 6),
                round(post_symmetric_kl, 6),
                int(cfg.get("selective_reset", False)),
                round(reset_stats["mean_reset_importance"], 6),
                round(reset_stats["mean_preserve_importance"], 6),
                round(reset_stats["mean_reset_count"], 3),
            )
            last_sbp_step = total_steps

        if need_sbp_check:
            buffer.clear()

        if done:
            episode_idx += 1
            recent_rewards.append(episode_reward)
            avg10 = float(np.mean(recent_rewards))
            logger.log(total_steps, "reward", episode_idx, float(episode_reward), round(avg10, 3), algo_name, env_name, seed)
            episode_reward = 0.0
            state, _ = env.reset()

    env.close()


def run_suite(envs, seeds, hp, dry_run=False, only_algos=None, force=False):
    os.makedirs(RUNS_DIR, exist_ok=True)
    selected_algos = (
        {name: ALGO_SETTINGS[name] for name in only_algos}
        if only_algos
        else ALGO_SETTINGS
    )
    for env_name in envs:
        for algo_name, cfg in selected_algos.items():
            for seed in seeds:
                run_dir = os.path.join(RUNS_DIR, f"{env_name}_{algo_name}_seed{seed}_steps{int(hp['train_steps'])}")
                if dry_run:
                    print("[plan]", run_dir)
                    continue
                metrics_path = os.path.join(run_dir, "metrics.csv")
                if (not force) and os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 64:
                    print(f"[skip] {run_dir}")
                    continue
                print(f"[run] env={env_name}, algo={algo_name}, seed={seed}")
                train_one(env_name, seed, algo_name, cfg, hp)


def parse_args():
    parser = argparse.ArgumentParser(description="Run P3O extra innovations experiments.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke setup: 1 env, 1 seed, 200k steps")
    parser.add_argument("--dry-run", action="store_true", help="Only print run plan")
    parser.add_argument(
        "--only-algo",
        action="append",
        choices=list(ALGO_SETTINGS.keys()),
        help="Run only selected algorithm(s). Can be repeated.",
    )
    parser.add_argument(
        "--env",
        action="append",
        help="Run only selected env(s). Can be repeated, e.g. --env Hopper-v4 --env Walker2d-v4",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Run only selected seed(s). Can be repeated, e.g. --seed 0 --seed 1",
    )
    parser.add_argument("--steps", type=int, help="Override train steps for all selected envs")
    parser.add_argument("--runs-dir", type=str, help="Override output directory for runs")
    parser.add_argument("--force", action="store_true", help="Force rerun even if metrics.csv exists")
    return parser.parse_args()


def main():
    args = parse_args()
    global RUNS_DIR

    hp = dict(HP)
    envs = ["Hopper-v4", "Walker2d-v4"]
    seeds = [0, 1]

    if args.quick:
        hp["train_steps"] = 200_000
        envs = ["Hopper-v4"]
        seeds = [0]
    if args.env:
        envs = args.env
    if args.seed:
        seeds = args.seed
    if args.steps is not None:
        hp["train_steps"] = int(args.steps)
    if args.runs_dir:
        RUNS_DIR = args.runs_dir

    run_suite(
        envs,
        seeds,
        hp,
        dry_run=args.dry_run,
        only_algos=args.only_algo,
        force=args.force,
    )


if __name__ == "__main__":
    main()

