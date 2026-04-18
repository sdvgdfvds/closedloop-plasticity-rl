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

from run_p3o_extra_innovations import (
    RUNS_DIR,
    blend_alpha,
    get_adaptive_alpha,
    get_dynamic_alpha,
    update_dual_alpha,
)


HP = {
    "lr": 3e-4,
    "buffer_size": 4096,
    "batch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "train_steps_per_phase": 80_000,
    "epochs_per_update": 10,
    "clip_range": 0.2,
    "clip_grad_norm": 0.5,
    "hidden_size": 256,
    "hidden_layers": 3,
    "activation": nn.Tanh,
    "reset_rate": 0.01,
    "reset_frequency": 30_000,
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
    "distill_max_steps": 60,
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
    "anchor_memory_size": 2048,
    "anchor_store_ratio": 0.25,
    "anchor_batch_ratio": 0.30,
    "anchor_loss_weight": 0.45,
    "anchor_random_mix": 0.20,
    "anchor_priority_power": 1.25,
    "sbp_warmup_steps": 10_000,
    "sbp_min_interval": 10_000,
    "sbp_force_interval": 40_000,
    "sbp_event_threshold": 0.35,
    "eval_every": 10_000,
    "eval_episodes": 5,
}

DEFAULT_SEQUENCE = ["Hopper-v4", "Walker2d-v4", "Hopper-v4"]
ALGO_SETTINGS = {
    "PPO": {
        "use_sbp": False,
        "alpha_mode": "fixed",
        "adaptive_alpha": False,
        "dual_alpha": False,
        "adaptive_reset": False,
        "hard_distill": False,
        "selective_reset": False,
        "event_trigger_sbp": False,
        "anchor_memory": False,
    },
    "P3O": {
        "use_sbp": True,
        "alpha_mode": "fixed",
        "adaptive_alpha": False,
        "dual_alpha": False,
        "adaptive_reset": False,
        "hard_distill": False,
        "selective_reset": False,
        "event_trigger_sbp": False,
        "anchor_memory": False,
    },
    "P3O-ClosedLoopFull": {
        "use_sbp": True,
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "selective_reset": True,
        "event_trigger_sbp": True,
        "anchor_memory": False,
    },
    "P3O-ClosedLoopMemory": {
        "use_sbp": True,
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": True,
        "hard_distill": True,
        "selective_reset": True,
        "event_trigger_sbp": True,
        "anchor_memory": True,
    },
    "P3O-DualAlpha+Memory": {
        "use_sbp": True,
        "alpha_mode": "dynamic",
        "adaptive_alpha": True,
        "dual_alpha": True,
        "adaptive_reset": False,
        "hard_distill": False,
        "selective_reset": False,
        "event_trigger_sbp": False,
        "anchor_memory": True,
    },
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SequenceLogger:
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


class UniversalActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super().__init__()
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

    def get_action(self, state, action_scale):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action_scale.dim() == 1:
            action_scale = action_scale.unsqueeze(0).expand(state.shape[0], -1)
        mean = torch.tanh(self.actor(state)) * action_scale
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return dist.Normal(mean, std)

    def get_value(self, state):
        return self.critic(state)

    def _actor_linear_layers(self):
        return [layer for layer in self.actor if isinstance(layer, nn.Linear)]

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
        return {"mean_reset_importance": 0.0, "mean_preserve_importance": 0.0, "mean_reset_count": 0.0}

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
        new_weights = torch.empty((len(row_idx), linear.weight.shape[1]), device=linear.weight.device, dtype=linear.weight.dtype)
        nn.init.xavier_uniform_(new_weights)
        linear.weight.data[row_idx, :] = new_weights
        linear.bias.data[row_idx] = 0.0

    @staticmethod
    def _reset_columns(linear, col_idx):
        if len(col_idx) == 0:
            return
        col_idx = torch.as_tensor(col_idx, dtype=torch.long, device=linear.weight.device)
        new_weights = torch.empty((linear.weight.shape[0], len(col_idx)), device=linear.weight.device, dtype=linear.weight.dtype)
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

    def add(self, state, action, reward, next_state, done, td_error, action_scale, action_mask):
        self.buffer.append((state, action, reward, next_state, done, td_error, action_scale, action_mask))

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def clear(self):
        self.buffer.clear()

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, ns, d, td, scale, mask = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
            np.array(td),
            np.array(scale),
            np.array(mask),
        )

    def sample_states_for_distill(self, batch_size, priorities=None, random_mix=0.0):
        if not self.buffer:
            return None
        items = list(self.buffer)
        take = min(batch_size, len(items))
        if priorities is None:
            picked = random.sample(items, take)
            return np.array([x[0] for x in picked])

        probs = np.asarray(priorities, dtype=np.float64)
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()
        priority_take = max(1, int(round(take * (1.0 - random_mix))))
        replace = len(items) < priority_take
        idx = np.random.choice(len(items), size=priority_take, replace=replace, p=probs)
        picked = [items[i] for i in idx.tolist()]
        if len(picked) < take:
            remain = take - len(picked)
            rand_idx = np.random.choice(len(items), size=remain, replace=len(items) < remain)
            picked.extend(items[i] for i in rand_idx.tolist())
        return np.array([x[0] for x in picked[:take]])


class AnchorMemoryBank:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []

    def __len__(self):
        return len(self.items)

    def add_batch(self, states, ref_mean, ref_std, priorities, action_scales, action_masks, phase_idx):
        if states is None or len(states) == 0:
            return 0, 0

        inserted = 0
        replaced = 0
        for i in range(len(states)):
            item = {
                "state": np.asarray(states[i], dtype=np.float32),
                "ref_mean": np.asarray(ref_mean[i], dtype=np.float32),
                "ref_std": np.asarray(ref_std[i], dtype=np.float32),
                "priority": float(priorities[i]),
                "action_scale": np.asarray(action_scales[i], dtype=np.float32),
                "action_mask": np.asarray(action_masks[i], dtype=np.float32),
                "phase_idx": int(phase_idx),
            }
            if len(self.items) < self.capacity:
                self.items.append(item)
                inserted += 1
                continue

            min_idx = min(range(len(self.items)), key=lambda j: self.items[j]["priority"])
            if item["priority"] > self.items[min_idx]["priority"]:
                self.items[min_idx] = item
                replaced += 1
            elif random.random() < 0.05:
                swap_idx = random.randrange(len(self.items))
                self.items[swap_idx] = item
                replaced += 1
        return inserted, replaced

    def sample(self, batch_size, random_mix=0.2, priority_power=1.0):
        if not self.items:
            return None

        take = min(int(batch_size), len(self.items))
        items = self.items
        priority_take = max(1, int(round(take * (1.0 - random_mix))))
        random_take = max(0, take - priority_take)

        priorities = np.asarray([max(x["priority"], 1e-8) for x in items], dtype=np.float64)
        probs = np.power(priorities, priority_power)
        probs = probs / probs.sum()

        chosen_idx = np.random.choice(len(items), size=priority_take, replace=len(items) < priority_take, p=probs)
        chosen = [items[i] for i in chosen_idx.tolist()]
        if random_take > 0:
            rand_idx = np.random.choice(len(items), size=random_take, replace=len(items) < random_take)
            chosen.extend(items[i] for i in rand_idx.tolist())

        return {
            "states": np.asarray([x["state"] for x in chosen], dtype=np.float32),
            "ref_mean": np.asarray([x["ref_mean"] for x in chosen], dtype=np.float32),
            "ref_std": np.asarray([x["ref_std"] for x in chosen], dtype=np.float32),
            "priorities": np.asarray([x["priority"] for x in chosen], dtype=np.float32),
            "action_scales": np.asarray([x["action_scale"] for x in chosen], dtype=np.float32),
            "action_masks": np.asarray([x["action_mask"] for x in chosen], dtype=np.float32),
            "phase_ids": np.asarray([x["phase_idx"] for x in chosen], dtype=np.int64),
        }


def pad_vector(vec, dim, fill=0.0):
    arr = np.full(dim, fill, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    arr[: len(vec)] = vec
    return arr


def build_env_meta(env_name, state_dim_max, action_dim_max):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    meta = {
        "env_name": env_name,
        "env": env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_scale": pad_vector(action_high, action_dim_max, fill=1.0),
        "action_mask": pad_vector(np.ones(action_dim, dtype=np.float32), action_dim_max, fill=0.0),
        "action_low": action_low,
        "action_high_raw": action_high,
        "state_dim_max": state_dim_max,
        "action_dim_max": action_dim_max,
    }
    return meta


def pad_state(state, meta):
    return pad_vector(state, meta["state_dim_max"], fill=0.0)


def action_to_env(action_full, meta):
    action = np.asarray(action_full[: meta["action_dim"]], dtype=np.float32)
    return np.clip(action, meta["action_low"], meta["action_high_raw"])


def update_ema_pair(ema, ref, value, ema_decay, ref_decay):
    if ema is None or ref is None:
        return value, value
    ema = ema_decay * ema + (1.0 - ema_decay) * value
    ref = ref_decay * ref + (1.0 - ref_decay) * value
    return ema, ref


def compute_adaptive_reset_rate(base_r, td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, hp):
    td_signal = td_ema / (td_ref + hp["metric_eps"]) - 1.0
    kl_signal = kl_ema / (kl_ref + hp["metric_eps"]) - 1.0
    ent_signal = ent_ref / (ent_ema + hp["metric_eps"]) - 1.0
    combined_signal = (
        hp["reset_td_weight"] * td_signal
        + hp["reset_kl_weight"] * kl_signal
        + hp["reset_ent_weight"] * ent_signal
    )
    rate = base_r * (1.0 + hp["reset_k"] * combined_signal)
    rate = float(np.clip(rate, hp["reset_rate_min"], hp["reset_rate_max"]))
    return rate, td_signal, kl_signal, ent_signal, combined_signal


def should_trigger_sbp(global_step, last_sbp_step, td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, cfg, hp):
    since_last = global_step - last_sbp_step
    if not cfg.get("use_sbp", True):
        return False, "disabled", 0.0, 0.0, 0.0, 0.0
    if global_step < hp["sbp_warmup_steps"]:
        return False, "warmup", 0.0, 0.0, 0.0, 0.0
    if since_last < hp["sbp_min_interval"]:
        return False, "cooldown", 0.0, 0.0, 0.0, 0.0
    td_signal = td_ema / (td_ref + hp["metric_eps"]) - 1.0 if td_ema is not None else 0.0
    kl_signal = kl_ema / (kl_ref + hp["metric_eps"]) - 1.0 if kl_ema is not None else 0.0
    ent_signal = ent_ref / (ent_ema + hp["metric_eps"]) - 1.0 if ent_ema is not None else 0.0
    combined_signal = (
        hp["reset_td_weight"] * td_signal
        + hp["reset_kl_weight"] * kl_signal
        + hp["reset_ent_weight"] * ent_signal
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


def estimate_td_error(ac, state, reward, next_state, done):
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0)
        ns = torch.FloatTensor(next_state).unsqueeze(0)
        v = ac.get_value(s).squeeze().item()
        nv = ac.get_value(ns).squeeze().item()
        td = reward + HP["gamma"] * nv * (1.0 - float(done)) - v
    return float(abs(td))


def compute_priority_scores(ac, ac_tem, buffer, hp):
    if not buffer.buffer:
        return None, None, None
    states = np.array([x[0] for x in buffer.buffer], dtype=np.float32)
    td_values = np.array([x[5] for x in buffer.buffer], dtype=np.float32)
    action_scales = np.array([x[6] for x in buffer.buffer], dtype=np.float32)
    action_masks = np.array([x[7] for x in buffer.buffer], dtype=np.float32)
    state_batch = torch.FloatTensor(states)
    scale_batch = torch.FloatTensor(action_scales)
    mask_batch = torch.FloatTensor(action_masks)
    with torch.no_grad():
        pi_new = ac.get_action(state_batch, scale_batch)
        pi_old = ac_tem.get_action(state_batch, scale_batch)
        kl_values = (dist.kl_divergence(pi_old, pi_new) * mask_batch).sum(dim=1).cpu().numpy()
    td_norm = td_values / (td_values.mean() + hp["metric_eps"])
    kl_norm = kl_values / (kl_values.mean() + hp["metric_eps"])
    scores = hp["priority_td_weight"] * td_norm + hp["priority_kl_weight"] * kl_norm
    scores = np.clip(scores, hp["metric_eps"], None)
    scores = np.power(scores, hp["priority_gamma"])
    return scores, td_values, kl_values


def alpha_dkl_loss(pi_tem, pi_theta, alpha, action_mask, sample_weight=None):
    kl_forward = (dist.kl_divergence(pi_tem, pi_theta) * action_mask).sum(dim=1)
    kl_backward = (dist.kl_divergence(pi_theta, pi_tem) * action_mask).sum(dim=1)
    loss_vec = alpha * kl_forward + (1.0 - alpha) * kl_backward
    if sample_weight is not None:
        weight = sample_weight / (sample_weight.mean() + 1e-8)
        return (loss_vec * weight).mean()
    return loss_vec.mean()


def build_anchor_candidates(ac_ref, buffer, hp, phase_idx, priorities=None):
    if not buffer.buffer:
        return None

    items = list(buffer.buffer)
    take = max(1, min(len(items), int(round(hp["batch_size"] * hp["anchor_store_ratio"]))))
    if priorities is not None:
        probs = np.asarray(priorities, dtype=np.float64)
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()
        idx = np.random.choice(len(items), size=take, replace=len(items) < take, p=probs)
    else:
        idx = np.random.choice(len(items), size=take, replace=len(items) < take)

    selected = [items[i] for i in idx.tolist()]
    states = np.asarray([x[0] for x in selected], dtype=np.float32)
    action_scales = np.asarray([x[6] for x in selected], dtype=np.float32)
    action_masks = np.asarray([x[7] for x in selected], dtype=np.float32)
    if priorities is not None:
        chosen_priorities = np.asarray([priorities[i] for i in idx.tolist()], dtype=np.float32)
    else:
        chosen_priorities = np.asarray([x[5] for x in selected], dtype=np.float32)

    state_batch = torch.FloatTensor(states)
    scale_batch = torch.FloatTensor(action_scales)
    with torch.no_grad():
        ref_pi = ac_ref.get_action(state_batch, scale_batch)
        ref_mean = ref_pi.mean.cpu().numpy()
        ref_std = ref_pi.stddev.cpu().numpy()

    return {
        "states": states,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "priorities": chosen_priorities,
        "action_scales": action_scales,
        "action_masks": action_masks,
        "phase_idx": phase_idx,
    }


def inner_distill(ac, ac_tem, state_batch, action_scale, action_mask, alpha, hp, anchor_batch=None):
    ac.eval()
    ac_tem.eval()
    opt = optim.Adam(ac.parameters(), lr=hp["lr"])
    loss_val = float("inf")
    steps = 0
    last_current_loss = 0.0
    last_anchor_loss = 0.0
    while loss_val > hp["distill_loss_bound"] and steps < int(hp["distill_max_steps"]):
        opt.zero_grad()
        pi_new = ac.get_action(state_batch, action_scale)
        pi_old = ac_tem.get_action(state_batch, action_scale)
        current_loss = alpha_dkl_loss(pi_old, pi_new, alpha, action_mask)
        anchor_loss = torch.tensor(0.0)
        if anchor_batch is not None and anchor_batch["states"].shape[0] > 0:
            anchor_states = torch.FloatTensor(anchor_batch["states"])
            anchor_scales = torch.FloatTensor(anchor_batch["action_scales"])
            anchor_masks = torch.FloatTensor(anchor_batch["action_masks"])
            ref_mean = torch.FloatTensor(anchor_batch["ref_mean"])
            ref_std = torch.FloatTensor(anchor_batch["ref_std"])
            ref_pi = dist.Normal(ref_mean, ref_std)
            new_anchor_pi = ac.get_action(anchor_states, anchor_scales)
            sample_weight = torch.FloatTensor(anchor_batch["priorities"])
            anchor_loss = alpha_dkl_loss(ref_pi, new_anchor_pi, alpha, anchor_masks, sample_weight=sample_weight)
        loss = current_loss + hp["anchor_loss_weight"] * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), hp["clip_grad_norm"])
        opt.step()
        loss_val = float(loss.item())
        last_current_loss = float(current_loss.item())
        last_anchor_loss = float(anchor_loss.item()) if isinstance(anchor_loss, torch.Tensor) else float(anchor_loss)
        steps += 1
    ac.train()
    return ac, loss_val, steps, last_current_loss, last_anchor_loss


def ppo_update(ac, buffer, hp, optimizer):
    states, actions, rewards, next_states, dones, _, action_scales, action_masks = buffer.get_batch()
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    action_scales = torch.FloatTensor(action_scales)
    action_masks = torch.FloatTensor(action_masks)

    with torch.no_grad():
        old_pi = ac.get_action(states, action_scales)
        old_log_probs = (old_pi.log_prob(actions) * action_masks).sum(dim=1)
        values = ac.get_value(states).squeeze()
        next_values = ac.get_value(next_states).squeeze()
        deltas = rewards + hp["gamma"] * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        adv = 0.0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + hp["gamma"] * hp["gae_lambda"] * adv * (1 - dones[t])
            advantages[t] = adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards + hp["gamma"] * next_values * (1 - dones)

    last_actor = 0.0
    last_critic = 0.0
    for _ in range(hp["epochs_per_update"]):
        pi = ac.get_action(states, action_scales)
        log_probs = (pi.log_prob(actions) * action_masks).sum(dim=1)
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

    with torch.no_grad():
        new_pi = ac.get_action(states, action_scales)
        mean_entropy = float((new_pi.entropy() * action_masks).sum(dim=1).mean().item())
        mean_policy_shift = float((dist.kl_divergence(old_pi, new_pi) * action_masks).sum(dim=1).mean().item())
        mean_abs_td = float(torch.mean(torch.abs(deltas)).item())
    return last_actor, last_critic, mean_abs_td, mean_entropy, mean_policy_shift


def evaluate_policy(ac, meta, episodes=5, seed=0):
    env = gym.make(meta["env_name"])
    rewards = []
    action_scale = torch.FloatTensor(meta["action_scale"])
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            state_pad = pad_state(state, meta)
            with torch.no_grad():
                pi = ac.get_action(torch.FloatTensor(state_pad), action_scale)
                action_full = pi.mean.squeeze(0).cpu().numpy()
            action = action_to_env(action_full, meta)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return float(np.mean(rewards)) if rewards else 0.0


def sequence_name(sequence):
    return "_to_".join(sequence)


def run_sequence(sequence, seed, hp, algo_name, cfg, quick=False, force=False):
    seed_everything(seed)
    probe_envs = [gym.make(name) for name in sequence]
    state_dim_max = max(env.observation_space.shape[0] for env in probe_envs)
    action_dim_max = max(env.action_space.shape[0] for env in probe_envs)
    for env in probe_envs:
        env.close()

    metas = {name: build_env_meta(name, state_dim_max, action_dim_max) for name in set(sequence)}
    ac = UniversalActorCritic(state_dim_max, action_dim_max, hp)
    optimizer = optim.Adam(ac.parameters(), lr=hp["lr"])
    buffer = ReplayBuffer(hp["buffer_size"], hp["batch_size"])
    anchor_memory = AnchorMemoryBank(hp["anchor_memory_size"])
    phase_steps = int(hp["train_steps_per_phase"])
    total_steps = 0
    alpha_dual_state = float(np.clip(hp["alpha_start"], hp["alpha_min"], hp["alpha_max"]))

    run_dir = os.path.join(
        RUNS_DIR,
        f"sequence_{algo_name}_{sequence_name(sequence)}_seed{seed}_steps{phase_steps}"
    )
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if (not force) and os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 64:
        print(f"[skip] {run_dir}")
        return run_dir
    logger = SequenceLogger(
        run_dir,
        {
            "algo": algo_name,
            "settings": cfg,
            "sequence": sequence,
            "seed": seed,
            "state_dim_max": state_dim_max,
            "action_dim_max": action_dim_max,
            "hp": hp,
        },
    )

    best_eval = {}
    first_phase_best = {}
    last_sbp_step = 0
    start_time = time.time()

    for phase_idx, env_name in enumerate(sequence, 1):
        meta = metas[env_name]
        env = meta["env"]
        state, _ = env.reset(seed=seed + phase_idx)
        state = pad_state(state, meta)
        episode_reward = 0.0
        recent_rewards = deque(maxlen=10)
        phase_step = 0
        td_ema = td_ref = ent_ema = ent_ref = kl_ema = kl_ref = None
        buffer.clear()
        action_scale = torch.FloatTensor(meta["action_scale"])
        action_mask = torch.FloatTensor(meta["action_mask"])

        logger.log(total_steps, "phase_start", phase_idx, env_name, phase_steps)

        while phase_step < phase_steps:
            pi = ac.get_action(torch.FloatTensor(state), action_scale)
            action_full = pi.sample().detach().squeeze(0).numpy()
            env_action = action_to_env(action_full, meta)
            next_raw_state, reward, terminated, truncated, _ = env.step(env_action)
            next_state = pad_state(next_raw_state, meta)
            done = terminated or truncated

            full_action = pad_vector(env_action, action_dim_max, fill=0.0)
            td_error = estimate_td_error(ac, state, reward, next_state, done)
            buffer.add(state, full_action, reward, next_state, done, td_error, meta["action_scale"], meta["action_mask"])

            state = next_state
            episode_reward += reward
            phase_step += 1
            total_steps += 1

            if buffer.is_full():
                actor_loss, critic_loss, mean_abs_td, mean_entropy, mean_policy_shift = ppo_update(ac, buffer, hp, optimizer)
                td_ema, td_ref = update_ema_pair(td_ema, td_ref, mean_abs_td, hp["td_ema_decay"], hp["td_ref_decay"])
                ent_ema, ent_ref = update_ema_pair(ent_ema, ent_ref, mean_entropy, hp["td_ema_decay"], hp["td_ref_decay"])
                kl_ema, kl_ref = update_ema_pair(kl_ema, kl_ref, mean_policy_shift, hp["td_ema_decay"], hp["td_ref_decay"])
                logger.log(total_steps, "loss", phase_idx, env_name, actor_loss, critic_loss, mean_abs_td, mean_entropy, mean_policy_shift)
                buffer.clear()

            trigger_sbp, trigger_mode, td_signal, kl_signal, ent_signal, combined_signal = should_trigger_sbp(
                total_steps, last_sbp_step, td_ema, td_ref, kl_ema, kl_ref, ent_ema, ent_ref, cfg, hp
            )
            if trigger_sbp:
                progress = total_steps / float(phase_steps * len(sequence))
                alpha_dynamic = get_dynamic_alpha(progress, hp["alpha_start"], hp["alpha_end"], hp["alpha_lambda"])
                alpha_base = alpha_dynamic if cfg.get("alpha_mode") == "dynamic" else hp["alpha_dkl"]
                if cfg.get("adaptive_alpha", False) and cfg.get("alpha_mode") == "dynamic":
                    alpha_base = get_adaptive_alpha(
                        alpha_dynamic,
                        td_ema if td_ema is not None else 1.0,
                        td_ref if td_ref is not None else 1.0,
                        hp,
                    )
                observed_kl = kl_ema if kl_ema is not None else hp["alpha_target_kl"]
                alpha_gap = 0.0
                if cfg.get("dual_alpha", False):
                    alpha_dual_state, alpha_gap = update_dual_alpha(alpha_dual_state, observed_kl, hp)
                    alpha_using = blend_alpha(alpha_base, alpha_dual_state, hp)
                else:
                    alpha_dual_state = float(alpha_base)
                    alpha_using = float(alpha_base)

                current_reset_rate = hp["reset_rate"]
                if cfg.get("adaptive_reset", False):
                    current_reset_rate, td_signal, kl_signal, ent_signal, combined_signal = compute_adaptive_reset_rate(
                        hp["reset_rate"],
                        td_ema if td_ema is not None else 1.0,
                        td_ref if td_ref is not None else 1.0,
                        kl_ema if kl_ema is not None else 1.0,
                        kl_ref if kl_ref is not None else 1.0,
                        ent_ema if ent_ema is not None else 1.0,
                        ent_ref if ent_ref is not None else 1.0,
                        hp,
                    )
                reset_states_np = buffer.sample_states_for_distill(hp["batch_size"]) 
                if reset_states_np is None:
                    reset_states_np = np.array([pad_state(env.observation_space.sample(), meta) for _ in range(hp["batch_size"])], dtype=np.float32)
                reset_state_batch = torch.FloatTensor(reset_states_np)

                ac_tem = UniversalActorCritic(state_dim_max, action_dim_max, hp)
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

                priorities, _, kl_values = compute_priority_scores(ac, ac_tem, buffer, hp) if cfg.get("hard_distill", False) else (None, None, None)
                mean_priority = float(np.mean(priorities)) if priorities is not None else 0.0
                mean_priority_kl = float(np.mean(kl_values)) if kl_values is not None else 0.0
                states_np = buffer.sample_states_for_distill(
                    hp["batch_size"],
                    priorities=priorities,
                    random_mix=hp["distill_mix_random"] if cfg.get("hard_distill", False) else 0.0,
                )
                if states_np is None:
                    states_np = reset_states_np
                state_batch = torch.FloatTensor(states_np)

                anchor_batch = None
                anchor_inserted = 0
                anchor_replaced = 0
                mean_anchor_priority = 0.0
                if cfg.get("anchor_memory", False):
                    anchor_candidates = build_anchor_candidates(ac_tem, buffer, hp, phase_idx, priorities=priorities)
                    if anchor_candidates is not None:
                        anchor_inserted, anchor_replaced = anchor_memory.add_batch(
                            anchor_candidates["states"],
                            anchor_candidates["ref_mean"],
                            anchor_candidates["ref_std"],
                            anchor_candidates["priorities"],
                            anchor_candidates["action_scales"],
                            anchor_candidates["action_masks"],
                            anchor_candidates["phase_idx"],
                        )
                    anchor_take = max(1, int(round(hp["batch_size"] * hp["anchor_batch_ratio"])))
                    anchor_batch = anchor_memory.sample(
                        anchor_take,
                        random_mix=hp["anchor_random_mix"],
                        priority_power=hp["anchor_priority_power"],
                    )
                    if anchor_batch is not None:
                        mean_anchor_priority = float(np.mean(anchor_batch["priorities"]))

                ac, distill_loss, distill_steps, current_distill_loss, anchor_distill_loss = inner_distill(
                    ac,
                    ac_tem,
                    state_batch,
                    action_scale.unsqueeze(0).expand(state_batch.shape[0], -1),
                    action_mask.unsqueeze(0).expand(state_batch.shape[0], -1),
                    alpha_using,
                    hp,
                    anchor_batch=anchor_batch,
                )
                logger.log(
                    total_steps,
                    "sbp",
                    phase_idx,
                    env_name,
                    trigger_mode,
                    round(alpha_using, 6),
                    round(alpha_dual_state, 6),
                    round(alpha_gap, 6),
                    round(current_reset_rate, 6),
                    round(combined_signal, 6),
                    round(mean_priority, 6),
                    round(mean_priority_kl, 6),
                    round(reset_stats["mean_reset_importance"], 6),
                    round(reset_stats["mean_preserve_importance"], 6),
                    round(reset_stats["mean_reset_count"], 3),
                    round(distill_loss, 6),
                    distill_steps,
                    round(current_distill_loss, 6),
                    round(anchor_distill_loss, 6),
                    len(anchor_memory),
                    anchor_inserted,
                    anchor_replaced,
                    round(mean_anchor_priority, 6),
                    int(cfg.get("anchor_memory", False)),
                )
                last_sbp_step = total_steps

            if total_steps % hp["eval_every"] == 0 or phase_step == phase_steps:
                for eval_env in dict.fromkeys(sequence):
                    eval_reward = evaluate_policy(ac, metas[eval_env], episodes=hp["eval_episodes"], seed=seed + phase_idx * 100)
                    best_eval[eval_env] = max(best_eval.get(eval_env, -float("inf")), eval_reward)
                    if phase_idx == 1 and eval_env == sequence[0]:
                        first_phase_best[eval_env] = max(first_phase_best.get(eval_env, -float("inf")), eval_reward)
                    ref_best = best_eval.get(eval_env, eval_reward)
                    forgetting = max(0.0, ref_best - eval_reward)
                    retention = eval_reward / max(ref_best, 1e-6)
                    logger.log(total_steps, "eval", phase_idx, env_name, eval_env, round(eval_reward, 3), round(retention, 6), round(forgetting, 3))

            if done:
                recent_rewards.append(episode_reward)
                avg10 = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                logger.log(total_steps, "reward", phase_idx, env_name, float(episode_reward), round(avg10, 3))
                episode_reward = 0.0
                state_raw, _ = env.reset()
                state = pad_state(state_raw, meta)

        current_a = evaluate_policy(ac, metas[sequence[0]], episodes=hp["eval_episodes"], seed=seed + 999 + phase_idx)
        reference_a = first_phase_best.get(sequence[0], current_a)
        forgetting_a = max(0.0, reference_a - current_a)
        retention_a = current_a / max(reference_a, 1e-6)
        logger.log(total_steps, "phase_summary", phase_idx, env_name, round(current_a, 3), round(reference_a, 3), round(retention_a, 6), round(forgetting_a, 3), round((time.time() - start_time) / 60, 2))

    for meta in metas.values():
        meta["env"].close()
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run A->B->A sequence plasticity experiment.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke run with fewer steps and eval episodes.")
    parser.add_argument(
        "--algo",
        choices=list(ALGO_SETTINGS.keys()),
        default="P3O-ClosedLoopMemory",
        help="Sequence algorithm to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-phase", type=int, help="Override steps for each phase.")
    parser.add_argument("--runs-dir", type=str, help="Override output directory for runs.")
    parser.add_argument("--sequence", nargs="+", help="Custom sequence, e.g. --sequence Hopper-v4 Walker2d-v4 Hopper-v4")
    parser.add_argument("--force", action="store_true", help="Force rerun even if metrics already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    global RUNS_DIR
    hp = dict(HP)
    if args.quick:
        hp["train_steps_per_phase"] = 20_000
        hp["buffer_size"] = 2048
        hp["eval_every"] = 5_000
        hp["eval_episodes"] = 2
        hp["distill_max_steps"] = 20
        hp["sbp_force_interval"] = 20_000
    if args.steps_per_phase is not None:
        hp["train_steps_per_phase"] = int(args.steps_per_phase)
    if args.runs_dir:
        RUNS_DIR = args.runs_dir
    sequence = args.sequence if args.sequence else DEFAULT_SEQUENCE
    cfg = dict(ALGO_SETTINGS[args.algo])
    run_dir = run_sequence(sequence, args.seed, hp, args.algo, cfg, quick=args.quick, force=args.force)
    print("Saved sequence run to:", run_dir)


if __name__ == "__main__":
    main()
