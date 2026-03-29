"""
pg_training.py
==============
Trains REINFORCE and PPO agents on the ParkiSense environment
using Stable Baselines 3 (and a custom REINFORCE implementation).

Runs 10 hyperparameter experiments per algorithm and saves results.

Note on REINFORCE
-----------------
SB3 does not ship a standalone REINFORCE implementation. We use a minimal
custom REINFORCE with entropy regularisation here so the assignment can
compare it directly with the SB3-based PPO.

Usage
-----
    python training/pg_training.py                   # all runs, both algorithms
    python training/pg_training.py --algo ppo        # PPO only
    python training/pg_training.py --algo reinforce  # REINFORCE only
    python training/pg_training.py --timesteps 80000
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import ParkiSenseEnv

RESULTS_DIR = Path(__file__).resolve().parents[1] / "models" / "pg"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# PPO — 10 Hyperparameter Configurations
# -----------------------------------------------------------------------
PPO_EXPERIMENTS = [
    # Run 1 — baseline
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=10),
    # Run 2 — lower LR
    dict(learning_rate=1e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=10),
    # Run 3 — high LR
    dict(learning_rate=1e-3, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=10),
    # Run 4 — tight clipping (conservative updates)
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, n_epochs=10),
    # Run 5 — wide clipping (aggressive updates)
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.3, ent_coef=0.0, n_epochs=10),
    # Run 6 — entropy bonus (encourages exploration)
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_epochs=10),
    # Run 7 — high entropy bonus
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.05, n_epochs=10),
    # Run 8 — lower gamma
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.95,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=10),
    # Run 9 — more SGD epochs per rollout
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=20),
    # Run 10 — small rollout buffer + small batch
    dict(learning_rate=3e-4, n_steps=512,  batch_size=32,  gamma=0.99,
         gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, n_epochs=10),
]


# -----------------------------------------------------------------------
# REINFORCE — 10 Hyperparameter Configurations
# -----------------------------------------------------------------------
REINFORCE_EXPERIMENTS = [
    # Run 1 — baseline
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.0, hidden_size=64,  episodes_per_update=5),
    # Run 2 — lower LR
    dict(learning_rate=5e-4, gamma=0.99, entropy_coef=0.0, hidden_size=64,  episodes_per_update=5),
    # Run 3 — high LR
    dict(learning_rate=5e-3, gamma=0.99, entropy_coef=0.0, hidden_size=64,  episodes_per_update=5),
    # Run 4 — lower gamma
    dict(learning_rate=1e-3, gamma=0.90, entropy_coef=0.0, hidden_size=64,  episodes_per_update=5),
    # Run 5 — higher gamma
    dict(learning_rate=1e-3, gamma=0.999,entropy_coef=0.0, hidden_size=64,  episodes_per_update=5),
    # Run 6 — entropy regularisation
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.01,hidden_size=64,  episodes_per_update=5),
    # Run 7 — high entropy
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.05,hidden_size=64,  episodes_per_update=5),
    # Run 8 — wider network
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.0, hidden_size=128, episodes_per_update=5),
    # Run 9 — more episodes per gradient update (lower variance)
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.0, hidden_size=64,  episodes_per_update=10),
    # Run 10 — fewer episodes per update (higher variance)
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.0, hidden_size=64,  episodes_per_update=1),
]


# -----------------------------------------------------------------------
# Minimal REINFORCE Policy Network
# -----------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def reinforce_train(run_idx: int, config: dict, total_episodes: int):
    print(f"\n{'='*60}")
    print(f"  REINFORCE Run {run_idx + 1}/10")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    env = ParkiSenseEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, config["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    gamma = config["gamma"]
    entropy_coef = config["entropy_coef"]
    eps_per_update = config["episodes_per_update"]

    reward_history = []
    episode_count = 0
    t0 = time.time()

    while episode_count < total_episodes:
        # Collect a batch of episodes
        batch_log_probs = []
        batch_returns = []
        batch_entropies = []
        batch_ep_rewards = []

        for _ in range(eps_per_update):
            obs, _ = env.reset()
            ep_rewards = []
            ep_log_probs = []
            ep_entropies = []
            done = False

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                probs = policy(obs_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                ep_log_probs.append(dist.log_prob(action))
                ep_entropies.append(dist.entropy())

                obs, reward, terminated, truncated, _ = env.step(action.item())
                ep_rewards.append(reward)
                done = terminated or truncated

            # Compute discounted returns
            G = 0.0
            returns = []
            for r in reversed(ep_rewards):
                G = r + gamma * G
                returns.insert(0, G)

            batch_log_probs.extend(ep_log_probs)
            batch_returns.extend(returns)
            batch_entropies.extend(ep_entropies)
            batch_ep_rewards.append(sum(ep_rewards))
            episode_count += 1

        # Normalise returns
        returns_t = torch.FloatTensor(batch_returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(batch_log_probs)
        entropies_t = torch.stack(batch_entropies)

        policy_loss = -(log_probs_t * returns_t).mean()
        entropy_loss = -entropy_coef * entropies_t.mean()
        loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        mean_ep = np.mean(batch_ep_rewards)
        reward_history.append(mean_ep)

        if episode_count % 50 == 0:
            print(f"  Episode {episode_count}/{total_episodes} | Mean reward: {mean_ep:.2f}")

    elapsed = time.time() - t0

    # Evaluate
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = policy(obs_t)
            action = probs.argmax().item()
            obs, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            done = terminated or truncated
        eval_rewards.append(ep_r)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    # Save model
    model_path = RESULTS_DIR / f"reinforce_run{run_idx + 1}.pt"
    torch.save(policy.state_dict(), str(model_path))
    env.close()

    print(f"  => Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  ({elapsed:.1f}s)")

    return {
        "run": run_idx + 1,
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "entropy_coef": config["entropy_coef"],
        "hidden_size": config["hidden_size"],
        "episodes_per_update": config["episodes_per_update"],
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "training_time_s": round(elapsed, 1),
    }, reward_history


# -----------------------------------------------------------------------
# PPO Training
# -----------------------------------------------------------------------

def train_ppo(run_idx: int, config: dict, total_timesteps: int):
    print(f"\n{'='*60}")
    print(f"  PPO Run {run_idx + 1}/10")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    env = make_vec_env(ParkiSenseEnv, n_envs=1)
    eval_env = ParkiSenseEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        n_epochs=config["n_epochs"],
    )

    reward_history = []

    class PPORewardCallback:
        """Lightweight reward tracking via rollout buffer."""
        pass  # We'll log via evaluate_policy checkpoints below

    t0 = time.time()
    checkpoints = 10
    ckpt_steps = total_timesteps // checkpoints

    for ck in range(checkpoints):
        model.learn(
            total_timesteps=ckpt_steps,
            reset_num_timesteps=(ck == 0),
            progress_bar=False,
        )
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        reward_history.append(mean_r)
        print(f"  Checkpoint {ck+1}/{checkpoints} | Eval mean reward: {mean_r:.2f}")

    elapsed = time.time() - t0
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)

    model_path = RESULTS_DIR / f"ppo_run{run_idx + 1}"
    model.save(str(model_path))
    env.close()
    eval_env.close()

    print(f"  => Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  ({elapsed:.1f}s)")

    return {
        "run": run_idx + 1,
        "learning_rate": config["learning_rate"],
        "n_steps": config["n_steps"],
        "batch_size": config["batch_size"],
        "gamma": config["gamma"],
        "gae_lambda": config["gae_lambda"],
        "clip_range": config["clip_range"],
        "ent_coef": config["ent_coef"],
        "n_epochs": config["n_epochs"],
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "training_time_s": round(elapsed, 1),
    }, reward_history


# -----------------------------------------------------------------------
# Plotting & CSV
# -----------------------------------------------------------------------

def plot_reward_curves(all_rewards: list, run_labels: list, algo: str):
    n = len(all_rewards)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 7))
    axes = axes.flatten()
    fig.suptitle(f"{algo.upper()} Reward Curves — ParkiSense (10 Runs)", fontsize=14)
    color = "#4EC9B0" if algo == "reinforce" else "#C586C0"

    for i in range(min(n, rows * cols)):
        ax = axes[i]
        if all_rewards[i]:
            ax.plot(all_rewards[i], color=color, linewidth=1.5)
            ax.set_title(run_labels[i], fontsize=8)
            ax.set_xlabel("Update step", fontsize=7)
            ax.set_ylabel("Mean reward", fontsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    out = RESULTS_DIR / f"{algo}_reward_curves.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    print(f"\n[Plot saved] {out}")


def save_csv(results: list, algo: str):
    out = RESULTS_DIR / f"{algo}_results.csv"
    if results:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"[CSV saved] {out}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80000)
    parser.add_argument("--algo", choices=["reinforce", "ppo", "all"], default="all")
    args = parser.parse_args()

    if args.algo in ("reinforce", "all"):
        print("\n" + "="*60)
        print("  RUNNING REINFORCE EXPERIMENTS")
        print("="*60)
        reinforce_results = []
        reinforce_rewards = []
        reinforce_labels = []
        episodes = max(200, args.timesteps // 100)  # convert timesteps → episodes approx

        for i, config in enumerate(REINFORCE_EXPERIMENTS):
            result, rewards = reinforce_train(i, config, total_episodes=episodes)
            reinforce_results.append(result)
            reinforce_rewards.append(rewards)
            reinforce_labels.append(
                f"Run {i+1}\nlr={config['learning_rate']}\nγ={config['gamma']}"
            )

        save_csv(reinforce_results, "reinforce")
        plot_reward_curves(reinforce_rewards, reinforce_labels, "reinforce")
        print(f"\nBest REINFORCE run: {max(reinforce_results, key=lambda x: x['mean_reward'])}")

    if args.algo in ("ppo", "all"):
        print("\n" + "="*60)
        print("  RUNNING PPO EXPERIMENTS")
        print("="*60)
        ppo_results = []
        ppo_rewards = []
        ppo_labels = []

        for i, config in enumerate(PPO_EXPERIMENTS):
            result, rewards = train_ppo(i, config, args.timesteps)
            ppo_results.append(result)
            ppo_rewards.append(rewards)
            ppo_labels.append(
                f"Run {i+1}\nlr={config['learning_rate']}\nclip={config['clip_range']}"
            )

        save_csv(ppo_results, "ppo")
        plot_reward_curves(ppo_rewards, ppo_labels, "ppo")
        print(f"\nBest PPO run: {max(ppo_results, key=lambda x: x['mean_reward'])}")

    print("\nAll policy gradient training complete.")


if __name__ == "__main__":
    main()
