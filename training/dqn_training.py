"""
dqn_training.py
================
Trains DQN agents on the ParkiSense environment using Stable Baselines 3.
Runs 10 hyperparameter experiments and saves results to CSV + plots.

Hyperparameters tuned per run:
  - learning_rate
  - buffer_size
  - batch_size
  - gamma
  - exploration_fraction
  - exploration_final_eps
  - target_update_interval
  - train_freq

Usage
-----
    python training/dqn_training.py
    python training/dqn_training.py --timesteps 100000
    python training/dqn_training.py --run 3          # run a single experiment index
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

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import ParkiSenseEnv

# -----------------------------------------------------------------------
# 10 Hyperparameter Configurations
# -----------------------------------------------------------------------
DQN_EXPERIMENTS = [
    # Run 1 — baseline
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.99,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 2 — lower LR, bigger buffer
    dict(learning_rate=5e-4, buffer_size=100000, batch_size=64, gamma=0.99,
         exploration_fraction=0.3, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 3 — high LR (risk of instability)
    dict(learning_rate=5e-3, buffer_size=50000, batch_size=64,  gamma=0.99,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 4 — low gamma (short-sighted)
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.90,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 5 — high gamma (long-sighted)
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.999,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 6 — small batch
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=32,  gamma=0.99,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 7 — large batch
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=256, gamma=0.99,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=500, train_freq=4),
    # Run 8 — high exploration
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.99,
         exploration_fraction=0.5, exploration_final_eps=0.10,
         target_update_interval=500, train_freq=4),
    # Run 9 — low exploration
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.99,
         exploration_fraction=0.1, exploration_final_eps=0.01,
         target_update_interval=500, train_freq=4),
    # Run 10 — frequent target updates + frequent training
    dict(learning_rate=1e-3, buffer_size=50000, batch_size=64,  gamma=0.99,
         exploration_fraction=0.2, exploration_final_eps=0.05,
         target_update_interval=100, train_freq=1),
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "models" / "dqn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RewardLogger(BaseCallback):
    """Logs mean episode reward every N steps for plotting."""

    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.rewards = []
        self._ep_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
        if self.n_calls % self.log_interval == 0 and self._ep_rewards:
            self.rewards.append(np.mean(self._ep_rewards[-20:]))
        return True


def train_dqn(run_idx: int, config: dict, total_timesteps: int):
    print(f"\n{'='*60}")
    print(f"  DQN Run {run_idx + 1}/10")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    env = make_vec_env(ParkiSenseEnv, n_envs=1)
    eval_env = ParkiSenseEnv()

    callback = RewardLogger(log_interval=1000)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        train_freq=config["train_freq"],
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)

    model_path = RESULTS_DIR / f"dqn_run{run_idx + 1}"
    model.save(str(model_path))

    env.close()
    eval_env.close()

    result = {
        "run": run_idx + 1,
        "learning_rate": config["learning_rate"],
        "buffer_size": config["buffer_size"],
        "batch_size": config["batch_size"],
        "gamma": config["gamma"],
        "exploration_fraction": config["exploration_fraction"],
        "exploration_final_eps": config["exploration_final_eps"],
        "target_update_interval": config["target_update_interval"],
        "train_freq": config["train_freq"],
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "training_time_s": round(elapsed, 1),
    }

    print(f"  => Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  ({elapsed:.1f}s)")
    return result, callback.rewards


def plot_reward_curves(all_rewards: list, run_labels: list):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=False)
    axes = axes.flatten()
    fig.suptitle("DQN Reward Curves — ParkiSense (10 Hyperparameter Runs)", fontsize=14)

    for i, (rewards, label) in enumerate(zip(all_rewards, run_labels)):
        ax = axes[i]
        if rewards:
            ax.plot(rewards, color="#4099FF", linewidth=1.5)
            ax.set_title(label, fontsize=8)
            ax.set_xlabel("Eval checkpoint (×1k steps)", fontsize=7)
            ax.set_ylabel("Mean reward", fontsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label, fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "dqn_reward_curves.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    print(f"\n[Plot saved] {out}")


def save_csv(results: list):
    out = RESULTS_DIR / "dqn_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    print(f"[CSV saved] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80000)
    parser.add_argument("--run", type=int, default=None, help="Run single experiment (1-10)")
    args = parser.parse_args()

    runs = list(enumerate(DQN_EXPERIMENTS))
    if args.run is not None:
        runs = [(args.run - 1, DQN_EXPERIMENTS[args.run - 1])]

    all_results = []
    all_rewards = []
    run_labels = []

    for idx, config in runs:
        result, rewards = train_dqn(idx, config, args.timesteps)
        all_results.append(result)
        all_rewards.append(rewards)
        run_labels.append(
            f"Run {idx+1}\nlr={config['learning_rate']}\nγ={config['gamma']}"
        )

    save_csv(all_results)

    if len(all_rewards) > 1:
        plot_reward_curves(all_rewards, run_labels)

    print("\nDQN training complete.")
    print(f"Best run: {max(all_results, key=lambda x: x['mean_reward'])}")


if __name__ == "__main__":
    main()
