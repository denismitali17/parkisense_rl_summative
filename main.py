"""
main.py
=======
Entry point for running the best-performing trained agent in the
ParkiSense environment with full pygame visualization and terminal verbose.

This script:
1. Loads the best saved model (auto-detects from results CSVs, or use --algo/--run)
2. Runs N episodes with the pygame GUI open
3. Prints verbose terminal output per step
4. Summarises performance at the end

Usage
-----
    python main.py                          # auto-load best model, 5 episodes
    python main.py --algo dqn --run 3       # load DQN run 3
    python main.py --algo ppo --run 1       # load PPO run 1
    python main.py --algo reinforce --run 2 # load REINFORCE run 2
    python main.py --episodes 10            # run 10 episodes
    python main.py --no-render              # headless (terminal only)
"""

import sys
import os
import argparse
import csv
import time
import json
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from environment.custom_env import ParkiSenseEnv, ACTION_NAMES

MODELS_DIR = Path(__file__).resolve().parent / "models"
LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]


# Load helpers

def best_run_from_csv(csv_path: Path):
    """Return run index of highest mean_reward in a results CSV."""
    if not csv_path.exists():
        return 1
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return 1
    best = max(rows, key=lambda r: float(r["mean_reward"]))
    return int(best["run"])


def load_sb3_model(algo: str, run: int):
    from stable_baselines3 import DQN, PPO
    cls = DQN if algo == "dqn" else PPO
    path = MODELS_DIR / algo / f"{algo}_run{run}"
    model = cls.load(str(path))
    print(f"[Loaded] SB3 {algo.upper()} run {run} from {path}")
    return model, "sb3"


def load_reinforce_model(run: int):
    class PolicyNet(nn.Module):
        def __init__(self, obs_dim=12, act_dim=9, hidden_size=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, act_dim),
            )
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    # Try to find saved model; fall back to default hidden size
    path = MODELS_DIR / "pg" / f"reinforce_run{run}.pt"
    policy = PolicyNet()
    if path.exists():
        policy.load_state_dict(torch.load(str(path), map_location="cpu"))
        print(f"[Loaded] REINFORCE run {run} from {path}")
    else:
        print(f"[Warning] No saved weights at {path}. Using random policy for demo.")
    policy.eval()
    return policy, "reinforce"


def get_action(model, model_type: str, obs: np.ndarray) -> int:
    if model_type == "sb3":
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    else:  # reinforce pytorch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = model(obs_t)
        return probs.argmax().item()


# Episode runner

def run_episode(env, model, model_type, ep_num, verbose=True):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  Episode {ep_num} | True Severity: {info['true_severity']:.3f} "
              f"| Ground Truth: {LABEL_NAMES[info['true_label']]}")
        print(f"{'─'*65}")

    while not done:
        action = get_action(model, model_type, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if verbose:
            print(
                f"  Step {steps:2d} | {ACTION_NAMES[action]:<30} "
                f"| Reward: {reward:+6.2f} | Total: {total_reward:+7.2f}"
            )

    outcome = ""
    if info.get("last_action") is not None and info["last_action"] >= 6:
        predicted = info["last_action"] - 6
        true_lbl = info["true_label"]
        correct = "CORRECT" if predicted == true_lbl else "WRONG"
        outcome = f"Classified as {LABEL_NAMES[predicted]} [{correct}]"
    else:
        outcome = "Episode truncated (max steps)"

    if verbose:
        print(f"\n  Outcome: {outcome}")
        print(f"  Final Reward: {total_reward:+.2f} | Steps: {steps}")

    return total_reward, steps, outcome


# Main

def main():
    parser = argparse.ArgumentParser(description="ParkiSense — Run best agent")
    parser.add_argument("--algo", choices=["dqn", "ppo", "reinforce"], default=None,
                        help="Algorithm to load (default: auto-select best)")
    parser.add_argument("--run", type=int, default=None, help="Which run number to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window")
    args = parser.parse_args()

    # Auto-detect best algorithm if not specified
    if args.algo is None:
        best_per_algo = {}
        for algo in ("dqn", "ppo"):
            csv_path = MODELS_DIR / algo / f"{algo}_results.csv"
            run = best_run_from_csv(csv_path)
            best_per_algo[algo] = run

        # Try to find which algo had the best run
        # Fallback to ppo if no CSVs exist
        chosen_algo = "ppo"
        chosen_run = best_per_algo.get("ppo", 1)
        for algo, run in best_per_algo.items():
            csv_path = MODELS_DIR / algo / f"{algo}_results.csv"
            if csv_path.exists():
                chosen_algo = algo
                chosen_run = run
                break
        args.algo = chosen_algo
        args.run = chosen_run
        print(f"[Auto-select] Using {args.algo.upper()} run {args.run}")
    else:
        if args.run is None:
            csv_path = MODELS_DIR / args.algo / f"{args.algo}_results.csv"
            args.run = best_run_from_csv(csv_path)
            print(f"[Auto-select] Best {args.algo.upper()} run: {args.run}")

    # Load model
    try:
        if args.algo == "reinforce":
            model, model_type = load_reinforce_model(args.run)
        else:
            model, model_type = load_sb3_model(args.algo, args.run)
    except Exception as e:
        print(f"[Error] Could not load model: {e}")
        print("Make sure you have run the training scripts first.")
        sys.exit(1)

    render_mode = "human" if not args.no_render else None
    env = ParkiSenseEnv(render_mode=render_mode, max_steps=20)

    print("\n" + "="*65)
    print(f"  ParkiSense RL — {args.algo.upper()} Agent (Run {args.run})")
    print(f"  Environment: ParkiSense Parkinson's Screening")
    print(f"  Objective: Accurately classify patient risk with minimal tests")
    print("="*65)

    all_rewards = []
    all_steps = []

    for ep in range(1, args.episodes + 1):
        reward, steps, outcome = run_episode(env, model, model_type, ep)
        all_rewards.append(reward)
        all_steps.append(steps)

    env.close()

    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Algorithm : {args.algo.upper()} (run {args.run})")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean Steps : {np.mean(all_steps):.1f}")
    print(f"  Min Reward : {min(all_rewards):.2f}")
    print(f"  Max Reward : {max(all_rewards):.2f}")
    print("="*65)

    # Export summary to JSON for API/frontend integration
    summary = {
        "algorithm": args.algo,
        "run": args.run,
        "episodes": args.episodes,
        "mean_reward": round(float(np.mean(all_rewards)), 3),
        "std_reward": round(float(np.std(all_rewards)), 3),
        "mean_steps": round(float(np.mean(all_steps)), 1),
    }
    out_json = MODELS_DIR.parent / "last_run_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] Run summary to {out_json}")
    print("  (This JSON can be consumed by a web/mobile API frontend.)")


if __name__ == "__main__":
    main()
