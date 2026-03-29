"""
random_agent_demo.py
=====================
Demonstrates the ParkiSense environment visualization using a RANDOM agent.
No model or training is involved — the agent simply samples uniformly from
the action space at each step. This script satisfies the assignment requirement:

    "Create a static file that shows the agent taking random actions
     (not using a model) in the custom environment."

Usage
-----
    python random_agent_demo.py                  # run with pygame window
    python random_agent_demo.py --no-render      # headless (logs only)
    python random_agent_demo.py --episodes 5     # run N episodes

Press Ctrl+C or close the window to stop early.
"""

import argparse
import time
import sys
import os

# Ensure project root is on path when called from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import ParkiSenseEnv, ACTION_NAMES


def run_random_demo(num_episodes: int = 3, render: bool = True, delay: float = 0.3):
    render_mode = "human" if render else None
    env = ParkiSenseEnv(render_mode=render_mode, max_steps=20)

    print("=" * 60)
    print("  ParkiSense — Random Agent Demo")
    print("  (No model — uniform random action selection)")
    print("=" * 60)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n--- Episode {ep + 1} | True Severity: {info['true_severity']:.3f} "
              f"(Label: {['Low','Medium','High'][info['true_label']]}) ---")

        while not done:
            action = env.action_space.sample()  # RANDOM — no model
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            print(f"  Step {step:2d} | Action: {action} ({ACTION_NAMES[action]:<30}) "
                  f"| Reward: {reward:+6.2f} | Cumulative: {total_reward:+7.2f}")

            if render:
                time.sleep(delay)

        outcome = "TERMINATED (classified/exhausted)" if terminated else "TRUNCATED (max steps)"
        print(f"  => {outcome} | Total Reward: {total_reward:+.2f}")

    env.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParkiSense random agent demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between steps")
    args = parser.parse_args()

    run_random_demo(
        num_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay,
    )
