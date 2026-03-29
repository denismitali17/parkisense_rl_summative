# ParkiSense RL

**Reinforcement learning agent for Parkinson's disease screening simulation.**

This project implements a custom Gymnasium environment that simulates the ParkiSense
diagnostic pipeline, and trains Value-Based (DQN) and Policy Gradient (REINFORCE, PPO)
agents to learn optimal screening strategies.

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   ├── rendering.py         # pygame 2D visualization dashboard
│   └── __init__.py
├── training/
│   ├── dqn_training.py      # DQN — 10 hyperparameter runs
│   └── pg_training.py       # REINFORCE + PPO — 10 runs each
├── models/
│   ├── dqn/                 # Saved DQN models + results CSV + plots
│   └── pg/                  # Saved REINFORCE/PPO models + results + plots
├── main.py                  # Run best-performing agent (with GUI)
├── random_agent_demo.py     # Random agent demo (no model, visualization only)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running

### 1. Random agent demo (environment visualization, no model)
```bash
python random_agent_demo.py
python random_agent_demo.py --episodes 5 --delay 0.2
python random_agent_demo.py --no-render   # headless
```

### 2. Train DQN (10 hyperparameter runs)
```bash
python training/dqn_training.py
python training/dqn_training.py --timesteps 100000
python training/dqn_training.py --run 3    # single run
```

### 3. Train REINFORCE + PPO (10 runs each)
```bash
python training/pg_training.py
python training/pg_training.py --algo ppo
python training/pg_training.py --algo reinforce
```

### 4. Run best agent with GUI
```bash
python main.py                        # auto-selects best
python main.py --algo dqn --run 1
python main.py --algo ppo --run 3
python main.py --episodes 10
python main.py --no-render            # headless mode
```

---

## Environment

| Component | Description |
|---|---|
| **Observation** | 12-dim vector: jitter, shimmer, HNR, MDVP features, tremor, quality, recordings count, filter status, severity |
| **Actions** | 9 discrete: request recording, noise filter, 4 analysis modules, 3 classification levels |
| **Reward** | +20–25 correct classification, –30 missed high-risk, –15 false positive, –1/step |
| **Terminal** | Classification issued, max 20 steps, or ≥6 recordings collected |

---

## Results

After training, results CSVs and reward curve plots are saved in `models/dqn/` and `models/pg/`.
A `last_run_summary.json` is exported after each `main.py` run — this JSON is designed
to be consumed by a REST API or mobile frontend.

---

## Integration with ParkiSense App

The trained policy can be serialised and served as a microservice:

```python
# Pseudo-code for API integration
from stable_baselines3 import PPO
model = PPO.load("models/pg/ppo_run_best")

@app.route("/screen", methods=["POST"])
def screen():
    features = request.json["voice_features"]  # from mobile app
    action, _ = model.predict(features, deterministic=True)
    return jsonify({"action": int(action), "label": ACTION_NAMES[action]})
```
