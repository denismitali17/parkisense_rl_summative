# ParkiSense RL: Reinforcement Learning for Parkinson's Diagnosis Screening

##  Project Overview

**ParkiSense** is a machine learning-based application designed to screen for Parkinson's disease and estimate symptom severity from voice recordings. This repository implements a **reinforcement learning (RL) agent** that learns to optimize the diagnostic screening pipeline — deciding when to collect additional recordings, apply signal processing, extract features, and issue a classification — to maximize diagnostic accuracy while minimizing patient burden.

### The Problem
Early detection of Parkinson's disease (PD) is critical for intervention. However, diagnosing PD from voice biomarkers requires careful feature extraction and analysis. A naive screening system might over-test patients or miss cases. This project asks: **Can an RL agent learn to screen efficiently?**

### The Solution
Train three different RL algorithms (DQN, PPO, REINFORCE) to optimize the diagnostic pipeline. The agent observes voice feature measurements and decides which actions to take to reach a confident diagnosis with minimal steps and cost.


##  Assignment Objectives

This project demonstrates:

1. **Environment Validity & Complexity** — Custom Gymnasium environment with realistic state space, action space, and reward structure reflecting a medical screening use case
2. **Hyperparameter Experiments** — 10 runs each for DQN, PPO, and REINFORCE with comprehensive hyperparameter tuning and performance analysis
3. **System Implementation & Behavior** — Agents that learn meaningful diagnostic strategies and can be integrated into production pipelines
4. **Visualizations & Analysis** — Reward curves, convergence plots, entropy analysis, and generalization tests
5. **Video Demonstration** — Agent behavior recorded with full-screen GUI and verbose terminal output


## Project Structure

```
pablo_rl_summative/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # pygame 2D visualization dashboard
│   └── __init__.py              # Package marker
├── training/
│   ├── dqn_training.py          # DQN training (10 hyperparameter runs)
│   ├── pg_training.py           # PPO & REINFORCE training (10 runs each)
│   └── __init__.py              # Package marker
├── models/
│   ├── dqn/                     # Saved DQN models + results
│   │   ├── dqn_run1.zip through dqn_run10.zip
│   │   ├── dqn_results.csv
│   │   └── dqn_reward_curves.png
│   └── pg/
│       ├── ppo/
│       │   ├── ppo_run1.zip through ppo_run10.zip
│       │   ├── ppo_results.csv
│       │   └── ppo_reward_curves.png
│       └── reinforce/
│           ├── reinforce_run1.pt through reinforce_run10.pt
│           ├── reinforce_results.csv
│           └── reinforce_reward_curves.png
├── plots/                       # Report visualizations
│   ├── combined_reward_curves.png
│   ├── dqn_objective_curve.png
│   ├── pg_entropy_curves.png
│   ├── convergence_plot.png
│   ├── generalization_test.png
│   └── agent_architecture.png
├── main.py                      # Run best-performing agent
├── random_agent_demo.py         # Static random agent demo
├── requirements.txt             # Dependencies
└── README.md                    # This file
```


## Environment Design

### Observation Space (12 dimensions)
Voice biomarker measurements from a patient recording session:
- **Jitter** — variance in pitch (parkinsonian indicator)
- **Shimmer** — variance in amplitude
- **HNR** — harmonics-to-noise ratio
- **MDVP measures** — voice quality metrics
- **Tremor score** — motor tremor
- **Audio quality** — signal-to-noise ratio
- **Recording count** — number of recordings collected
- **Filter status** — noise filter applied?
- **Severity label** — ground truth (0=Low, 1=Medium, 2=High)

### Action Space (9 discrete actions)
1. **Request Recording** — collect another voice sample
2. **Apply Noise Filter** — preprocess audio
3. **Extract Jitter Features** — analyze pitch variance
4. **Extract Shimmer Features** — analyze amplitude variance
5. **Extract HNR Features** — analyze harmonics
6. **Extract MDVP Features** — analyze voice quality
7. **Tremor Analysis** — measure motor component
8. **Classify: Low Risk** — issue diagnosis
9. **Classify: Medium Risk** — issue diagnosis
10. **Classify: High Risk** — issue diagnosis

### Reward Structure
- **+25**: Correct high-risk classification
- **+20**: Correct medium/low risk classification
- **+5**: Evidence bonus (successful analysis)
- **+1**: Recording collection (encourages data gathering)
- **-30**: Missed a Parkinsonian patient (worst error)
- **-15**: False positive (unnecessary referral)
- **-1**: Per-step cost (encourages efficiency)

### Terminal Conditions
- Classification issued (episode ends)
- Maximum 20 steps reached
- ≥6 recordings collected (fatigue limit)


## Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `gymnasium` — RL environment framework
- `stable-baselines3` — DQN, PPO implementation
- `torch` — PyTorch backend
- `pygame` — 2D visualization
- `matplotlib` — plotting


## 1. Running the Project

### 1. Environment Visualization (No Model)
Test the environment with random actions:
```bash
python random_agent_demo.py                           # Run with pygame window
python random_agent_demo.py --episodes 10 --delay 0.5 # Slower visualization
python random_agent_demo.py --no-render               # Headless (terminal only)
```

### 2. Train DQN (Value-Based Method)
```bash
python training/dqn_training.py                       # All 10 runs
python training/dqn_training.py --timesteps 100000    # Custom budget
python training/dqn_training.py --run 7               # Single run
```

### 3. Train PPO & REINFORCE (Policy Gradient Methods)
```bash
python training/pg_training.py                        # All runs
python training/pg_training.py --algo ppo             # PPO only
python training/pg_training.py --algo reinforce       # REINFORCE only
```

### 4. Run Best-Performing Agent
```bash
python main.py                           # Auto-selects best model
python main.py --algo dqn --run 7        # Specific run
python main.py --episodes 10             # Multiple episodes
python main.py --no-render               # Headless mode
```

**Output**: `last_run_summary.json` (API-ready policy evaluation)


## Training Results Summary

### Overall Performance

| Algorithm | Best Run | Mean Reward | Std Dev | Training Time |
|-----------|----------|-------------|---------|---------------|
| **DQN** | Run 7 | **30.23** | ±2.998 | ~100s |
| **PPO** | Run 10 | **27.65** | ±2.174 | ~240s |
| **REINFORCE** | Run 10 | 4.45 | ±18.393 | ~220s |

### Key Findings

**DQN (Value-Based)**
- Consistently high performance across runs
- Best: batch size 256 for stability
- Sensitive to gamma (discount factor)
- Faster convergence than policy gradient methods

**PPO (Policy Gradient)**
- Most stable algorithm (lowest variance across runs)
- Best: entropy coefficient 0.01 (exploration bonus)
- Clipping constraint prevents destructive updates
- Strong performance with modest hyperparameter variations

**REINFORCE (Policy Gradient)**
- Highest variance due to Monte Carlo returns
- Converges to suboptimal policy (immediate classification)
- Demonstrates importance of variance reduction techniques
- Shows why modern PG algorithms (PPO) are preferred

### Agent Behavior (DQN Best Run)
The learned policy exhibits near-optimal behavior:
1. Request a recording on step 1
2. Apply noise filter
3. Run 2-3 feature analyses
4. Classify based on accumulated evidence
5. **Achieved 31.30 mean reward (90%+ of theoretical max)**


## Experimental Analysis

### Hyperparameter Sensitivity

**DQN**
- Learning rate: 5e-4 optimal (avoid > 5e-3)
- Batch size: Larger (256) > smaller (32)
- Gamma: 0.99 best; avoid 0.90 (too short horizon)

**PPO**
- Entropy coefficient: 0.01 critical (prevents mode collapse)
- Clip range: 0.2 balanced; tighter (0.1) more conservative
- n_epochs: 20 > 10, but with diminishing returns

**REINFORCE**
- Episodes per update: ≥5 reduces variance
- Entropy coef: 0.01 only marginal improvement
- Hidden layer size: 64 sufficient (128 no gain)


## Report Visualizations

All plots saved in `plots/`:

1. **combined_reward_curves.png** — Bar charts of all 30 runs with error bars
2. **dqn_objective_curve.png** — TD loss decay over training
3. **pg_entropy_curves.png** — Policy entropy for PPO and REINFORCE
4. **convergence_plot.png** — Mean reward trajectory for all algorithms
5. **generalization_test.png** — 100 unseen episodes performance by severity
6. **agent_architecture.png** — System diagram (policy network → environment → reward)
7. **dqn_reward_curves.png**, **ppo_reward_curves.png**, **reinforce_reward_curves.png** — Per-run curves


## Integration & API Readiness

### For Mobile/Web Frontend
Each run exports `last_run_summary.json`:

```json
{
  "algorithm": "dqn",
  "run": 7,
  "total_episodes": 5,
  "mean_reward": 31.30,
  "accuracy": 1.0,
  "episode_details": [
    {
      "episode": 1,
      "true_severity": 0.85,
      "predicted_label": 2,
      "correct": true,
      "reward": 31.50,
      "steps": 5
    }
  ]
}
```

This JSON can be:
- Serialized and served via REST API
- Consumed by a React/Flutter frontend
- Logged to a database for clinical review
- Used for A/B testing different policies

### Model Export (for Production)
```python
from stable_baselines3 import DQN
model = DQN.load('models/dqn/dqn_run7.zip')

# Use in microservice
observation, info = env.reset()
action, _states = model.predict(observation, deterministic=True)
```


##  Files & Their Purpose

| File | Purpose |
|------|---------|
| `environment/custom_env.py` | Gymnasium environment: state, action, reward logic |
| `environment/rendering.py` | pygame dashboard: patient state, actions, rewards |
| `training/dqn_training.py` | DQN hyperparameter sweep (10 runs) |
| `training/pg_training.py` | PPO + REINFORCE hyperparameter sweep (20 runs) |
| `main.py` | Load & run best model with GUI + logging |
| `random_agent_demo.py` | Random baseline for visualization validation |
| `models/dqn/dqn_results.csv` | DQN run metrics |
| `models/pg/ppo/ppo_results.csv` | PPO run metrics |
| `models/pg/reinforce/reinforce_results.csv` | REINFORCE run metrics |
| `plots/` | Report-ready visualizations |



## References & Notes

- **Gymnasium**: https://gymnasium.farama.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **DQN Paper**: Mnih et al. (2015) — "Human-level control through deep reinforcement learning"
- **PPO Paper**: Schulman et al. (2017) — "Proximal Policy Optimization Algorithms"
- **Voice Biomarkers**: Standard MDVP (Multi-Dimensional Voice Program) features

## Credits
**Author:** Denis Mitali
**Capstone Project:** ParkiSense RL  
**Course:** Machine Learning Techniques II
**School:** African Leadership University