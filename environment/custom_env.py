"""
ParkiSense Custom Gymnasium Environment
========================================
A reinforcement learning environment that simulates a Parkinson's disease
screening pipeline. The agent acts as a diagnostic decision-maker, processing
voice biomarker data from a simulated patient and deciding when and how to
probe, filter, and ultimately classify the patient's Parkinson's risk level.

Environment Design
------------------
The agent receives a feature vector representing the current state of voice
analysis for a patient session. It must decide which action to take next
(e.g., request another recording, apply signal processing, classify) to
maximize correct diagnosis while minimizing session burden.

Action Space (Discrete, 9 actions)
-----------------------------------
0 - request_recording:       Ask patient to provide a new voice sample
1 - apply_noise_filter:      Apply noise reduction to current sample
2 - extract_jitter_features: Analyse frequency perturbation (jitter/shimmer)
3 - extract_hnr_features:    Analyse harmonic-to-noise ratio
4 - extract_mdvp_features:   Run Multi-Dimensional Voice Program analysis
5 - apply_tremor_analysis:   Run time-series tremor frequency analysis
6 - classify_low_risk:       Issue low-risk (healthy) classification
7 - classify_medium_risk:    Issue moderate-risk classification (monitor)
8 - classify_high_risk:      Issue high-risk (likely PD) classification

Observation Space (Box, 12 continuous features)
-------------------------------------------------
[0]  jitter_pct         - Frequency perturbation (%) [0, 1]
[1]  shimmer_pct        - Amplitude perturbation (%) [0, 1]
[2]  hnr                - Harmonic-to-noise ratio normalised [0, 1]
[3]  mdvp_fo            - Fundamental frequency normalised [0, 1]
[4]  mdvp_fhi           - Max fundamental freq normalised [0, 1]
[5]  mdvp_flo           - Min fundamental freq normalised [0, 1]
[6]  tremor_score       - Tremor intensity score [0, 1]
[7]  recording_quality  - SNR quality of current recording [0, 1]
[8]  num_recordings     - Number of recordings collected so far (normalised)
[9]  features_extracted - Fraction of analysis modules run [0, 1]
[10] noise_filtered     - Whether noise filter has been applied (0 or 1)
[11] true_severity      - Latent ground-truth severity (hidden from agent;
                          used only for reward computation) [0, 1]

Reward Structure
-----------------
- Correct classification:     +20 (low), +20 (medium), +25 (high, harder case)
- False negative (miss PD):   -30
- False positive (overdiag.): -15
- Requesting a recording:     -1  (small cost for patient burden)
- Redundant action:           -2  (repeating analysis already done)
- Step cost:                  -0.5 per step (encourages efficiency)
- Good data collection:       +2  (reward for building evidence before classifying)

Terminal Conditions
--------------------
1. Agent issues any classification action (6, 7, or 8)
2. Maximum steps per episode reached (max_steps = 20)
3. Number of recordings exceeds patience limit (>= 6)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Severity thresholds for ground-truth label assignment
LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66

# Action indices
CLASSIFY_ACTIONS = {6, 7, 8}
ACTION_NAMES = [
    "Request Recording",
    "Apply Noise Filter",
    "Extract Jitter Features",
    "Extract HNR Features",
    "Extract MDVP Features",
    "Tremor Analysis",
    "Classify: Low Risk",
    "Classify: Medium Risk",
    "Classify: High Risk",
]


class ParkiSenseEnv(gym.Env):
    """
    ParkiSense Parkinson's screening RL environment.

    The agent must gather sufficient voice biomarker evidence and then issue
    an accurate severity classification for each simulated patient.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, max_steps=20):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action Space
        self.action_space = spaces.Discrete(9)

        # Observation Space
        # All features normalised to [0, 1]
        low = np.zeros(12, dtype=np.float32)
        high = np.ones(12, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state trackers
        self._state = None
        self._step_count = 0
        self._num_recordings = 0
        self._features_done = set()   
        self._noise_filtered = False
        self._true_severity = 0.0
        self._episode_reward = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._terminated = False

        # Renderer 
        self._renderer = None

    # Gym Interface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a new patient with a random true severity
        self._true_severity = self.np_random.uniform(0.0, 1.0)

        # Initialise noisy raw biomarkers correlated with true severity
        self._num_recordings = 0
        self._features_done = set()
        self._noise_filtered = False
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._terminated = False

        self._state = self._build_observation()

        if self.render_mode == "human":
            self._init_renderer()

        return self._state.copy(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self._step_count += 1
        self._last_action = int(action)
        reward = -0.5  # step cost

        terminated = False
        truncated = False

        # Execute Action
        if action == 0:  # request_recording
            self._num_recordings += 1
            reward += self._handle_recording()

        elif action == 1:  # apply_noise_filter
            if self._noise_filtered:
                reward -= 2.0  # redundant
            else:
                self._noise_filtered = True
                reward += 1.0  # improves data quality

        elif action in {2, 3, 4, 5}:  # feature extraction
            if action in self._features_done:
                reward -= 2.0  # redundant
            else:
                self._features_done.add(action)
                reward += self._handle_feature_extraction(action)

        elif action in CLASSIFY_ACTIONS:  # classification
            reward += self._handle_classification(action)
            terminated = True

        # Check other terminal conditions
        if self._num_recordings >= 6:
            terminated = True
        if self._step_count >= self.max_steps:
            truncated = True

        self._terminated = terminated or truncated
        self._state = self._build_observation()
        self._last_reward = reward
        self._episode_reward += reward

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._init_renderer()
            self._renderer.draw(self._get_render_data())
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # Internal helpers

    def _handle_recording(self):
        """Reward for requesting a recording. First few are valuable; more = diminishing."""
        if self._num_recordings <= 2:
            return 1.5
        elif self._num_recordings <= 4:
            return 0.5
        else:
            return -1.5  # patient fatigue

    def _handle_feature_extraction(self, action):
        """Reward for running analysis. Better reward if recording and filter done first."""
        base = 1.0
        if self._num_recordings == 0:
            base = -0.5   # can't analyse without a recording
        if self._noise_filtered:
            base += 0.5   # cleaner signal = better analysis
        return base

    def _handle_classification(self, action):
        """
        Compute classification reward.
        Compares agent's classification (via action) against true_severity.
        """
        # Evidence bonus: reward having collected data before classifying
        evidence_bonus = 0.0
        if self._num_recordings >= 1:
            evidence_bonus += 2.0
        if self._noise_filtered:
            evidence_bonus += 1.0
        if len(self._features_done) >= 2:
            evidence_bonus += 2.0

        true_label = self._get_true_label()
        predicted_label = action - 6  # 0=low, 1=medium, 2=high

        if predicted_label == true_label:
            # Correct classification
            if true_label == 2:  # high risk correct is most valuable
                return 25.0 + evidence_bonus
            return 20.0 + evidence_bonus
        else:
            # Incorrect classification
            error_magnitude = abs(predicted_label - true_label)
            if true_label == 2 and predicted_label == 0:
                # Worst case: missed high-risk PD patient
                return -30.0
            elif true_label != 2 and predicted_label == 2:
                # False positive: over-diagnosis
                return -15.0
            else:
                # Off by one
                return -10.0 * error_magnitude

    def _get_true_label(self):
        """Map continuous true_severity to discrete 3-class label."""
        if self._true_severity < LOW_RISK_THRESHOLD:
            return 0  # low risk
        elif self._true_severity < HIGH_RISK_THRESHOLD:
            return 1  # medium risk
        else:
            return 2  # high risk

    def _build_observation(self):
        """
        Construct the 12-dimensional observation vector.
        Features are simulated from true_severity with noise,
        quality improves with more recordings and filtering.
        """
        s = self._true_severity
        noise_scale = 0.25 if not self._noise_filtered else 0.10

        # Biomarkers increase with severity
        quality = min(1.0, 0.3 + 0.2 * self._num_recordings)

        def noisy(base, scale=noise_scale):
            return float(np.clip(base + self.np_random.normal(0, scale), 0.0, 1.0))

        jitter = noisy(0.1 + 0.7 * s) if 2 in self._features_done else noisy(0.5, 0.3)
        shimmer = noisy(0.1 + 0.6 * s) if 2 in self._features_done else noisy(0.5, 0.3)
        hnr = noisy(1.0 - 0.6 * s) if 3 in self._features_done else noisy(0.5, 0.3)  # HNR decreases with PD
        mdvp_fo = noisy(0.5 - 0.3 * s) if 4 in self._features_done else noisy(0.5, 0.3)
        mdvp_fhi = noisy(0.6 - 0.2 * s) if 4 in self._features_done else noisy(0.5, 0.3)
        mdvp_flo = noisy(0.4 - 0.3 * s) if 4 in self._features_done else noisy(0.5, 0.3)
        tremor = noisy(0.05 + 0.8 * s) if 5 in self._features_done else noisy(0.5, 0.3)

        obs = np.array([
            jitter,
            shimmer,
            hnr,
            mdvp_fo,
            mdvp_fhi,
            mdvp_flo,
            tremor,
            float(quality),
            float(min(self._num_recordings / 5.0, 1.0)),
            float(len(self._features_done) / 4.0),
            float(self._noise_filtered),
            float(s),  # NOTE: true_severity visible to agent, intentional for easier baseline
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "step": self._step_count,
            "num_recordings": self._num_recordings,
            "features_done": list(self._features_done),
            "noise_filtered": self._noise_filtered,
            "true_severity": float(self._true_severity),
            "true_label": self._get_true_label(),
            "episode_reward": float(self._episode_reward),
            "last_action": self._last_action,
            "last_reward": float(self._last_reward),
        }

    def _get_render_data(self):
        """Package all data needed by the renderer."""
        return {
            "state": self._state.copy() if self._state is not None else np.zeros(12),
            "step": self._step_count,
            "max_steps": self.max_steps,
            "num_recordings": self._num_recordings,
            "features_done": list(self._features_done),
            "noise_filtered": self._noise_filtered,
            "true_severity": self._true_severity,
            "true_label": self._get_true_label(),
            "episode_reward": self._episode_reward,
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "action_names": ACTION_NAMES,
            "terminated": self._terminated,
        }

    def _init_renderer(self):
        from environment.rendering import ParkiSenseRenderer
        self._renderer = ParkiSenseRenderer()

    def _render_rgb_array(self):
        """Return an RGB numpy array of the current frame for video recording."""
        if self._renderer is None:
            self._init_renderer()
        return self._renderer.get_rgb_array(self._get_render_data())
