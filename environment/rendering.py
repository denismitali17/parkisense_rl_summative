"""
ParkiSense Environment Renderer
================================
pygame-based 2D visualization dashboard for the ParkiSense RL environment.

Layout
------
+---------------------------------------------+
|  PARKISENSE  — Parkinson's Screening Agent  |
+------------------+--------------------------+
|  PATIENT PANEL   |   BIOMARKER READINGS     |
|  severity bar    |   bar charts per feature |
|  recording count |                          |
+------------------+--------------------------+
|  AGENT PANEL     |   ACTION LOG             |
|  current action  |   last 8 actions         |
|  reward display  |                          |
+------------------+--------------------------+
|  PROGRESS BAR (steps)  |  CUMULATIVE REWARD |
+---------------------------------------------+
"""

import numpy as np

# Defer pygame import so the env can be imported headlessly
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ---------- Colour palette ----------
BG_DARK      = (15,  20,  35)
BG_PANEL     = (25,  33,  55)
BG_PANEL2    = (20,  28,  48)
ACCENT_BLUE  = (64, 156, 255)
ACCENT_GREEN = (52, 211, 153)
ACCENT_RED   = (248,  87,  87)
ACCENT_ORANGE= (251, 171,  52)
ACCENT_PURPLE= (167, 139, 250)
TEXT_PRIMARY = (220, 230, 255)
TEXT_MUTED   = (120, 135, 170)
BAR_BG       = (40,  52,  80)
SEVERITY_LOW = (52, 211, 153)
SEVERITY_MED = (251, 171,  52)
SEVERITY_HI  = (248,  87,  87)

FEATURE_NAMES = ["Jitter", "Shimmer", "HNR", "MDVP-Fo", "MDVP-Fhi",
                 "MDVP-Flo", "Tremor", "Quality", "Recordings",
                 "Features", "Filtered", "Severity"]

FEATURE_COLORS = [
    ACCENT_BLUE, ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
    ACCENT_PURPLE, ACCENT_PURPLE, ACCENT_RED, ACCENT_ORANGE,
    ACCENT_GREEN, ACCENT_GREEN, ACCENT_BLUE, ACCENT_RED,
]

ACTION_COLORS = {
    0: ACCENT_BLUE,
    1: ACCENT_GREEN,
    2: ACCENT_PURPLE,
    3: ACCENT_PURPLE,
    4: ACCENT_PURPLE,
    5: ACCENT_RED,
    6: SEVERITY_LOW,
    7: SEVERITY_MED,
    8: SEVERITY_HI,
}

W, H = 900, 620


class ParkiSenseRenderer:
    """Manages the pygame window and drawing logic."""

    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("ParkiSense — Parkinson's Screening RL Agent")
        self.clock = pygame.time.Clock()
        self._load_fonts()
        self._action_log = []

    def _load_fonts(self):
        pygame.font.init()
        self.font_title  = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.font_header = pygame.font.SysFont("dejavusans", 14, bold=True)
        self.font_body   = pygame.font.SysFont("dejavusans", 12)
        self.font_small  = pygame.font.SysFont("dejavusans", 10)
        self.font_big    = pygame.font.SysFont("dejavusans", 28, bold=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, data: dict):
        """Render one frame from environment data dict."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if data.get("last_action") is not None:
            name = data["action_names"][data["last_action"]]
            r = data["last_reward"]
            self._action_log.append((name, r, data["last_action"]))
            if len(self._action_log) > 8:
                self._action_log.pop(0)

        self.screen.fill(BG_DARK)
        self._draw_title(data)
        self._draw_patient_panel(data)
        self._draw_biomarker_panel(data)
        self._draw_agent_panel(data)
        self._draw_action_log(data)
        self._draw_progress_bar(data)

        pygame.display.flip()
        self.clock.tick(5)

    def get_rgb_array(self, data: dict) -> np.ndarray:
        """Return current frame as H×W×3 uint8 numpy array."""
        self.draw(data)
        return np.transpose(
            np.array(pygame.surfarray.array3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_title(self, data):
        # Title bar
        pygame.draw.rect(self.screen, BG_PANEL, (0, 0, W, 48))
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, 48), (W, 48), 2)

        title = self.font_title.render("ParkiSense  —  Parkinson's Screening RL Agent", True, TEXT_PRIMARY)
        self.screen.blit(title, (16, 14))

        # Episode / step counter top-right
        step_txt = self.font_body.render(
            f"Step {data['step']} / {data['max_steps']}", True, TEXT_MUTED
        )
        self.screen.blit(step_txt, (W - 150, 17))

    def _draw_patient_panel(self, data):
        """Left column — patient info and severity."""
        px, py, pw, ph = 10, 58, 280, 240
        self._panel(px, py, pw, ph, "PATIENT STATUS")

        # True severity bar (gradient colour)
        sev = data["true_severity"]
        bar_col = self._severity_color(sev)
        self._labelled_bar(px + 12, py + 38, 256, 22, sev, "True Severity", bar_col)

        # True label
        label_names = ["Low Risk", "Medium Risk", "High Risk"]
        label_cols  = [SEVERITY_LOW, SEVERITY_MED, SEVERITY_HI]
        lbl = data["true_label"]
        label_surf = self.font_header.render(
            f"Ground Truth: {label_names[lbl]}", True, label_cols[lbl]
        )
        self.screen.blit(label_surf, (px + 12, py + 72))

        # Recordings
        rec = data["num_recordings"]
        rec_surf = self.font_body.render(
            f"Recordings collected: {rec}", True, TEXT_PRIMARY
        )
        self.screen.blit(rec_surf, (px + 12, py + 100))

        # Noise filter
        filt_col = ACCENT_GREEN if data["noise_filtered"] else ACCENT_RED
        filt_txt = "Noise Filter: ON" if data["noise_filtered"] else "Noise Filter: OFF"
        self.screen.blit(self.font_body.render(filt_txt, True, filt_col), (px + 12, py + 122))

        # Features extracted
        done = data["features_done"]
        feat_map = {2: "Jitter", 3: "HNR", 4: "MDVP", 5: "Tremor"}
        feats_txt = "Analyses done: " + (", ".join(feat_map[f] for f in sorted(done)) or "None")
        self.screen.blit(
            self.font_body.render(feats_txt, True, TEXT_PRIMARY), (px + 12, py + 144)
        )

        # Cumulative reward large display
        reward_col = ACCENT_GREEN if data["episode_reward"] >= 0 else ACCENT_RED
        r_surf = self.font_big.render(f"{data['episode_reward']:+.1f}", True, reward_col)
        self.screen.blit(r_surf, (px + 12, py + 175))
        self.screen.blit(
            self.font_small.render("Cumulative Reward", True, TEXT_MUTED), (px + 12, py + 215)
        )

    def _draw_biomarker_panel(self, data):
        """Right column — biomarker bar chart."""
        px, py, pw, ph = 300, 58, 590, 240
        self._panel(px, py, pw, ph, "BIOMARKER READINGS")

        obs = data["state"]
        bar_w = 34
        gap = 12
        total = len(FEATURE_NAMES)
        start_x = px + 16

        for i, (name, val, col) in enumerate(zip(FEATURE_NAMES, obs, FEATURE_COLORS)):
            bx = start_x + i * (bar_w + gap)
            by = py + 40
            max_h = ph - 80

            # Background
            pygame.draw.rect(self.screen, BAR_BG, (bx, by, bar_w, max_h), border_radius=4)
            # Fill
            fill_h = int(val * max_h)
            pygame.draw.rect(
                self.screen, col,
                (bx, by + max_h - fill_h, bar_w, fill_h),
                border_radius=4,
            )
            # Value label
            val_surf = self.font_small.render(f"{val:.2f}", True, TEXT_PRIMARY)
            self.screen.blit(val_surf, (bx + bar_w // 2 - val_surf.get_width() // 2, by + max_h - fill_h - 14))
            # Name label (rotated)
            name_surf = self.font_small.render(name, True, TEXT_MUTED)
            name_rot  = pygame.transform.rotate(name_surf, 45)
            self.screen.blit(name_rot, (bx, by + max_h + 4))

    def _draw_agent_panel(self, data):
        """Bottom-left — current action info."""
        px, py, pw, ph = 10, 308, 280, 250
        self._panel(px, py, pw, ph, "AGENT DECISION")

        act = data["last_action"]
        if act is not None:
            act_name = data["action_names"][act]
            col = ACTION_COLORS.get(act, ACCENT_BLUE)
            a_surf = self.font_header.render(f"Action #{act}", True, col)
            self.screen.blit(a_surf, (px + 12, py + 38))
            n_surf = self.font_body.render(act_name, True, TEXT_PRIMARY)
            self.screen.blit(n_surf, (px + 12, py + 62))

            # Reward for last action
            lr = data["last_reward"]
            lr_col = ACCENT_GREEN if lr >= 0 else ACCENT_RED
            lr_surf = self.font_header.render(f"Reward: {lr:+.2f}", True, lr_col)
            self.screen.blit(lr_surf, (px + 12, py + 90))
        else:
            self.screen.blit(
                self.font_body.render("Awaiting first action...", True, TEXT_MUTED),
                (px + 12, py + 38),
            )

        # If terminated — show final result
        if data["terminated"]:
            t_surf = self.font_header.render("EPISODE COMPLETE", True, ACCENT_ORANGE)
            self.screen.blit(t_surf, (px + 12, py + 130))

        # Legend: action colour key
        self.screen.blit(
            self.font_small.render("Action colour key:", True, TEXT_MUTED), (px + 12, py + 165)
        )
        pairs = [("Record/Filter", ACCENT_BLUE), ("Analysis", ACCENT_PURPLE),
                 ("Classify", ACCENT_GREEN)]
        for i, (lbl, col) in enumerate(pairs):
            pygame.draw.rect(self.screen, col, (px + 12, py + 183 + i * 18, 10, 10))
            self.screen.blit(
                self.font_small.render(lbl, True, TEXT_MUTED), (px + 28, py + 182 + i * 18)
            )

    def _draw_action_log(self, data):
        """Bottom-right — scrolling action history."""
        px, py, pw, ph = 300, 308, 590, 250
        self._panel(px, py, pw, ph, "ACTION LOG")

        for i, (name, reward, act_id) in enumerate(reversed(self._action_log)):
            col = ACTION_COLORS.get(act_id, ACCENT_BLUE)
            row_y = py + 38 + i * 24
            # Step bullet
            pygame.draw.circle(self.screen, col, (px + 20, row_y + 8), 5)
            # Action name
            self.screen.blit(
                self.font_body.render(name, True, TEXT_PRIMARY), (px + 32, row_y)
            )
            # Reward
            r_col = ACCENT_GREEN if reward >= 0 else ACCENT_RED
            r_txt = self.font_body.render(f"{reward:+.2f}", True, r_col)
            self.screen.blit(r_txt, (px + pw - 70, row_y))

    def _draw_progress_bar(self, data):
        """Bottom strip — step progress bar."""
        by = H - 38
        pygame.draw.rect(self.screen, BG_PANEL, (0, by, W, 38))
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, by), (W, by), 1)

        frac = data["step"] / max(data["max_steps"], 1)
        bar_w = int((W - 200) * frac)
        pygame.draw.rect(self.screen, BAR_BG, (10, by + 10, W - 200, 16), border_radius=4)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (10, by + 10, bar_w, 16), border_radius=4)

        self.screen.blit(
            self.font_body.render(
                f"Progress: {data['step']}/{data['max_steps']} steps", True, TEXT_MUTED
            ),
            (W - 185, by + 12),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _panel(self, x, y, w, h, title):
        pygame.draw.rect(self.screen, BG_PANEL, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, BG_PANEL2, (x, y, w, 26), border_radius=6)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x, y, w, h), width=1, border_radius=6)
        t = self.font_header.render(title, True, ACCENT_BLUE)
        self.screen.blit(t, (x + 10, y + 6))

    def _labelled_bar(self, x, y, w, h, val, label, color):
        pygame.draw.rect(self.screen, BAR_BG, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, int(w * val), h), border_radius=4)
        lbl = self.font_small.render(f"{label}: {val:.3f}", True, TEXT_PRIMARY)
        self.screen.blit(lbl, (x + 4, y + 3))

    def _severity_color(self, sev):
        if sev < 0.33:
            return SEVERITY_LOW
        elif sev < 0.66:
            return SEVERITY_MED
        return SEVERITY_HI
