"""
core/pose_estimator.py
======================
Maintains the drone's estimated position and confidence by fusing
three independent sources of evidence:

  SOURCE 1 — Optical-flow (fast, ~5 ms every frame)
    Tracks how the scene moves between consecutive frames.
    Updates position every frame but accumulates drift over time.
    Confidence decays slightly each frame.

  SOURCE 2 — DINOv2 / EfficientLoFTR absolute fix (slow, ~100 ms, background)
    Matches the live frame against the full satellite map.
    Gives a globally accurate position but runs only every few seconds.
    High-confidence fixes can reset the estimate.

  SOURCE 3 — Landmark match correction (medium, ~20 ms, every N frames)
    When a known landmark is recognised in the live frame, its known
    map position provides a localised correction.
    Acts as a soft anchor — partially pulls the estimate toward truth.

The fusion uses exponential blending (no Kalman for simplicity, but
the structure maps directly to a Kalman if you want to extend it).
Confidence is a 0-1 score; navigation decisions use it to decide
when to re-plan.
"""

import time
import numpy as np


class PoseEstimator:
    """
    Maintains (x, y, confidence) position state of the drone on the map.

    Parameters
    ----------
    init_x, init_y : initial position estimate (pixels on the satellite map)
    init_conf      : starting confidence (0–1)
    map_w, map_h   : map dimensions for clamping
    flow_decay     : confidence decay per optical-flow step (e.g. 0.998)
    fix_alpha      : blending weight for DINOv2 fix (1.0 = full override)
    lm_alpha       : blending weight for landmark correction
    """

    def __init__(self, init_x, init_y, init_conf=0.5,
                 map_w=1280, map_h=590,
                 flow_decay=0.998,
                 fix_alpha=0.35,
                 lm_alpha=0.35):
        self.x    = float(init_x)
        self.y    = float(init_y)
        self.conf = float(init_conf)

        self.map_w = map_w
        self.map_h = map_h

        self.flow_decay = flow_decay   # per-frame confidence decay
        self.fix_alpha  = fix_alpha    # DINOv2 blend weight
        self.lm_alpha   = lm_alpha     # landmark blend weight

        # history for debugging / visualisation
        self.history = [(self.x, self.y, self.conf)]
        self._last_fix_time = 0.0
        self._last_lm_time  = 0.0

    # ── Source 1: optical flow ─────────────────────────────────────────────
    def update_flow(self, dx: float, dy: float):
        """
        Apply scene-motion (dx, dy) to update position.

        dx, dy are the measured pixel shift of the scene in the image.
        If the scene moved right by dx, the drone moved left (we subtract).

        Confidence decays slightly — drift accumulates over time.
        """
        self.x    = float(np.clip(self.x + dx, 0, self.map_w - 1))
        self.y    = float(np.clip(self.y + dy, 0, self.map_h - 1))
        self.conf = max(0.05, self.conf * self.flow_decay)
        self.history.append((self.x, self.y, self.conf))

    # ── Source 2: DINOv2 / EfficientLoFTR absolute fix ────────────────────
    def update_fix(self, fix_x: float, fix_y: float, fix_conf: float,
                   min_conf=0.45, motion_since=(0.0, 0.0), heading=None):
        """
        Apply a global localisation fix from the retrieval system.

        Only applied if fix_conf >= min_conf.

        motion_since : (mx, my) — how far the drone has moved since the frame
                       this fix was computed on. The fix is advanced by this
                       so it corrects to "now", not to a stale past position.
        heading      : optional (hx, hy) unit vector of travel direction. If a
                       fix would shove the estimate strongly *backward* against
                       the heading (a sign of a stale/wrong fix), it is damped.
        """
        if fix_conf < min_conf:
            return False

        # 1) latency compensation — advance the fix by motion during compute
        fix_x = fix_x + motion_since[0]
        fix_y = fix_y + motion_since[1]

        # 2) backward-jump rejection — damp fixes that point against heading
        dx, dy = fix_x - self.x, fix_y - self.y
        damp = 1.0
        if heading is not None:
            hx, hy = heading
            # component of the correction along the (reverse) heading
            backward = -(dx*hx + dy*hy)
            jump = float(np.hypot(dx, dy))
            if backward > 0 and jump > 40:      # large backward correction
                damp = 0.25                      # treat as likely stale → weak

        # 3) adaptive + damped blend
        alpha = self.fix_alpha * min(1.0, fix_conf / 0.9) * damp
        self.x = alpha * fix_x + (1 - alpha) * self.x
        self.y = alpha * fix_y + (1 - alpha) * self.y

        # confidence eases toward the fix confidence (smooth, not a hard reset)
        self.conf = 0.6 * self.conf + 0.4 * fix_conf

        self.x = float(np.clip(self.x, 0, self.map_w - 1))
        self.y = float(np.clip(self.y, 0, self.map_h - 1))
        self._last_fix_time = time.time()
        self.history.append((self.x, self.y, self.conf))
        return True

    # ── Source 3: landmark correction ─────────────────────────────────────
    def update_landmark(self, lm_x: float, lm_y: float, inliers: int,
                        min_inliers=8):
        """
        Soft-correct the position toward a confirmed landmark.

        The landmark's known pixel position acts as a local anchor.
        More inliers → stronger pull.
        """
        if inliers < min_inliers:
            return False

        # strength proportional to inlier count (capped at ~30)
        strength = self.lm_alpha * min(1.0, inliers / 30.0)
        self.x = (1 - strength) * self.x + strength * lm_x
        self.y = (1 - strength) * self.y + strength * lm_y

        # confirmed landmark boosts confidence
        bonus = 0.05 * min(1.0, inliers / 15.0)
        self.conf = min(1.0, self.conf + bonus)

        self.x = float(np.clip(self.x, 0, self.map_w - 1))
        self.y = float(np.clip(self.y, 0, self.map_h - 1))
        self._last_lm_time = time.time()
        self.history.append((self.x, self.y, self.conf))
        return True

    # ── Accessors ──────────────────────────────────────────────────────────
    def get_state(self):
        """Return (x, y, confidence)."""
        return self.x, self.y, self.conf

    def get_pos(self):
        return self.x, self.y

    def is_reliable(self, threshold=0.45):
        return self.conf >= threshold

    def seconds_since_fix(self):
        return time.time() - self._last_fix_time

    def seconds_since_landmark(self):
        return time.time() - self._last_lm_time
