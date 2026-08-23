"""
core/landmark_matcher.py
========================
ONLINE step — used during navigation every N frames to confirm the drone is
on the correct path toward its target.

How it works:
  1. The drone's camera captures a frame.
  2. We compute which landmarks from the database should theoretically be
     visible given the drone's current estimated position (spatial filter).
  3. For each nearby landmark we attempt an ORB feature match between the
     live frame and the landmark's stored patch.
  4. If enough matches (inliers) are found, we confirm:
       "LANDMARK [LABEL] confirmed — you are on the correct path."
  5. The confirmed landmark's known map position provides an additional
     absolute-position correction (independent of DINOv2).

This gives the navigation loop a second source of position evidence — purely
from recognisable visual features — improving robustness when DINOv2
retrieval is uncertain.
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from core import config as C


class LandmarkMatcher:
    def __init__(self, db_path=None):
        db_path = db_path or str(C.LANDMARK_DIR / "landmark_db.pkl")
        with open(db_path, "rb") as f:
            self.db = pickle.load(f)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        print(f"[LandmarkMatcher] loaded {len(self.db)} landmarks")

    def nearby(self, est_x, est_y, radius_px=250):
        """Return landmarks within radius_px of the current estimated position."""
        near = []
        for lm in self.db:
            cx, cy = lm["pixel"]
            if np.hypot(cx - est_x, cy - est_y) <= radius_px:
                near.append(lm)
        return near

    def match_frame(self, frame_bgr, est_x, est_y, radius_px=250,
                    min_inliers=8):
        """
        Try to match the current drone frame against nearby landmarks.

        Returns a list of confirmed landmarks:
          [{"label", "type", "pixel", "gps", "inliers", "distance_m"}, ...]
        Each confirmed landmark is a visual proof the drone is near that
        known location — path confirmation.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        if des_f is None or len(des_f) < 4:
            return []

        confirmed = []
        for lm in self.nearby(est_x, est_y, radius_px):
            des_lm = lm.get("descriptor")
            if des_lm is None or len(des_lm) < 4:
                continue
            try:
                matches = self.bf.match(des_f, des_lm)
            except Exception:
                continue

            matches = sorted(matches, key=lambda m: m.distance)[:50]
            if len(matches) < 4:
                continue

            # RANSAC homography to count inliers
            pts_f  = np.float32([kp_f[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts_lm = np.float32([
                (m.trainIdx % lm["patch_size"], m.trainIdx // lm["patch_size"])
                for m in matches
            ]).reshape(-1,1,2)

            _, mask = cv2.findHomography(pts_f, pts_lm, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0

            if inliers >= min_inliers:
                cx, cy = lm["pixel"]
                dist_px = float(np.hypot(cx - est_x, cy - est_y))
                dist_m  = round(dist_px * 0.3, 1)   # assuming gsd=0.3m/px
                confirmed.append({
                    "label":    lm["label"],
                    "type":     lm["type"],
                    "pixel":    lm["pixel"],
                    "gps":      lm["gps"],
                    "inliers":  inliers,
                    "distance_m": dist_m,
                })

        return confirmed

    def position_correction(self, confirmed, est_x, est_y):
        """
        If one or more landmarks are confirmed, compute a weighted position
        correction toward the known landmark positions.
        Returns corrected (x, y) or original if no landmarks confirmed.
        """
        if not confirmed:
            return est_x, est_y
        # weighted average of confirmed landmark positions (weight = inliers)
        total_w = sum(c["inliers"] for c in confirmed)
        cx = sum(c["pixel"][0] * c["inliers"] for c in confirmed) / total_w
        cy = sum(c["pixel"][1] * c["inliers"] for c in confirmed) / total_w
        # blend 30% toward landmark position (soft correction)
        nx = est_x * 0.7 + cx * 0.3
        ny = est_y * 0.7 + cy * 0.3
        return float(nx), float(ny)
