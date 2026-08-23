"""
core/tracker.py
===============
FAST LOOP — frame-to-frame landmark tracking via Lucas-Kanade optical flow.

This is the high-rate part of navigation. It does NOT run any neural network.
It detects distinctive landmarks once, then *tracks* them between consecutive
camera frames in 2-5 ms (CPU only), estimating how the drone moved.

Why optical flow and not a CNN here:
  - runs at 50-100+ Hz so a fast drone never drifts off target between fixes
  - the heavy DINOv2/EfficientLoFTR fix runs only ~1 Hz on a background thread
    to correct accumulated drift (see navigator.py)
"""

import numpy as np
import cv2


class LandmarkTracker:
    """
    Tracks landmark points across frames and reports the median pixel
    translation (dx, dy) of the scene between the last two frames.

    Usage:
        trk = LandmarkTracker()
        trk.init(first_frame)
        ...
        dx, dy, n = trk.update(next_frame)   # scene shift in pixels
    """

    def __init__(self, max_points=300, quality=0.01, min_dist=12):
        # parameters for detecting good landmark points
        self.feat_params = dict(maxCorners=max_points,
                                qualityLevel=quality,
                                minDistance=min_dist,
                                blockSize=7)
        # parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.prev_gray = None
        self.prev_pts = None
        self.min_track = 12          # re-detect if fewer than this remain

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(gray, mask=None, **self.feat_params)

    def init(self, frame_bgr):
        self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.prev_pts = self._detect(self.prev_gray)

    def update(self, frame_bgr):
        """
        Returns (dx, dy, n_tracked):
          dx, dy : median scene translation in pixels since last frame
                   (how far the ground appeared to move under the drone)
          n      : number of points successfully tracked
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_pts is None or len(self.prev_pts) < self.min_track:
            self.prev_pts = self._detect(gray)
            self.prev_gray = gray
            return 0.0, 0.0, 0

        # track points forward
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if nxt is None:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return 0.0, 0.0, 0

        good_old = self.prev_pts[status == 1]
        good_new = nxt[status == 1]

        if len(good_new) < 4:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return 0.0, 0.0, len(good_new)

        # robust scene translation = median of point displacements
        flow = good_new - good_old
        dx = float(np.median(flow[:, 0]))
        dy = float(np.median(flow[:, 1]))

        # carry good points forward; re-detect if too few
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        if len(self.prev_pts) < self.min_track:
            self.prev_pts = self._detect(gray)

        return dx, dy, int(len(good_new))
