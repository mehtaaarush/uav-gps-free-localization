"""
core/localizer.py
=================
ONLINE engine. Load once, query per frame.

Pipeline per query image:
  1. DINOv2 embedding             (Role 1)
  2. FAISS top-k retrieval        (<1 ms)
  3. EfficientLoFTR dense match   (Role 2) on best candidate
  4. RANSAC homography → refined sub-pixel (x, y) on the map

Returns position + confidence + per-stage latency.
"""

import os
import time
import pickle
import numpy as np
import cv2
import torch

from core.models import RetrievalBackbone, DenseMatcher, IMAGENET_MEAN, IMAGENET_STD
from core.map_index import make_transform

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


class Localizer:
    def __init__(self, model_dir=None,
                 retrieval="dinov2", matcher="efficientloftr",
                 top_k=5, do_refine=True):
        t0 = time.time()

        if model_dir is None:
            from core import config as C
            model_dir = str(C.MODELS_DIR)

        with open(os.path.join(model_dir, "map_meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)

        self.positions = np.array(self.meta["positions"], dtype=np.float32)
        self.map_size  = self.meta["map_size"]
        self.tile      = self.meta["tile"]
        # per-tile crop size (multi-scale); fall back to single tile size
        self.tile_scales = self.meta.get("tile_scales",
                                         [self.tile] * len(self.positions))
        self.top_k     = top_k
        self.do_refine = do_refine

        # retrieval index
        self.backend = self.meta["backend"]
        if self.backend == "faiss" and HAVE_FAISS:
            self.index = faiss.read_index(os.path.join(model_dir, "map.faiss"))
        else:
            self.feats = self.meta["feats"]   # numpy fallback

        # models
        self.backbone = RetrievalBackbone(prefer=retrieval)
        self.transform, _ = make_transform(self.backbone.kind, self.tile)
        self.matcher = DenseMatcher(prefer=matcher) if do_refine else None

        # full satellite map for refinement + visual output
        self.sat_bgr = cv2.imread(self.meta["map_path"])

        print(f"[Localizer] ready in {time.time()-t0:.2f}s | "
              f"retrieval={self.backbone.kind} | "
              f"matcher={self.matcher.kind if self.matcher else 'none'} | "
              f"tiles={len(self.positions)}")

    # ── retrieval ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _embed(self, frame_bgr):
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t   = self.transform(Image.fromarray(rgb)).unsqueeze(0)
        return self.backbone(t).numpy().astype("float32")

    def _search(self, q):
        if self.backend == "faiss" and HAVE_FAISS:
            scores, idx = self.index.search(q, self.top_k)
            return scores[0], idx[0]
        sims = (self.feats @ q[0])
        idx  = np.argsort(-sims)[:self.top_k]
        return sims[idx], idx

    # ── refinement ────────────────────────────────────────────────────────────
    def _refine(self, frame_bgr, cx, cy, tile=None):
        if self.matcher is None or self.sat_bgr is None:
            return cx, cy, 0
        tile = tile or self.tile
        half = tile // 2
        H, W = self.sat_bgr.shape[:2]
        x1, x2 = max(0, cx - half), min(W, cx + half)
        y1, y2 = max(0, cy - half), min(H, cy + half)
        sat_crop = self.sat_bgr[y1:y2, x1:x2]
        if sat_crop.size == 0:
            return cx, cy, 0

        uav = cv2.resize(frame_bgr, (sat_crop.shape[1], sat_crop.shape[0]))
        try:
            pts0, pts1, _ = self.matcher.match(uav, sat_crop)
        except Exception:
            return cx, cy, 0
        if pts0 is None or len(pts0) < 4:
            return cx, cy, int(len(pts0) if pts0 is not None else 0)

        pts0 = np.asarray(pts0, dtype=np.float32).reshape(-1, 1, 2)
        pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
        Hmat, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        if Hmat is None:
            return cx, cy, len(pts0)

        uh, uw = uav.shape[:2]
        centre = np.array([[[uw/2, uh/2]]], dtype=np.float32)
        proj   = cv2.perspectiveTransform(centre, Hmat)[0][0]
        rx, ry = int(x1 + proj[0]), int(y1 + proj[1])
        if np.hypot(rx-cx, ry-cy) > tile:
            return cx, cy, int(mask.sum())
        return rx, ry, int(mask.sum())

    # ── public API ────────────────────────────────────────────────────────────
    def locate(self, frame_bgr):
        t = {}
        t0 = time.time()

        s = time.time()
        q = self._embed(frame_bgr)
        t["retrieval_ms"] = (time.time()-s)*1000

        s = time.time()
        scores, idx = self._search(q)
        t["search_ms"] = (time.time()-s)*1000

        valid = idx[idx >= 0]
        topk  = [(int(self.positions[i][0]), int(self.positions[i][1]))
                 for i in valid]
        topk_scales = [int(self.tile_scales[i]) for i in valid]
        retr_conf = float(scores[0])

        # default = DINOv2's #1 pick
        best_cx, best_cy = topk[0]
        best_inliers = 0

        s = time.time()
        if self.do_refine and self.matcher is not None:
            # Re-rank EVERY top-k candidate by geometric match quality.
            # DINOv2 narrows the field; EfficientLoFTR decides the winner.
            # This recovers the correct tile even when retrieval's #1 is wrong.
            best_score = -1
            for (cx0, cy0), tsc in zip(topk, topk_scales):
                rx, ry, inl = self._refine(frame_bgr, cx0, cy0, tile=tsc)
                if inl > best_score:
                    best_score, best_inliers = inl, inl
                    best_cx, best_cy = rx, ry
        elif self.do_refine:
            best_cx, best_cy, best_inliers = self._refine(
                frame_bgr, best_cx, best_cy, tile=topk_scales[0])
        t["match_ms"] = (time.time()-s)*1000
        t["total_ms"] = (time.time()-t0)*1000

        # confidence = blend of retrieval score and match strength
        match_conf = min(1.0, best_inliers / 50.0)
        conf = round(0.4*max(retr_conf, 0) + 0.6*match_conf, 4) \
               if self.do_refine else round(retr_conf, 4)

        return {
            "pixel_x": best_cx, "pixel_y": best_cy,
            "confidence": conf,
            "inliers": best_inliers,
            "top_k": topk,
            "timing": {k: round(v, 1) for k, v in t.items()},
        }

    def pixel_to_gps(self, x, y, tl_lat, tl_lon, br_lat, br_lon):
        W, H = self.map_size
        lat = tl_lat + (y / H) * (br_lat - tl_lat)
        lon = tl_lon + (x / W) * (br_lon - tl_lon)
        return round(lat, 6), round(lon, 6)

    def render(self, result, frame_bgr=None):
        """Draw the estimate on the satellite map and return a BGR image."""
        vis = self.sat_bgr.copy()
        x, y = result["pixel_x"], result["pixel_y"]
        # candidate ring(s)
        for (tx, ty) in result["top_k"][1:]:
            cv2.circle(vis, (tx, ty), 14, (0, 180, 255), 2)
        # best estimate
        cv2.circle(vis, (x, y), 26, (0, 0, 255), 3)
        cv2.drawMarker(vis, (x, y), (0, 0, 255),
                       cv2.MARKER_CROSS, 40, 3)
        label = f"({x},{y})  conf={result['confidence']:.2f}  {result['timing']['total_ms']:.0f}ms"
        cv2.putText(vis, label, (x+30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return vis
