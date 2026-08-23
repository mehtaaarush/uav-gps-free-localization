"""
core/simulate.py
================
Generates realistic "UAV camera" test images from the satellite map so you can
demo and evaluate without an actual flight.

It crops a patch at a known location and applies augmentations that mimic the
difference between a satellite view and a live drone camera:
  - rotation (drone heading)         - brightness / contrast (time of day)
  - perspective warp (drone tilt)    - blur (motion)  - colour shift (sensor)

The TRUE pixel position is recorded so accuracy can be measured exactly.
"""

import os
import csv
import random
import argparse
import numpy as np
import cv2
from PIL import Image


def augment(crop_bgr, rng):
    h, w = crop_bgr.shape[:2]
    img = crop_bgr.copy()

    # rotation (drone heading) — moderate
    ang = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # mild perspective (drone tilt)
    d = int(min(h, w) * 0.04)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[rng.uniform(0,d), rng.uniform(0,d)],
                      [w-rng.uniform(0,d), rng.uniform(0,d)],
                      [w-rng.uniform(0,d), h-rng.uniform(0,d)],
                      [rng.uniform(0,d), h-rng.uniform(0,d)]])
    P = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, P, (w, h), borderMode=cv2.BORDER_REFLECT)

    # brightness / contrast — mild
    alpha = rng.uniform(0.85, 1.15)   # contrast
    beta  = rng.uniform(-15, 15)      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # colour temperature shift — subtle
    img = img.astype(np.int16)
    img[:, :, 0] += int(rng.uniform(-8, 8))    # B
    img[:, :, 2] += int(rng.uniform(-8, 8))    # R
    img = np.clip(img, 0, 255).astype(np.uint8)

    # light motion blur occasionally
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate(map_path, out_dir, n=20, size=300, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    sat = cv2.imread(map_path)
    H, W = sat.shape[:2]
    rng = random.Random(seed)

    # auto-cap size so patches always fit regardless of map dimensions
    max_size = min(W, H) // 3
    if size > max_size:
        print(f"[simulate] size={size} too large for {W}x{H} map, "
              f"capping to {max_size}")
        size = max_size

    rows = []
    for i in range(n):
        half = size // 2
        # safe margins: centre must be at least half from each edge
        x = rng.randint(half, W - half)
        y = rng.randint(half, H - half)
        crop = sat[y-half:y+half, x-half:x+half]
        aug  = augment(crop, rng)

        fname = f"query_{i:03d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), aug)
        rows.append({"filename": fname, "true_x": x, "true_y": y})

    with open(os.path.join(out_dir, "ground_truth.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["filename", "true_x", "true_y"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"[simulate] wrote {n} query images + ground_truth.csv → {out_dir}")
    return rows


if __name__ == "__main__":
    from core import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--map",  default=str(C.DEFAULT_MAP))
    ap.add_argument("--out",  default=str(C.QUERY_DIR))
    ap.add_argument("--n",    type=int, default=20)
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()
    generate(a.map, a.out, a.n, a.size)
