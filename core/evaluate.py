"""
core/evaluate.py
================
Measures localization accuracy against ground-truth positions.

Run after generating samples + building the index:
    python -m core.evaluate --gsd 0.3
(--gsd = ground sample distance, metres per pixel of the satellite map)
"""

import os
import csv
import argparse
import numpy as np
import cv2

from core.localizer import Localizer


def run(samples_dir, model_dir, gsd, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    loc = Localizer(model_dir=model_dir, do_refine=True)

    gt = {}
    with open(os.path.join(samples_dir, "ground_truth.csv")) as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = (int(row["true_x"]), int(row["true_y"]))

    rows, errs_px, lat_ms = [], [], []
    for fname, (tx, ty) in gt.items():
        frame = cv2.imread(os.path.join(samples_dir, fname))
        if frame is None:
            continue
        r = loc.locate(frame)
        e_px = float(np.hypot(r["pixel_x"]-tx, r["pixel_y"]-ty))
        errs_px.append(e_px)
        lat_ms.append(r["timing"]["total_ms"])
        rows.append({
            "filename": fname, "true_x": tx, "true_y": ty,
            "est_x": r["pixel_x"], "est_y": r["pixel_y"],
            "error_px": round(e_px, 1), "error_m": round(e_px*gsd, 2),
            "confidence": r["confidence"], "inliers": r["inliers"],
            "latency_ms": r["timing"]["total_ms"],
        })
        print(f"  {fname:16s} err={e_px*gsd:6.1f}m  "
              f"conf={r['confidence']:.3f}  {r['timing']['total_ms']:.0f}ms")

    errs_m = np.array(errs_px) * gsd
    print("\n" + "="*52)
    print(f"  ACCURACY REPORT  ({len(errs_m)} frames, gsd={gsd} m/px)")
    print("="*52)
    print(f"  mean error      : {errs_m.mean():.2f} m")
    print(f"  median error    : {np.median(errs_m):.2f} m")
    print(f"  90th percentile : {np.percentile(errs_m,90):.2f} m")
    print(f"  < 10 m          : {(errs_m<10).mean()*100:.0f}%")
    print(f"  < 25 m          : {(errs_m<25).mean()*100:.0f}%")
    print(f"  mean latency    : {np.mean(lat_ms):.0f} ms")
    print("="*52)

    with open(os.path.join(out_dir, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"  per-frame CSV → {out_dir}/eval_results.csv")


if __name__ == "__main__":
    from core import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(C.QUERY_DIR))
    ap.add_argument("--models",  default=str(C.MODELS_DIR))
    ap.add_argument("--gsd",     type=float, default=0.3, help="metres per pixel")
    ap.add_argument("--out",     default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    run(a.samples, a.models, a.gsd, a.out)
