"""
core/landmark_detector.py
=========================
OFFLINE step — detects ONLY meaningful, navigation-useful landmarks on the
satellite map and builds a landmark database for path confirmation.

DESIGN PRINCIPLE (important):
  We deliberately detect landmarks that are UNIQUE and DISTINCTIVE — things a
  drone can reliably re-identify from the air and that pin down location:

    BUILDING   — large distinctive building footprints / rooftops with a
                 strong, well-defined rectangular structure (e.g. the Main
                 Building, the courtyard block, the red-roof block).
    JUNCTION   — road / path intersections (crossings of linear roads).
    COURT      — sports courts / rectangular painted surfaces (high-contrast,
                 very distinctive from above).
    COMPLEX    — visually complex unique regions (high local structure /
                 entropy) that are not plain grass or tree canopy.

  We EXPLICITLY REJECT useless landmarks:
    • plain tree canopy / generic vegetation (low structure, green, repetitive)
    • flat uniform grass or bare ground (low texture)
    • low-contrast featureless patches
  These are filtered out by a "distinctiveness score" gate, so we never store a
  random tree or empty field as a landmark.

Each landmark stores: label, type, pixel position, GPS, an ORB descriptor of a
patch around it, a distinctiveness score, and a saved thumbnail.

Run:
    python -m core.landmark_detector
"""

import os, pickle, argparse
import numpy as np
import cv2
from pathlib import Path

from core import config as C


# ── colour + symbol per type (for the annotated map) ────────────────────────
TYPE_COLOUR = {
    "BUILDING": (0, 160, 255),    # orange
    "JUNCTION": (0, 220, 255),    # cyan
    "COURT":    (255, 90, 200),   # magenta/pink
    "COMPLEX":  (0, 255, 120),    # green
}
TYPE_SYMBOL = {
    "BUILDING": cv2.MARKER_SQUARE,
    "JUNCTION": cv2.MARKER_CROSS,
    "COURT":    cv2.MARKER_DIAMOND,
    "COMPLEX":  cv2.MARKER_STAR,
}


# ════════════════════════════════════════════════════════════════════════════
# Distinctiveness scoring — the gate that rejects generic / useless spots
# ════════════════════════════════════════════════════════════════════════════
def _local_entropy(gray_patch):
    """Shannon entropy of the patch histogram — high = visually complex."""
    hist = cv2.calcHist([gray_patch], [0], None, [32], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())          # 0..5


def _edge_density(gray_patch):
    """Fraction of edge pixels — high = strong man-made structure."""
    edges = cv2.Canny(gray_patch, 60, 160)
    return float((edges > 0).mean())                # 0..1


def _greenness(bgr_patch):
    """Fraction of vegetation-green pixels — high = tree/grass (to penalise)."""
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 30), (90, 255, 210))
    return float((green > 0).mean())                # 0..1


def distinctiveness(bgr_patch):
    """
    Combine the cues into a single 0..1 usefulness score.
    Rewards structure (edges) and visual complexity (entropy);
    penalises vegetation (greenness) and flat low-contrast patches.
    """
    if bgr_patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    ent  = _local_entropy(gray) / 5.0               # → 0..1
    edg  = _edge_density(gray)                      # 0..1
    grn  = _greenness(bgr_patch)                    # 0..1
    contrast = min(1.0, gray.std() / 60.0)          # 0..1

    score = (0.40 * edg + 0.30 * ent + 0.30 * contrast) * (1.0 - 0.8 * grn)
    return float(np.clip(score, 0, 1))


def _nms(points, scores, min_dist):
    """Keep the strongest point within each min_dist neighbourhood."""
    if len(points) == 0:
        return []
    scores = np.asarray(scores, np.float32)
    order  = np.argsort(-scores)
    kept, used = [], np.zeros(len(points), bool)
    for i in order:
        if used[i]:
            continue
        kept.append(i)
        cx, cy = points[i]
        for j in range(len(points)):
            if not used[j]:
                dx, dy = points[j][0]-cx, points[j][1]-cy
                if dx*dx + dy*dy < min_dist*min_dist:
                    used[j] = True
    return kept


# ════════════════════════════════════════════════════════════════════════════
# Type-specific detectors — each proposes candidate centres
# ════════════════════════════════════════════════════════════════════════════
def detect_buildings(gray, bgr, min_dist, patch, want):
    """
    Distinctive buildings = large bright rectangular rooftop blobs.
    We find big bright regions, keep those that are reasonably rectangular
    and sizeable (not small noise), and use their centroids.
    """
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cands, scores = [], []
    H, W = gray.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:                       # ignore small bright specks
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect_fill = area / (w * h + 1e-6)     # how rectangular / solid
        if rect_fill < 0.45:
            continue
        cx, cy = x + w//2, y + h//2
        half = patch//2
        sub = bgr[max(0,cy-half):cy+half, max(0,cx-half):cx+half]
        d = distinctiveness(sub)
        if d < 0.18:                          # reject low-structure
            continue
        cands.append([cx, cy])
        scores.append(d * np.log(area))       # bigger + distinctive = better
    keep = _nms(cands, scores, min_dist)[:want]
    return [{"pos": cands[i], "type": "BUILDING"} for i in keep]


def detect_junctions(gray, bgr, min_dist, patch, want):
    """
    Road/path intersections. Build a road mask (mid-dark, low-green linear
    regions), skeleton-ish via morphology, then find corners ON the road mask
    where multiple road directions meet.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    notgreen = cv2.bitwise_not(cv2.inRange(hsv, (30,35,30), (90,255,210)))
    _, midband = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    _, notbright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    road = cv2.bitwise_and(cv2.bitwise_and(midband, notbright), notgreen)
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN,
                            np.ones((3,3), np.uint8), iterations=1)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE,
                            np.ones((5,5), np.uint8), iterations=2)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.02,
                                      minDistance=min_dist//2, mask=road)
    if corners is None:
        return []
    cands, scores = [], []
    half = patch//2
    for p in corners.reshape(-1, 2).astype(int):
        cx, cy = int(p[0]), int(p[1])
        sub = bgr[max(0,cy-half):cy+half, max(0,cx-half):cx+half]
        # a junction should have road in several directions → some edge density
        d = _edge_density(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)) if sub.size else 0
        if d < 0.06:
            continue
        cands.append([cx, cy]); scores.append(d)
    keep = _nms(cands, scores, min_dist)[:want]
    return [{"pos": cands[i], "type": "JUNCTION"} for i in keep]


def detect_courts(gray, bgr, min_dist, patch, want):
    """
    Sports courts / painted rectangular surfaces — strongly coloured
    (blue/red) rectangular regions, very distinctive from the air.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # bluish or reddish saturated surfaces
    blue = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))
    red1 = cv2.inRange(hsv, (0, 60, 40), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 60, 40), (180, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(blue, red1), red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cands, scores = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 400:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cands.append([cx, cy]); scores.append(area)
    keep = _nms(cands, scores, min_dist)[:want]
    return [{"pos": cands[i], "type": "COURT"} for i in keep]


def detect_complex(gray, bgr, min_dist, patch, want):
    """
    Generic distinctive regions: scan a grid, score each cell's patch by
    distinctiveness, keep only the highest-scoring NON-vegetation cells.
    This captures unique structures the other detectors miss, while the
    score gate guarantees we never keep plain trees/grass.
    """
    H, W = gray.shape
    step = max(24, patch//2)
    half = patch//2
    cands, scores = [], []
    for cy in range(half, H-half, step):
        for cx in range(half, W-half, step):
            sub = bgr[cy-half:cy+half, cx-half:cx+half]
            d = distinctiveness(sub)
            if d >= 0.45:                     # only genuinely complex/unique
                cands.append([cx, cy]); scores.append(d)
    keep = _nms(cands, scores, min_dist)[:want]
    return [{"pos": cands[i], "type": "COMPLEX"} for i in keep]


def build_descriptor(gray, cx, cy, patch_size):
    half = patch_size//2
    H, W = gray.shape
    x1, y1 = max(0, cx-half), max(0, cy-half)
    x2, y2 = min(W, cx+half), min(H, cy+half)
    sub = gray[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    sub = cv2.resize(sub, (patch_size, patch_size))
    orb = cv2.ORB_create(nfeatures=300)
    _, des = orb.detectAndCompute(sub, None)
    return des


def detect_all(map_path=None, out_dir=None,
               min_dist=None, max_landmarks=None, patch_size=None):
    map_path      = map_path or str(C.DEFAULT_MAP)
    out_dir       = Path(out_dir or C.LANDMARK_DIR)
    min_dist      = min_dist or C.LANDMARK_MIN_DIST
    max_landmarks = max_landmarks or C.MAX_LANDMARKS
    patch_size    = patch_size or C.LANDMARK_PATCH_SIZE

    bgr = cv2.imread(map_path)
    if bgr is None:
        raise FileNotFoundError(map_path)
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"[landmark_detector] Map {W}x{H}  (meaningful-landmarks mode)")

    # how many of each type to aim for (buildings + junctions prioritised)
    raw = (detect_buildings(gray, bgr, min_dist, patch_size, want=14) +
           detect_junctions(gray, bgr, min_dist, patch_size, want=12) +
           detect_courts   (gray, bgr, min_dist, patch_size, want=6)  +
           detect_complex  (gray, bgr, min_dist, patch_size, want=12))

    # score every candidate by distinctiveness, drop weak ones, global NMS
    half = patch_size//2
    scored = []
    for r in raw:
        cx, cy = r["pos"]
        sub = bgr[max(0,cy-half):cy+half, max(0,cx-half):cx+half]
        r["score"] = distinctiveness(sub)
        if r["score"] >= 0.16:               # final usefulness gate
            scored.append(r)

    pts    = [r["pos"] for r in scored]
    scores = [r["score"] for r in scored]
    keep   = _nms(pts, scores, min_dist)[:max_landmarks]
    chosen = [scored[i] for i in keep]

    # label per type and store
    counters = {}
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    db = []
    for r in chosen:
        t = r["type"]
        counters[t] = counters.get(t, 0) + 1
        prefix = {"BUILDING":"BL","JUNCTION":"JN","COURT":"CT","COMPLEX":"CX"}[t]
        label = f"{prefix}{counters[t]:02d}"
        cx, cy = r["pos"]
        des = build_descriptor(gray, cx, cy, patch_size)
        lat = C.GPS_BBOX["tl_lat"] + (cy/H)*(C.GPS_BBOX["br_lat"]-C.GPS_BBOX["tl_lat"])
        lon = C.GPS_BBOX["tl_lon"] + (cx/W)*(C.GPS_BBOX["br_lon"]-C.GPS_BBOX["tl_lon"])
        x1,y1 = max(0,cx-half), max(0,cy-half)
        thumb = bgr[y1:cy+half, x1:cx+half]
        db.append({
            "label": label, "type": t, "pixel": (cx, cy),
            "gps": (round(lat,6), round(lon,6)),
            "descriptor": des, "patch_size": patch_size,
            "score": round(r["score"], 3),
        })
        cv2.imwrite(str(out_dir/"patches"/f"{label}.png"), thumb)

    with open(out_dir/"landmark_db.pkl", "wb") as f:
        pickle.dump(db, f)

    # annotated map
    vis = bgr.copy()
    for e in db:
        cx, cy = e["pixel"]; col = TYPE_COLOUR[e["type"]]; sym = TYPE_SYMBOL[e["type"]]
        cv2.drawMarker(vis, (cx,cy), col, sym, 24, 2)
        cv2.putText(vis, e["label"], (cx+13, cy-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    ly = 20
    for t, col in TYPE_COLOUR.items():
        cv2.putText(vis, f"  {t}", (8, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2); ly += 22
    cv2.imwrite(str(out_dir/"landmark_map.png"), vis)

    print(f"[landmark_detector] kept {len(db)} MEANINGFUL landmarks "
          f"(generic trees/grass rejected by score gate)")
    for t in TYPE_COLOUR:
        print(f"  {t}: {sum(1 for e in db if e['type']==t)}")
    print(f"  database → {out_dir/'landmark_db.pkl'}")
    print(f"  map      → {out_dir/'landmark_map.png'}")
    return db


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map",      default=str(C.DEFAULT_MAP))
    ap.add_argument("--out",      default=str(C.LANDMARK_DIR))
    ap.add_argument("--min_dist", type=int, default=C.LANDMARK_MIN_DIST)
    ap.add_argument("--max",      type=int, default=C.MAX_LANDMARKS)
    a = ap.parse_args()
    detect_all(a.map, a.out, a.min_dist, a.max)
