# VISION-NAV — GPS-Free UAV Localization for IIT Roorkee Campus

A camera-only navigation system that tells a UAV exactly where it is over the
IIT Roorkee campus by matching its live camera frame against a pre-stored
satellite map — **no GPS, no IMU, no altimeter**. Instant visual output via a
mission-control web interface.

This is a complete, final-year-project-grade implementation built around the
two best-in-class transformer models for the job:

| Role | Model | Why |
|------|-------|-----|
| **Global retrieval** | **DINOv2 ViT-S/14** | Self-supervised vision transformer with view-invariant features — recognises the same place across the satellite↔drone appearance gap |
| **Local matching** | **EfficientLoFTR** (CVPR 2024) | Detector-free dense transformer matcher — works on textureless campus regions (lawns, rooftops, roads) where ORB/SIFT fail |
| **Retrieval index** | **FAISS** | Sub-millisecond nearest-neighbour search across all map tiles |

If a model can't be downloaded (offline machine), the code automatically falls
back (EfficientNet-B0 for retrieval, ORB for matching) so the pipeline always runs.

---

## Architecture

```
                    OFFLINE (run once)                ONLINE (per frame, instant)
   ┌──────────────────────────────────┐      ┌────────────────────────────────────┐
   │  Satellite map (IITR campus)      │      │   Live UAV camera frame             │
   │            │                      │      │            │                        │
   │      tile into patches            │      │   DINOv2 embedding   ── Role 1      │
   │            │                      │      │            │                        │
   │   DINOv2 embed every tile         │      │   FAISS top-k search  (<1 ms)       │
   │            │                      │      │            │                        │
   │      build FAISS index ───────────┼──────┼──> top-5 candidate tiles            │
   └──────────────────────────────────┘      │            │                        │
                                              │   EfficientLoFTR dense match ─ Role 2│
                                              │            │                        │
                                              │   RANSAC homography                 │
                                              │            │                        │
                                              │   refined (x, y) → GPS (lat, lon)   │
                                              └────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Run the whole pipeline (simulate data → build index → evaluate)
bash run_all.sh

# 3. Launch the live demo
python -m web.app
#    → open http://localhost:5000
```

The satellite map of IIT Roorkee is already included at
`data/satellite_map/iitr_campus.png`.

---

## Step-by-step (manual)

### 1. Generate test "UAV" frames (optional, for demo/eval)
```bash
python -m core.simulate --n 20 --size 300
```
Crops 20 patches from the map at known positions and applies drone-like
augmentations (rotation, perspective tilt, brightness, blur, colour shift).
Writes `ground_truth.csv` so accuracy can be measured exactly.

### 2. Build the map index (run once, offline)
```bash
python -m core.map_index --scales 256,384 --stride 48 --backbone dinov2
```
Tiles the map at multiple scales (so any drone altitude matches), embeds every
tile with DINOv2, builds the FAISS index.
Output: `models/map.faiss` + `models/map_meta.pkl`.

| Flag | Meaning | Default |
|------|---------|---------|
| `--scales` | comma-separated tile sizes (multi-scale) | 256,384 |
| `--stride` | gap between tiles (px) — smaller = finer position resolution | 48 |
| `--backbone` | `dinov2` or `efficientnet` | dinov2 |

**How accuracy is achieved:** DINOv2 retrieves the top-5 candidate tiles, then
EfficientLoFTR matches the query against *each* candidate and the one with the most
geometric (RANSAC) inliers wins. This re-ranking recovers the correct location even
when retrieval's first guess is wrong, and the confidence score reflects true match
quality (inlier count), not just visual similarity.

### 3. Localize
```bash
python -m web.app            # interactive web demo (recommended)
```
or in code:
```python
from core.localizer import Localizer
import cv2

loc = Localizer()                       # loads once (~1s)
frame = cv2.imread("uav_frame.png")
r = loc.locate(frame)                   # ~30–180 ms

print(r["pixel_x"], r["pixel_y"], r["confidence"])
lat, lon = loc.pixel_to_gps(r["pixel_x"], r["pixel_y"],
                            29.8675, 77.8930, 29.8590, 77.9080)
cv2.imwrite("fix.jpg", loc.render(r, frame))   # annotated map
```

### 4. Evaluate accuracy
```bash
python -m core.evaluate --gsd 0.3        # gsd = metres per pixel of your map
```
Prints mean/median error in metres, percentage of frames under 10 m / 25 m,
and mean latency. Writes `results/eval_results.csv`.

---

## The web demo

A mission-control style interface (`web/app.py` + `web/templates/index.html`):

- Drag-and-drop a UAV frame (or click the SAMPLE buttons)
- Hit **LOCALIZE** — a scan animation runs, then the map zooms to the fix
- Live readout of map coordinates, GPS lat/lon, confidence, RANSAC inliers
- Per-stage latency bars (retrieval / FAISS / matcher / total)
- The estimated position is marked with a red crosshair; other top-k
  candidates appear as amber rings

Everything runs server-side through one persistent `Localizer` instance, so
each query is processed instantly without reloading models.

---

## Project layout

```
iitr_uav_nav/
├── core/
│   ├── models.py        DINOv2 backbone + EfficientLoFTR matcher (+ fallbacks)
│   ├── map_index.py     OFFLINE: tile map → embed → FAISS index
│   ├── localizer.py     ONLINE: retrieve → match → refine → (x,y) + render
│   ├── simulate.py      generate drone-like test frames with ground truth
│   └── evaluate.py      accuracy report in metres
├── web/
│   ├── app.py           Flask server (loads engine once, serves /locate)
│   ├── templates/index.html   mission-control demo UI
│   └── static/samples/  sample query frames
├── data/
│   ├── satellite_map/iitr_campus.png   the IITR map (included)
│   └── query_samples/   generated test frames + ground_truth.csv
├── models/              FAISS index + metadata (built by map_index.py)
├── results/             evaluation outputs
├── requirements.txt
└── run_all.sh
```

---

## Performance

Measured per frame (excludes one-time model load):

| Stage | CPU | GPU (T4) |
|-------|-----|----------|
| DINOv2 retrieval | 30–80 ms | 6–12 ms |
| FAISS search | <1 ms | <1 ms |
| EfficientLoFTR match | 40–120 ms | 25–35 ms |
| **Total** | **~80–200 ms** | **~30–50 ms** |

Expected localization error with real DINOv2 weights: single-digit to low-tens
of metres, depending on map resolution and altitude.

---

## Adapting to a different area / real flight

1. Replace `data/satellite_map/iitr_campus.png` with your area's satellite image
   (Google Earth Pro → Save Image at max resolution).
2. Update the GPS bounding box in `web/app.py` (`GPS_BBOX`) to your map's corners.
3. Rebuild the index: `python -m core.map_index`.
4. For real flights, feed live frames into `Localizer.locate()` instead of
   simulated ones.

---

## Notes on model downloads

The first run downloads model weights:
- DINOv2 from `torch.hub` (facebookresearch/dinov2)
- EfficientLoFTR from HuggingFace (`zju-community/efficientloftr`)

These require internet on first run; afterwards they're cached locally. On a
fully offline machine the code prints a warning and uses the EfficientNet + ORB
fallback automatically — the whole pipeline still runs.

---

## Credits

- DINOv2 — Oquab et al., Meta AI, 2023
- EfficientLoFTR — Wang et al., CVPR 2024
- FAISS — Johnson et al., Meta AI
- Problem formulation after Goforth & Lucey, "GPS-Denied UAV Localization
  using Pre-existing Satellite Imagery", ICRA 2019
