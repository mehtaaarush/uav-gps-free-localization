# DYNAMIC LANDMARK-BASED NAVIGATION

This replaces fixed-path navigation with an integrated **Localization +
Landmark-Based Navigation** system. The drone never follows a precomputed line —
every navigation decision is made in real time from the map's landmarks.

## Run it

```bash
python quickstart.py --setup       # builds index + detects 45 landmarks
python -m core.navigator_dynamic --target data/query_samples/query_005.png
```

## The four stages (all integrated in `core/navigator_dynamic.py`)

### 1. Target Localization
The target photo is localized on the satellite map with DINOv2 + EfficientLoFTR,
giving its **map pixel position and GPS coordinates**.

### 2. Continuous Drone Localization  (`core/pose_estimator.py`)
A fused `PoseEstimator` maintains the drone's position + confidence from three
independent sources:
- **Optical flow** (every frame, ~5 ms) — tracks scene motion, accumulates drift
- **DINOv2 global fix** (background, periodic) — resets drift with an absolute fix
- **Landmark recognition** (every N frames) — soft-corrects toward known landmarks

Confidence decays with drift and jumps back up after a good fix or landmark match.

### 3. Dynamic Landmark-Based Navigation  (`core/route_planner.py`)
The 45 landmarks form a **navigation graph** (nodes = landmarks, edges = nearby
reachable pairs). At every re-plan, **A\*** finds the lowest-cost route from the
drone's *current* estimated position to the target through this graph.

Re-planning is triggered by:
- reaching the current way-landmark
- a significant **confidence change** (> 0.25)
- a periodic refresh (lets a better route emerge as the drone moves)

Because A\* always starts from wherever the drone actually is, the route is
**never precomputed** — it emerges from the landmarks and updates continuously.

### 4. Goal-Directed Autonomous Flight
The drone steers toward the next way-landmark (pure pursuit between nodes),
continuously verifies progress through the fused position, and stops when within
the target error threshold (default 20 m).

## What you see in the output image

- **grey lines** — the landmark navigation graph (all reachable connections)
- **coloured markers** — the 45 landmarks (junctions, buildings, structures, treelines)
- **bright circles** — landmarks confirmed during flight
- **magenta line** — the *current* dynamically-planned route (changes as it re-plans)
- **green line** — the drone's true path
- **cyan line** — the drone's estimated path (from fused localization)
- **red star + circle** — target and arrival zone

## Why this meets the requirement

| Requirement | How it's met |
|-------------|--------------|
| No precomputed path | A\* re-plans from current position every few steps |
| Locate target on map | Stage 0: DINOv2 localization → pixel + GPS |
| Continuous self-localization | PoseEstimator fuses flow + DINOv2 + landmarks |
| Use visual landmarks for navigation | Route planned through the landmark graph |
| Dynamic re-planning | Triggered by waypoint/confidence/refresh |
| Reach target in GPS-denied env | Goal-directed flight, vision-only, stops at threshold |

## Models used

- **DINOv2 ViT-S/14** — global retrieval (view-invariant embeddings)
- **EfficientLoFTR** — dense matching / re-ranking
- **FAISS** — sub-millisecond tile search
- **Lucas-Kanade optical flow** — fast per-frame tracking
- **ORB + RANSAC** — landmark recognition
- **A\*** — dynamic route planning over the landmark graph
