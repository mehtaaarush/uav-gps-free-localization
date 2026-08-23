# HOW TO RUN — Step by Step (Conda + RTX 4060)

This guide takes you from the downloaded zip to a working drone localizer that
prints the **exact location** of any image. Every command works no matter what
folder you run it from — all paths are anchored to the project automatically.

---

## STEP 1 — Extract the project

1. Move `iitr_uav_nav.zip` somewhere simple, e.g. `D:\IIT Roorkee\`
2. Right-click → Extract All
3. You'll get a folder `iitr_uav_nav` containing `core`, `web`, `data`,
   `quickstart.py`, `environment.yml`, etc.

> The folder name does NOT matter anymore — even `iitr_uav_nav (2)` works fine.

---

## STEP 2 — Open a terminal in that folder

Open **Anaconda Prompt** (search it in the Start menu), then navigate in:

```bash
cd "D:\IIT Roorkee\iitr_uav_nav"
```

(Use quotes — there's a space in the path.)

---

## STEP 3 — Create the conda environment

Easiest way — use the included environment file (sets up Python + CUDA PyTorch +
everything in one shot):

```bash
conda env create -f environment.yml
conda activate uav
```

This takes ~5–10 minutes (it downloads PyTorch with CUDA for your 4060).

> **If `conda env create` is slow or errors**, do it manually instead:
> ```bash
> conda create -n uav python=3.11 -y
> conda activate uav
> conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
> pip install faiss-cpu timm transformers opencv-python pillow numpy flask
> ```

---

## STEP 4 — Verify the GPU is being used

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

You MUST see:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060
```

If it says `False`, your PyTorch is CPU-only — redo Step 3's manual install path.

---

## STEP 5 — One-time setup (build the map index)

```bash
python quickstart.py --setup
```

What happens:
- Generates 20 sample drone frames from the campus map
- Downloads DINOv2 + EfficientLoFTR weights (~400 MB, first time only)
- Tiles the map at two scales and builds the FAISS search index

Wait for `✓ setup complete`. On your 4060 this takes about 1–2 minutes
(plus the one-time model download).

---

## STEP 6 — Localize an image (the main event)

### Localize one of the sample frames:
```bash
python quickstart.py --image data/query_samples/query_003.png --save
```

You'll get output like:
```
════════════════════════════════════════════════════════
            DRONE LOCATION RESULT
════════════════════════════════════════════════════════
  Image            : query_003.png
  Map position (px): x = 1820   y = 760
  GPS coordinates  : 29.861234, 77.902341
  Confidence       : 0.812   [HIGH — reliable fix]
  Match inliers    : 47
  Processing time  : 41 ms
════════════════════════════════════════════════════════
  Google Maps link : https://maps.google.com/?q=29.861234,77.902341
════════════════════════════════════════════════════════
Annotated map saved → results/located_query_003.png
```

- **Map position (px)** — where the drone is on the satellite map
- **GPS coordinates** — same point as latitude/longitude
- **Confidence** — HIGH (≥0.6) means a reliable fix
- The `--save` flag writes an annotated map (red crosshair = location) into `results/`

### Localize YOUR OWN drone image:
```bash
python quickstart.py --image "C:\path\to\your_drone_photo.jpg" --save
```

---

## STEP 7 — (Optional) Launch the live web demo

For a polished interface (great for your presentation):

```bash
python -m web.app
```

Wait for `Running on http://0.0.0.0:5000`, then open your browser to:
```
http://localhost:5000
```

Drag in a drone image (or click a SAMPLE button) and hit LOCALIZE. The map
zooms to the estimated position with full readouts. Press `Ctrl+C` in the
terminal to stop.

---

## Coming back later (after the first setup)

You only do Steps 3–5 once. Next time:
```bash
cd "D:\IIT Roorkee\iitr_uav_nav"
conda activate uav
python quickstart.py --image data/query_samples/query_005.png --save
```

---

## Getting the EXACT GPS for your map (important for accuracy)

The GPS coordinates use an approximate bounding box for the campus screenshot.
To make them precise:

1. Open your satellite map in Google Earth Pro
2. Hover over the **top-left corner** — note the lat/lon
3. Hover over the **bottom-right corner** — note the lat/lon
4. Open `core/config.py` and edit the `GPS_BBOX` values:
   ```python
   GPS_BBOX = {
       "tl_lat": 29.8675, "tl_lon": 77.8930,   # your top-left
       "br_lat": 29.8590, "br_lon": 77.9080,   # your bottom-right
   }
   ```

The pixel position is always exact; only the lat/lon conversion depends on this box.

---

## Tuning accuracy vs speed

Edit these in `core/config.py`:

| Setting | For more accuracy | For more speed |
|---------|-------------------|----------------|
| `STRIDE` | smaller (e.g. 32) | larger (e.g. 96) |
| `TOP_K` | larger (e.g. 8) | smaller (e.g. 3) |
| `TILE_SCALES` | `(192, 256, 384)` | `(256,)` |

After changing `STRIDE` or `TILE_SCALES`, rebuild the index:
`python quickstart.py --setup`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Reinstall PyTorch via Step 3 manual path |
| `ModuleNotFoundError: core` | Make sure you're in the project folder and `conda activate uav` |
| `No module named 'faiss'` | `pip install faiss-cpu` |
| First localize is slow | Normal — models load once; subsequent calls are fast |
| Low confidence on real images | Set the exact GPS box; lower `STRIDE`; ensure the image is roughly top-down |
| Out of memory | Lower the embedding batch in `core/map_index.py` (`batch=48` → 24) |
