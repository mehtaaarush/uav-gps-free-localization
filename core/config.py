"""
core/config.py — single source of truth for all paths and settings.
All paths are absolute (anchored to project root) so cwd never matters.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
SAT_MAP_DIR   = DATA_DIR / "satellite_map"
QUERY_DIR     = DATA_DIR / "query_samples"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"
WEB_SAMPLES   = PROJECT_ROOT / "web" / "static" / "samples"
LANDMARK_DIR  = PROJECT_ROOT / "data" / "landmarks"   # ← new

DEFAULT_MAP   = SAT_MAP_DIR / "iitr_campus.png"

# ── GPS bounding box (new map: 1498x704) ──────────────────────────────────
# Full campus coverage — update corners after checking in Google Earth.
GPS_BBOX = {
    "tl_lat": 29.8675, "tl_lon": 77.8920,
    "br_lat": 29.8590, "br_lon": 77.9070,
}

# ── model settings ────────────────────────────────────────────────────────
RETRIEVAL_BACKBONE = "dinov2"
MATCHER            = "efficientloftr"
TILE_SCALES        = (160, 224, 288)  # 3 scales → matches more zoom levels
STRIDE             = 32                # finer stride → denser coverage, better accuracy
TOP_K              = 8                 # more candidates re-ranked by the matcher
CONF_THRESHOLD     = 0.45         # reject localisations below this

# ── landmark settings ─────────────────────────────────────────────────────
LANDMARK_PATCH_SIZE = 160         # px crop around each landmark centre
LANDMARK_MIN_DIST   = 70          # minimum pixel distance between landmarks
LANDMARK_MATCH_CONF = 0.35        # min matcher confidence to confirm a landmark
MAX_LANDMARKS       = 40          # only the most distinctive ones

for d in (MODELS_DIR, RESULTS_DIR, QUERY_DIR, WEB_SAMPLES, LANDMARK_DIR):
    d.mkdir(parents=True, exist_ok=True)
