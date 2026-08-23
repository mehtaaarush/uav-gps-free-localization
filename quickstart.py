"""
quickstart.py  —  run the WHOLE thing with one command
=======================================================
Runs from anywhere; all paths are anchored to the project root.

USAGE
-----
  # First time (builds the map index, ~1 min on GPU):
  python quickstart.py --setup

  # Localize a single image and print its exact location:
  python quickstart.py --image data/query_samples/query_003.png

  # Localize + save an annotated map showing where the drone is:
  python quickstart.py --image path/to/drone.jpg --save

  # Do everything in one go (setup if needed, then localize):
  python quickstart.py --image data/query_samples/query_003.png --setup
"""

import argparse
import time
import os
import cv2

from core import config as C


def do_setup():
    """Generate samples + build the DINOv2/FAISS index + detect landmarks."""
    from core.simulate import generate
    from core.map_index import build
    from core.landmark_detector import detect_all
    import shutil, glob

    print("\n=== [setup 1/3] generating sample drone frames ===")
    generate(str(C.DEFAULT_MAP), str(C.QUERY_DIR), n=20, size=180)
    for f in glob.glob(str(C.QUERY_DIR / "query_*.png")):
        shutil.copy(f, C.WEB_SAMPLES)

    print("\n=== [setup 2/3] building map index (multi-scale) ===")
    build(str(C.DEFAULT_MAP), str(C.MODELS_DIR),
          scales=C.TILE_SCALES, stride=C.STRIDE, prefer=C.RETRIEVAL_BACKBONE)

    print("\n=== [setup 3/3] detecting landmarks for path confirmation ===")
    db = detect_all(str(C.DEFAULT_MAP), str(C.LANDMARK_DIR))
    print(f"  {len(db)} landmarks ready for navigation confirmation")

    print("\n✓ setup complete\n")


def index_exists():
    return (C.MODELS_DIR / "map_meta.pkl").exists()


def localize(image_path, save=False):
    from core.localizer import Localizer

    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    print("\nLoading localization engine (DINOv2 + EfficientLoFTR)…")
    loc = Localizer(retrieval=C.RETRIEVAL_BACKBONE,
                    matcher=C.MATCHER, top_k=C.TOP_K, do_refine=True)

    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    t0 = time.time()
    r = loc.locate(frame)
    lat, lon = loc.pixel_to_gps(r["pixel_x"], r["pixel_y"], **C.GPS_BBOX)

    # confidence verdict
    conf = r["confidence"]
    verdict = ("HIGH — reliable fix"      if conf >= 0.6 else
               "MEDIUM — likely correct"  if conf >= 0.4 else
               "LOW — treat with caution")

    print("\n" + "═"*56)
    print("            DRONE LOCATION RESULT")
    print("═"*56)
    print(f"  Image            : {os.path.basename(image_path)}")
    print(f"  Map position (px): x = {r['pixel_x']}   y = {r['pixel_y']}")
    print(f"  GPS coordinates  : {lat}, {lon}")
    print(f"  Confidence       : {conf:.3f}   [{verdict}]")
    print(f"  Match inliers    : {r['inliers']}")
    print(f"  Processing time  : {r['timing']['total_ms']:.0f} ms")
    print(f"      · retrieval  : {r['timing']['retrieval_ms']:.0f} ms")
    print(f"      · FAISS srch : {r['timing']['search_ms']:.1f} ms")
    print(f"      · matcher    : {r['timing']['match_ms']:.0f} ms")
    print("═"*56)
    print(f"  Google Maps link : https://maps.google.com/?q={lat},{lon}")
    print("═"*56 + "\n")

    if save:
        vis = loc.render(r, frame)
        out = C.RESULTS_DIR / ("located_" + os.path.basename(image_path))
        cv2.imwrite(str(out), vis)
        print(f"Annotated map saved → {out}\n")


def main():
    ap = argparse.ArgumentParser(description="GPS-free UAV localizer — quickstart")
    ap.add_argument("--setup",  action="store_true",
                    help="generate samples + build the map index (run once)")
    ap.add_argument("--image",  default=None,
                    help="path to a drone/UAV image to localize")
    ap.add_argument("--save",   action="store_true",
                    help="save an annotated map showing the location")
    args = ap.parse_args()

    if args.setup or not index_exists():
        if not index_exists():
            print("(no map index found — running setup first)")
        do_setup()

    if args.image:
        localize(args.image, save=args.save)
    else:
        print("Setup done. Now localize an image, e.g.:")
        print("  python quickstart.py --image data/query_samples/query_003.png --save")


if __name__ == "__main__":
    main()
