"""
web/app.py
==========
Flask web server for the live demo. Loads the Localizer once at startup,
then serves an instant-localization endpoint.

Run:
    python -m web.app
Then open http://localhost:5000
"""

import os
import io
import base64
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_file

from core.localizer import Localizer
from core import config as C

# absolute template/static folders so Flask runs from anywhere
app = Flask(__name__,
            template_folder=str(C.PROJECT_ROOT / "web" / "templates"),
            static_folder=str(C.PROJECT_ROOT / "web" / "static"))

# ── load the engine once ────────────────────────────────────────────────────
print("Loading localization engine…")
LOC = Localizer(retrieval=C.RETRIEVAL_BACKBONE, matcher=C.MATCHER,
                top_k=C.TOP_K, do_refine=True)

# GPS bounding box of the IIT Roorkee satellite map (from config)
GPS_BBOX = dict(C.GPS_BBOX)


def bgr_to_dataurl(img_bgr, max_w=1400):
    h, w = img_bgr.shape[:2]
    if w > max_w:
        s = max_w / w
        img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("ascii")
    return "data:image/jpeg;base64," + b64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/map")
def get_map():
    """Serve the base satellite map as a data URL + its size."""
    return jsonify({
        "map": bgr_to_dataurl(LOC.sat_bgr),
        "width":  LOC.map_size[0],
        "height": LOC.map_size[1],
        "tiles":  len(LOC.positions),
        "backbone": LOC.backbone.kind,
        "matcher":  LOC.matcher.kind if LOC.matcher else "none",
    })


@app.route("/locate", methods=["POST"])
def locate():
    """Receive an uploaded image, return position + annotated map instantly."""
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    data = np.frombuffer(request.files["image"].read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    result = LOC.locate(frame)
    lat, lon = LOC.pixel_to_gps(result["pixel_x"], result["pixel_y"],
                                **GPS_BBOX)
    annotated = LOC.render(result, frame)

    return jsonify({
        "pixel_x": result["pixel_x"],
        "pixel_y": result["pixel_y"],
        "lat": lat, "lon": lon,
        "confidence": result["confidence"],
        "inliers": result["inliers"],
        "timing": result["timing"],
        "top_k": result["top_k"],
        "query_preview": bgr_to_dataurl(frame, max_w=320),
        "annotated_map": bgr_to_dataurl(annotated),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
