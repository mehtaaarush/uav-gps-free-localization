"""Export the retrieval stage of Vision-Nav for the browser demo in docs/.

1. DINOv2 ViT-S/14 (the same backbone as core/models.py) -> ONNX at 154x154, int8-quantised weights.
2. Re-tile the satellite map exactly like core/map_index.py (scales 160/224/288, stride 32) and embed
   every tile with the exported model, so the browser and the index use one model.
3. Evaluate retrieval-only localisation on the 20 simulated drone frames with ground truth.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import cv2
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config as C
from core.map_index import tile_map

DOCS = ROOT / "docs"
SIZE = 154                       # _round14(160), the size map_index embeds tiles at
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

class Embed(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.nn.functional.normalize(self.m(x), dim=1)

def prep(img):
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)

def main():
    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub), "dinov2_vits14", source="local", verbose=False).eval()
    emb = Embed(model).eval()
    fp32 = DOCS / "model" / "dinov2_vits14_fp32.onnx"
    torch.onnx.export(emb, torch.randn(1, 3, SIZE, SIZE), str(fp32), input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    q = DOCS / "model" / "dinov2_vits14_int8.onnx"
    quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    print("onnx sizes MB", round(fp32.stat().st_size / 1e6, 1), round(q.stat().st_size / 1e6, 1))

    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
    tiles, positions, scales, (W, H) = tile_map(str(C.DEFAULT_MAP), C.TILE_SCALES, C.STRIDE)
    X = np.stack([prep(t) for t in tiles])
    t0 = time.time()
    with torch.no_grad():
        ref = emb(torch.from_numpy(X[:64])).numpy()
    f32 = s32.run(None, {"image": X[:64]})[0]
    feats = np.vstack([s8.run(None, {"image": X[i:i + 64]})[0] for i in range(0, len(X), 64)])
    print("tiles", len(tiles), "embed s", round(time.time() - t0, 1))
    print("cos torch vs onnx fp32", float((ref * f32).sum(1).min()), "vs int8", float((ref * feats[:64]).sum(1).min()))

    # tiles file: header json + int8 embeddings with a per-row scale
    scale = np.abs(feats).max(1, keepdims=True) / 127.0
    qf = np.round(feats / scale).astype(np.int8)
    (DOCS / "data" / "tiles.bin").write_bytes(qf.tobytes())
    meta = {"count": len(tiles), "dim": int(feats.shape[1]), "size": SIZE, "map": [W, H],
            "positions": [list(map(int, p)) for p in positions], "scales": list(map(int, scales)),
            "rowScale": [round(float(s), 7) for s in scale[:, 0]], "gps": C.GPS_BBOX}
    (DOCS / "data" / "tiles.json").write_text(json.dumps(meta, separators=(",", ":")))

    # evaluate on the simulated drone frames: DINOv2 top-8, then ORB + RANSAC re-ranking (core/models.py fallback)
    gt = [l.split(",") for l in (C.QUERY_DIR / "ground_truth.csv").read_text().splitlines()[1:] if l.strip()]
    deq = qf.astype(np.float32) * scale
    deq /= np.linalg.norm(deq, axis=1, keepdims=True)
    sat = cv2.imread(str(C.DEFAULT_MAP))
    orb = cv2.ORB_create(nfeatures=2000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bb = C.GPS_BBOX
    def to_gps(x, y): return (bb["tl_lat"] + (bb["br_lat"] - bb["tl_lat"]) * y / H, bb["tl_lon"] + (bb["br_lon"] - bb["tl_lon"]) * x / W)
    def metres(a, b):
        R = 6371000; p1, p2 = math.radians(a[0]), math.radians(b[0]); dp = p2 - p1; dl = math.radians(b[1] - a[1])
        return 2 * R * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))
    def refine(frame, cx, cy, tile):
        half = tile // 2; x1, x2 = max(0, cx - half), min(W, cx + half); y1, y2 = max(0, cy - half), min(H, cy + half)
        crop = sat[y1:y2, x1:x2]; uav = cv2.resize(frame, (crop.shape[1], crop.shape[0]))
        k0, d0 = orb.detectAndCompute(uav, None); k1, d1 = orb.detectAndCompute(crop, None)
        if d0 is None or d1 is None or len(d0) < 4 or len(d1) < 4: return cx, cy, 0
        m = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:200]
        if len(m) < 4: return cx, cy, len(m)
        p0 = np.float32([k0[a.queryIdx].pt for a in m]).reshape(-1, 1, 2); p1 = np.float32([k1[a.trainIdx].pt for a in m]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
        if Hm is None: return cx, cy, len(m)
        pr = cv2.perspectiveTransform(np.array([[[uav.shape[1] / 2, uav.shape[0] / 2]]], np.float32), Hm)[0][0]
        rx, ry = int(x1 + pr[0]), int(y1 + pr[1])
        if np.hypot(rx - cx, ry - cy) > tile: return cx, cy, int(mask.sum())
        return rx, ry, int(mask.sum())
    rows = []
    for name, tx, ty in gt:
        frame = cv2.imread(str(C.QUERY_DIR / name)); tx, ty = int(tx), int(ty)
        e = s8.run(None, {"image": prep(Image.open(C.QUERY_DIR / name))[None]})[0][0]
        idx = np.argsort(-(deq @ e))[:C.TOP_K]
        r0 = metres(to_gps(*positions[idx[0]]), to_gps(tx, ty))
        best = (-1, *positions[idx[0]])
        for i in idx:
            rx, ry, inl = refine(frame, *positions[i], scales[i])
            if inl > best[0]: best = (inl, rx, ry)
        rows.append({"file": name, "truth": [tx, ty], "retrieval_m": round(r0, 1), "refined_m": round(metres(to_gps(best[1], best[2]), to_gps(tx, ty)), 1), "inliers": best[0]})
    r = np.array([x["retrieval_m"] for x in rows]); f = np.array([x["refined_m"] for x in rows])
    summary = {"frames": len(rows), "retrieval_median_m": round(float(np.median(r)), 1), "refined_median_m": round(float(np.median(f)), 1),
               "refined_under_25m": int((f < 25).sum()), "retrieval_under_25m": int((r < 25).sum()), "top_k": C.TOP_K}
    print(summary)
    (DOCS / "data" / "eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, separators=(",", ":")))
    fp32.unlink()

if __name__ == "__main__":
    main()
