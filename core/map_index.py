"""
core/map_index.py
=================
OFFLINE step (run once). Tiles the satellite map, embeds every tile with the
DINOv2 backbone, and builds a FAISS index for sub-millisecond retrieval.

Run:
    python -m core.map_index --map data/satellite_map/iitr_campus.png
"""

import os
import time
import pickle
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from core.models import RetrievalBackbone, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from core import config as C

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


# ── DINOv2 needs input sizes that are multiples of 14 ───────────────────────
def _round14(n):
    return max(14, (n // 14) * 14)


def make_transform(kind: str, size: int):
    if kind == "dinov2":
        size = _round14(size)
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]), size


def tile_map(map_path, scales=(256,), stride=64):
    """
    Tile the map at one or more scales. Multi-scale tiling lets a query taken
    at any altitude/zoom match the tile whose ground footprint is closest.

    Returns
    -------
    tiles      : list[PIL.Image]
    positions  : list[(cx, cy)]      pixel centre of each tile
    scales_out : list[int]           the tile size each tile was cut at
    map_size   : (W, H)
    """
    sat = Image.open(map_path).convert("RGB")
    W, H = sat.size
    tiles, positions, scales_out = [], [], []
    for tile in scales:
        if tile > min(W, H):
            continue
        step = max(8, int(stride * tile / scales[0]))   # scale the stride too
        for y in range(0, H - tile + 1, step):
            for x in range(0, W - tile + 1, step):
                tiles.append(sat.crop((x, y, x + tile, y + tile)))
                positions.append((x + tile // 2, y + tile // 2))
                scales_out.append(tile)
    return tiles, positions, scales_out, (W, H)


@torch.no_grad()
def embed_tiles(tiles, backbone, transform, batch=48):
    feats = []
    for i in range(0, len(tiles), batch):
        batch_t = torch.stack([transform(t) for t in tiles[i:i+batch]])
        feats.append(backbone(batch_t).numpy())
        if (i // batch) % 10 == 0:
            print(f"   embedded {min(i+batch, len(tiles))}/{len(tiles)} tiles")
    return np.vstack(feats).astype("float32")


def build_index(feats):
    if HAVE_FAISS:
        idx = faiss.IndexFlatIP(feats.shape[1])
        idx.add(feats)
        return ("faiss", idx)
    # numpy fallback — still fast for <100k tiles
    return ("numpy", feats)


def build(map_path, out_dir="models", scales=(256, 384), stride=48, prefer="dinov2"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    backbone = RetrievalBackbone(prefer=prefer)
    # all tiles resized to a common embedding input (based on the smallest scale)
    transform, eff_size = make_transform(backbone.kind, scales[0])

    print(f"[map_index] Tiling map (scales={scales}, stride={stride})…")
    tiles, positions, tile_scales, map_size = tile_map(map_path, scales, stride)
    print(f"[map_index] {len(tiles)} tiles  |  map {map_size[0]}x{map_size[1]}px")

    print("[map_index] Embedding tiles with retrieval backbone…")
    feats = embed_tiles(tiles, backbone, transform)

    backend, index = build_index(feats)

    if backend == "faiss":
        faiss.write_index(index, os.path.join(out_dir, "map.faiss"))
        feats_blob = None
    else:
        feats_blob = feats

    meta = {
        "positions":    positions,
        "tile_scales":  tile_scales,      # per-tile crop size for refinement
        "map_size":     map_size,
        "scales":       list(scales),
        "tile":         scales[0],        # default/back-compat
        "stride":       stride,
        "map_path":     map_path,
        "backbone":     backbone.kind,
        "dim":          backbone.dim,
        "backend":      backend,
        "feats":        feats_blob,
    }
    with open(os.path.join(out_dir, "map_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"[map_index] ✓ done in {time.time()-t0:.1f}s  "
          f"(backend={backend}, backbone={backbone.kind})")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map",    default=str(C.DEFAULT_MAP))
    ap.add_argument("--out",    default=str(C.MODELS_DIR))
    ap.add_argument("--scales", default=",".join(str(s) for s in C.TILE_SCALES),
                    help="comma-separated tile sizes, e.g. 256,384")
    ap.add_argument("--stride", type=int, default=C.STRIDE)
    ap.add_argument("--backbone", default=C.RETRIEVAL_BACKBONE,
                    choices=["dinov2", "efficientnet"])
    a = ap.parse_args()
    scales = tuple(int(s) for s in a.scales.split(","))
    build(a.map, a.out, scales, a.stride, a.backbone)
