#!/usr/bin/env bash
# run_all.sh — one-command setup for the IITR UAV localizer
# Usage:  bash run_all.sh
set -e

echo "════════════════════════════════════════════════"
echo "  IITR GPS-Free UAV Localizer — full setup"
echo "════════════════════════════════════════════════"

echo "[1/4] Installing dependencies…"
pip install -r requirements.txt

echo "[2/4] Generating simulated UAV query frames…"
python -m core.simulate --n 20 --size 300
cp data/query_samples/query_*.png web/static/samples/ 2>/dev/null || true

echo "[3/4] Building DINOv2 + FAISS map index (multi-scale, run once)…"
python -m core.map_index --scales 256,384 --stride 48 --backbone dinov2

echo "[4/4] Evaluating accuracy…"
python -m core.evaluate --gsd 0.3 || true

echo ""
echo "✓ Setup complete. Launch the live demo with:"
echo "      python -m web.app"
echo "  then open http://localhost:5000"
