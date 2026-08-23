# NAVIGATION (Stage A) — Fly to a target by tracking landmarks

This adds **visual navigation** on top of localization: give it a photo of a
destination landmark, and it guides a (simulated) drone there while tracking
landmarks in real time — fast enough for a real UAV.

## The two-rate design (why it's fast)

A real UAV flies fast, so you can't run a heavy neural net every frame. This uses
two loops at different rates:

| Loop | Rate | What it does | Cost |
|------|------|--------------|------|
| **Fast** | every frame (~200 FPS) | Lucas-Kanade optical flow tracks landmarks between frames → position update | 2–5 ms (CPU) |
| **Slow** | ~1 Hz, background thread | DINOv2 + EfficientLoFTR absolute fix vs satellite map → corrects drift | ~100 ms (GPU) |

The fast loop keeps the drone locked on target; the slow loop occasionally
corrects accumulated drift without ever blocking the fast loop.

## Run the navigation demo

```bash
python -m core.navigator --target data/query_samples/query_010.png
```

What happens:
1. The target photo is localized on the map → destination point
2. A simulated drone starts from a far corner
3. It flies toward the target, tracking landmarks each frame
4. Console prints live guidance: distance, compass bearing, plain direction
5. Outputs `results/navigation_path.png` and `results/navigation_demo.mp4`

Options:
- `--gsd 0.3`        metres per pixel of your map (for real distances)
- `--speed 6`        simulated drone speed (pixels/step)
- `--fast_only`      disable the slow loop (shows pure tracking + drift)

## What you'll see

- **Green line** — true drone path
- **Cyan line** — estimated path (what the drone thinks; tracks the green closely)
- **Red star + circle** — target and arrival radius
- **HUD bar** — live distance, bearing, and steering direction

## The output commands

Each step produces a guidance dict with:
- `bearing_deg` + `compass` (e.g. 47°, "NE")
- `distance_m` to target
- `direction` — plain English ("head forward-right")
- `command` — a MAVLink-style velocity setpoint

**SAFETY:** the `command` is only ever printed/logged. The real
flight-controller link is disabled (`armed=False` in `core/guidance.py`) and must
stay that way until you've tested in a simulator (SITL). See the staging plan below.

## Staging plan toward real flight

- **Stage A (this):** simulated flight over the map — proves the algorithm, fully demoable
- **Stage B:** replay recorded drone video through the same pipeline
- **Stage C:** live flight — connect a Pixhawk (ArduPilot/PX4) via MAVLink on a
  companion computer (Raspberry Pi/Jetson), add a downward LiDAR for altitude,
  and only then implement `Guidance.send_command()` for real

For a final-year submission, demo Stage A and describe B and C as the deployment path.
