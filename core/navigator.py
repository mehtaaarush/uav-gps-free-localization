"""
core/navigator.py  —  Stage A (simulated flight + landmark path confirmation)
===========================================================================
Two-rate navigation with landmark-based path confirmation:

  FAST loop  (every frame)   : optical-flow landmark tracking
  SLOW loop  (background)    : DINOv2/EfficientLoFTR absolute fix
  LANDMARK loop (every 10 fr): match frame against landmark DB →
                                confirms correct path or flags deviation

Run:
    python -m core.navigator --target data/query_samples/query_010.png
"""

import os, time, threading, argparse
import numpy as np
import cv2

from core import config as C
from core.localizer import Localizer
from core.tracker import LandmarkTracker
from core.guidance import Guidance


class Navigator:
    def __init__(self, gsd=0.3, fast_only=False, use_landmarks=True):
        self.loc = Localizer(retrieval=C.RETRIEVAL_BACKBONE,
                             matcher=C.MATCHER, top_k=C.TOP_K, do_refine=True)
        self.sat = self.loc.sat_bgr
        self.H, self.W = self.sat.shape[:2]
        self.tracker = LandmarkTracker()
        self.guide   = Guidance(metres_per_pixel=gsd,
                                arrive_radius_m=20.0, armed=False)
        self.fast_only = fast_only
        self.gsd       = gsd

        # landmark path confirmation (optional)
        self.lm_matcher = None
        if use_landmarks:
            db = C.LANDMARK_DIR / "landmark_db.pkl"
            if db.exists():
                from core.landmark_matcher import LandmarkMatcher
                self.lm_matcher = LandmarkMatcher(str(db))
                print(f"[Navigator] landmark confirmation: ON "
                      f"({len(self.lm_matcher.db)} landmarks)")
            else:
                print("[Navigator] landmark_db.pkl not found — "
                      "run landmark_detector first. Continuing without.")

        # shared state
        self._lock     = threading.Lock()
        self._abs_fix  = None
        self._fix_busy = False
        self.confirmed_landmarks = []   # most recent confirmed landmarks

    def locate_target(self, target_img_path):
        img = cv2.imread(target_img_path)
        if img is None:
            raise SystemExit(f"Cannot read target photo: {target_img_path}")
        r = self.loc.locate(img)
        print(f"[Navigator] target '{os.path.basename(target_img_path)}' "
              f"→ map ({r['pixel_x']},{r['pixel_y']}) conf={r['confidence']}")
        return (r["pixel_x"], r["pixel_y"])

    # ── slow absolute fix ──────────────────────────────────────────────────
    def _async_fix(self, frame):
        def worker(f):
            r = self.loc.locate(f)
            with self._lock:
                if r["confidence"] >= C.CONF_THRESHOLD:
                    self._abs_fix = (r["pixel_x"], r["pixel_y"],
                                     r["confidence"])
                self._fix_busy = False
        with self._lock:
            if self._fix_busy:
                return
            self._fix_busy = True
        threading.Thread(target=worker, args=(frame.copy(),),
                         daemon=True).start()

    # ── simulated camera ───────────────────────────────────────────────────
    def _sim_view(self, cx, cy, fov=320):
        half = fov // 2
        H, W = self.H, self.W
        x1 = int(np.clip(cx - half, 0, W - fov))
        y1 = int(np.clip(cy - half, 0, H - fov))
        return self.sat[y1:y1+fov, x1:x1+fov].copy()

    # ── main simulation ────────────────────────────────────────────────────
    def run_sim(self, target_xy, start_xy=None, speed_px=6.0,
                fix_every=20, lm_every=10,
                save_video=True, max_steps=400):
        tx, ty = target_xy
        if start_xy is None:
            start_xy = (int(self.W*0.10), int(self.H*0.85))
        cx, cy = float(start_xy[0]), float(start_xy[1])
        ex, ey = cx, cy

        self.tracker.init(self._sim_view(cx, cy))
        path_true, path_est = [(cx,cy)], [(ex,ey)]
        confirmed_log = []

        writer = None
        if save_video:
            out_path = str(C.RESULTS_DIR / "navigation_demo.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, 20.0,
                                       (self.W, self.H))

        print("\n[Navigator] flying  "
              f"start=({int(cx)},{int(cy)})  target=({tx},{ty})\n")
        fast_ms = []

        for step in range(max_steps):
            g = self.guide.to_target((ex, ey), (tx, ty))
            if g["arrived"]:
                print(f"\n[Navigator] ✓ ARRIVED (step {step}, "
                      f"{g['distance_m']:.1f}m)\n")
                break

            # move true position
            cmd = g["command"]
            cx  = float(np.clip(cx + cmd["vy_east"]*speed_px, 0, self.W-1))
            cy  = float(np.clip(cy - cmd["vx_north"]*speed_px, 0, self.H-1))

            # fast loop
            t0  = time.time()
            view = self._sim_view(cx, cy)
            dx, dy, n = self.tracker.update(view)
            ex  = float(np.clip(ex - dx, 0, self.W-1))
            ey  = float(np.clip(ey - dy, 0, self.H-1))
            fast_ms.append((time.time()-t0)*1000)

            # slow absolute fix
            if not self.fast_only and step % fix_every == 0:
                self._async_fix(view)
            with self._lock:
                if self._abs_fix is not None:
                    ex, ey, _ = self._abs_fix
                    self._abs_fix = None

            # landmark path confirmation
            confirmed_here = []
            if self.lm_matcher and step % lm_every == 0:
                confirmed_here = self.lm_matcher.match_frame(
                    view, ex, ey, radius_px=280)
                if confirmed_here:
                    ex, ey = self.lm_matcher.position_correction(
                        confirmed_here, ex, ey)
                    for c in confirmed_here:
                        msg = (f"  step {step:3d} ✔ LANDMARK {c['label']} "
                               f"({c['type']}) confirmed — "
                               f"{c['inliers']} inliers, "
                               f"{c['distance_m']}m from est. position")
                        print(msg)
                        confirmed_log.append(
                            {"step": step, **c, "est": (ex,ey)})

            path_true.append((cx, cy))
            path_est.append((ex, ey))
            self.confirmed_landmarks = confirmed_log

            if step % 5 == 0:
                lm_tag = (f" | LM:{confirmed_here[0]['label']}"
                          if confirmed_here else "")
                print(f"  step {step:3d} | est=({int(ex)},{int(ey)}) "
                      f"| {g['distance_m']:6.1f}m {g['compass']:2s} "
                      f"| {g['direction']:16s} | trk {n:3d} pts "
                      f"| {fast_ms[-1]:.1f}ms{lm_tag}")

            if writer is not None:
                writer.write(self._draw(path_true, path_est,
                                        (tx,ty), g, confirmed_log))

        if writer is not None:
            writer.release()
            print(f"[Navigator] video → {C.RESULTS_DIR/'navigation_demo.mp4'}")

        final = self._draw(path_true, path_est, (tx,ty), g, confirmed_log)
        cv2.imwrite(str(C.RESULTS_DIR/"navigation_path.png"), final)
        print(f"[Navigator] path image → {C.RESULTS_DIR/'navigation_path.png'}")

        if fast_ms:
            print(f"[Navigator] fast-loop: mean {np.mean(fast_ms):.1f}ms "
                  f"→ ~{1000/np.mean(fast_ms):.0f} FPS")
        if confirmed_log:
            print(f"[Navigator] {len(confirmed_log)} landmark "
                  f"confirmations during flight")

    # ── visualisation ──────────────────────────────────────────────────────
    def _draw(self, path_true, path_est, target, g, confirmed_log):
        vis = self.sat.copy()

        # draw landmark positions from database (faint)
        if self.lm_matcher:
            for lm in self.lm_matcher.db:
                from core.landmark_detector import TYPE_COLOUR, TYPE_SYMBOL
                col = TYPE_COLOUR[lm["type"]]
                sym = TYPE_SYMBOL[lm["type"]]
                cv2.drawMarker(vis, lm["pixel"], col, sym, 14, 1)
                cv2.putText(vis, lm["label"],
                            (lm["pixel"][0]+8, lm["pixel"][1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

        # draw confirmed landmarks (bright, larger)
        confirmed_pixels = {c["label"] for c in confirmed_log}
        if self.lm_matcher:
            for lm in self.lm_matcher.db:
                if lm["label"] in confirmed_pixels:
                    from core.landmark_detector import TYPE_COLOUR
                    col = TYPE_COLOUR[lm["type"]]
                    cv2.circle(vis, lm["pixel"], 18, col, 2)
                    cv2.putText(vis, "✔"+lm["label"],
                                (lm["pixel"][0]+12, lm["pixel"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # paths
        for i in range(1, len(path_true)):
            cv2.line(vis, tuple(map(int,path_true[i-1])),
                     tuple(map(int,path_true[i])), (0,200,0), 2)
        for i in range(1, len(path_est)):
            cv2.line(vis, tuple(map(int,path_est[i-1])),
                     tuple(map(int,path_est[i])), (255,200,0), 2)

        # target
        tx, ty = map(int, target)
        cv2.drawMarker(vis, (tx,ty), (0,0,255), cv2.MARKER_STAR, 50, 4)
        cv2.circle(vis, (tx,ty),
                   int(self.guide.arrive_radius/self.guide.mpp), (0,0,255), 2)

        # drone
        if path_est:
            ex, ey = map(int, path_est[-1])
            cv2.circle(vis, (ex,ey), 12, (255,200,0), -1)

        # HUD
        lm_txt = (f"  LM confirmed: {len(confirmed_log)}"
                  if self.lm_matcher else "  landmarks: OFF")
        hud = (f"{g['distance_m']:.0f}m {g['compass']}  "
               f"{g['direction']}{lm_txt}")
        cv2.rectangle(vis, (0,0), (self.W, 52), (0,0,0), -1)
        cv2.putText(vis, hud, (10,34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
        return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",    required=True)
    ap.add_argument("--gsd",       type=float, default=0.3)
    ap.add_argument("--fast_only", action="store_true")
    ap.add_argument("--speed",     type=float, default=6.0)
    ap.add_argument("--no_landmarks", action="store_true")
    a = ap.parse_args()

    nav = Navigator(gsd=a.gsd, fast_only=a.fast_only,
                    use_landmarks=not a.no_landmarks)
    txy = nav.locate_target(a.target)
    nav.run_sim(txy, speed_px=a.speed)


if __name__ == "__main__":
    main()
