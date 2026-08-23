"""
core/navigator_dynamic.py
=========================
Integrated Localization + Landmark-Based Navigation (no fixed path).

This replaces the old "fly a straight precomputed line" navigator with a
system where every navigation decision is made in real time from the map:

  STEP 0  TARGET LOCALIZATION
          Localize the user-given target photo on the map (DINOv2 +
          EfficientLoFTR) → target pixel + GPS coordinates.

  STEP 1  COLD-START DRONE LOCALIZATION
          Localize the drone's first frame on the map → initial position.
          (No assumed start — the drone discovers where it is.)

  Then every frame:

  STEP 2  CONTINUOUS LOCALIZATION  (PoseEstimator fuses 3 sources)
            • optical-flow scene tracking (every frame)
            • DINOv2 absolute fix       (background, periodic)
            • landmark recognition      (every N frames)

  STEP 3  DYNAMIC ROUTE PLANNING  (DynamicNavigator + LandmarkGraph)
            • A* through the landmark graph from CURRENT position to target
            • re-plans on: waypoint reached / confidence change / refresh
            • the route emerges from landmarks, never precomputed

  STEP 4  GOAL-DIRECTED FLIGHT
            • steer toward the next way-landmark (pure pursuit between nodes)
            • verify progress via the fused position
            • stop when within the target error threshold

Run:
    python -m core.navigator_dynamic --target data/query_samples/query_005.png
"""

import os, time, threading, argparse
import numpy as np
import cv2

from core import config as C
from core.localizer import Localizer
from core.tracker import LandmarkTracker
from core.guidance import Guidance
from core.pose_estimator import PoseEstimator
from core.route_planner import LandmarkGraph, DynamicNavigator


class DynamicMapNavigator:
    def __init__(self, gsd=0.3):
        self.loc = Localizer(retrieval=C.RETRIEVAL_BACKBONE,
                             matcher=C.MATCHER, top_k=C.TOP_K, do_refine=True)
        self.sat = self.loc.sat_bgr
        self.H, self.W = self.sat.shape[:2]
        self.gsd = gsd
        self.tracker = LandmarkTracker()
        self.guide   = Guidance(metres_per_pixel=gsd,
                                arrive_radius_m=20.0, armed=False)

        # landmark database + matcher
        from core.landmark_matcher import LandmarkMatcher
        db_path = C.LANDMARK_DIR / "landmark_db.pkl"
        if not db_path.exists():
            raise SystemExit("Run  python -m core.landmark_detector  first.")
        self.lm_matcher = LandmarkMatcher(str(db_path))
        self.landmarks  = self.lm_matcher.db

        # build the navigation graph from the landmarks
        self.graph = LandmarkGraph(self.landmarks, self.W, self.H,
                                   connect_radius=240)
        print(f"[DynamicNav] graph: {len(self.graph.nodes)} landmark nodes, "
              f"{sum(len(v) for v in self.graph.adj.values())//2} edges")

        # background fix state
        self._lock = threading.Lock()
        self._fix  = None
        self._busy = False

    # ── STEP 0: target localization ────────────────────────────────────────
    def localize_target(self, target_img_path):
        img = cv2.imread(target_img_path)
        if img is None:
            raise SystemExit(f"Cannot read target: {target_img_path}")
        r = self.loc.locate(img)
        lat, lon = self.loc.pixel_to_gps(r["pixel_x"], r["pixel_y"], **C.GPS_BBOX)
        print(f"\n[STEP 0] TARGET LOCALIZED")
        print(f"  map pixel : ({r['pixel_x']}, {r['pixel_y']})")
        print(f"  GPS       : {lat}, {lon}")
        print(f"  confidence: {r['confidence']}")
        return (r["pixel_x"], r["pixel_y"]), (lat, lon)

    # ── background absolute fix ─────────────────────────────────────────────
    def _async_fix(self, frame, pose_at_dispatch):
        def worker(f, p0):
            r = self.loc.locate(f)
            with self._lock:
                # store fix + the pose position at the moment it was dispatched
                self._fix = (r["pixel_x"], r["pixel_y"], r["confidence"], p0)
                self._busy = False
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=worker,
                         args=(frame.copy(), pose_at_dispatch),
                         daemon=True).start()

    def _sim_view(self, cx, cy, fov=300):
        half = fov//2
        x1 = int(np.clip(cx-half, 0, self.W-fov))
        y1 = int(np.clip(cy-half, 0, self.H-fov))
        return self.sat[y1:y1+fov, x1:x1+fov].copy()

    # ── main integrated loop ────────────────────────────────────────────────
    def run(self, target_xy, target_gps, true_start=None,
            speed_px=9.0, fix_every=18, lm_every=8,
            max_steps=400, save_video=True):

        # STEP 1: cold-start localization (drone discovers where it is)
        if true_start is None:
            true_start = (int(self.W*0.10), int(self.H*0.85))
        tcx, tcy = float(true_start[0]), float(true_start[1])
        first_view = self._sim_view(tcx, tcy)
        r0 = self.loc.locate(first_view)
        print(f"\n[STEP 1] COLD-START DRONE LOCALIZATION")
        print(f"  drone localized to ({r0['pixel_x']},{r0['pixel_y']}) "
              f"conf={r0['confidence']}")

        # fused pose estimator
        pose = PoseEstimator(r0["pixel_x"], r0["pixel_y"],
                             init_conf=max(0.4, r0["confidence"]),
                             map_w=self.W, map_h=self.H)

        # dynamic planner over the landmark graph
        dynnav = DynamicNavigator(self.graph, target_xy,
                                  reach_radius=70, replan_interval=22)

        self.tracker.init(first_view)
        true_path, est_path, replans = [(tcx,tcy)], [pose.get_pos()], []
        confirmed_all = []

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(C.RESULTS_DIR/"dynamic_nav.mp4"),
                                     fourcc, 20.0, (self.W, self.H))

        print(f"\n[STEP 2-4] DYNAMIC NAVIGATION  (re-planning enabled)\n")
        fast_ms = []
        arrival_thresh = self.guide.arrive_radius / self.gsd

        for step in range(max_steps):
            cur = pose.get_pos()

            # arrival check (goal-directed termination)
            if dynnav.reached_target(cur, arrival_thresh):
                print(f"\n[STEP 4] ✓ TARGET REACHED at step {step} "
                      f"(within {self.guide.arrive_radius:.0f} m)\n")
                break

            # STEP 3: dynamic route planning — next way-landmark
            waypoint, replan_reason = dynnav.step(cur, pose.conf)
            if replan_reason:
                replans.append((step, replan_reason, cur))
                print(f"  step {step:3d}  RE-PLAN ({replan_reason}) → "
                      f"route has {len(dynnav.route)} nodes, "
                      f"heading to ({int(waypoint[0])},{int(waypoint[1])})")

            # STEP 4: steer toward the current waypoint (not the final target)
            g = self.guide.to_target(cur, waypoint)
            cmd = g["command"]
            tcx = float(np.clip(tcx + cmd["vy_east"]*speed_px, 0, self.W-1))
            tcy = float(np.clip(tcy - cmd["vx_north"]*speed_px, 0, self.H-1))

            # STEP 2a: optical-flow continuous localization
            t0 = time.time()
            view = self._sim_view(tcx, tcy)
            dx, dy, ntrk = self.tracker.update(view)
            pose.update_flow(-dx, -dy)
            fast_ms.append((time.time()-t0)*1000)

            # STEP 2b: periodic DINOv2 absolute fix (background) → drift reset
            if step % fix_every == 0:
                self._async_fix(view, pose.get_pos())
            with self._lock:
                if self._fix is not None:
                    fx, fy, fc, p0 = self._fix
                    # motion the drone made while the fix was computing
                    cur_x, cur_y = pose.get_pos()
                    motion_since = (cur_x - p0[0], cur_y - p0[1])
                    # current heading (unit vector toward the active waypoint)
                    hdg = None
                    wpx, wpy = waypoint
                    hvx, hvy = wpx - cur_x, wpy - cur_y
                    hn = float(np.hypot(hvx, hvy))
                    if hn > 1e-3:
                        hdg = (hvx/hn, hvy/hn)
                    pose.update_fix(fx, fy, fc, min_conf=C.CONF_THRESHOLD,
                                    motion_since=motion_since, heading=hdg)
                    self._fix = None

            # STEP 2c: landmark recognition correction
            confirmed = []
            if step % lm_every == 0:
                cx, cy = pose.get_pos()
                confirmed = self.lm_matcher.match_frame(view, cx, cy,
                                                        radius_px=300)
                for c in confirmed:
                    pose.update_landmark(c["pixel"][0], c["pixel"][1],
                                         c["inliers"])
                    confirmed_all.append({"step": step, **c})
                    print(f"  step {step:3d}  ✔ landmark {c['label']} "
                          f"({c['type']}) — {c['inliers']} inliers "
                          f"→ position corrected")

            true_path.append((tcx, tcy))
            est_path.append(pose.get_pos())

            if step % 5 == 0:
                gt = self.guide.to_target(pose.get_pos(), target_xy)
                print(f"  step {step:3d} | est=({int(pose.x)},{int(pose.y)}) "
                      f"conf={pose.conf:.2f} | {gt['distance_m']:6.1f}m to target "
                      f"| trk {ntrk:3d} | {fast_ms[-1]:.1f}ms")

            if writer is not None:
                writer.write(self._draw(true_path, est_path, target_xy,
                                        dynnav, pose, confirmed_all))

        if writer is not None:
            writer.release()
            print(f"[DynamicNav] video → {C.RESULTS_DIR/'dynamic_nav.mp4'}")

        final = self._draw(true_path, est_path, target_xy, dynnav, pose,
                           confirmed_all)
        cv2.imwrite(str(C.RESULTS_DIR/"dynamic_nav_path.png"), final)
        print(f"[DynamicNav] path image → {C.RESULTS_DIR/'dynamic_nav_path.png'}")
        print(f"\n  SUMMARY")
        print(f"  re-plans          : {dynnav.replan_count}")
        print(f"  landmark confirms : {len(confirmed_all)}")
        if fast_ms:
            print(f"  fast-loop latency : {np.mean(fast_ms):.1f}ms "
                  f"(~{1000/np.mean(fast_ms):.0f} FPS)")

    # ── visualization ────────────────────────────────────────────────────────
    def _draw(self, true_path, est_path, target, dynnav, pose, confirmed_all):
        from core.landmark_detector import TYPE_COLOUR, TYPE_SYMBOL
        vis = self.sat.copy()

        # graph edges (faint) — shows the navigation network
        for i in self.graph.adj:
            for j, _ in self.graph.adj[i]:
                if j > i:
                    cv2.line(vis, self.graph.nodes[i], self.graph.nodes[j],
                             (90, 90, 90), 1)

        # landmarks
        for lm in self.landmarks:
            col = TYPE_COLOUR[lm["type"]]; sym = TYPE_SYMBOL[lm["type"]]
            cv2.drawMarker(vis, lm["pixel"], col, sym, 13, 1)

        # confirmed landmarks (bright)
        confirmed_labels = {c["label"] for c in confirmed_all}
        for lm in self.landmarks:
            if lm["label"] in confirmed_labels:
                col = TYPE_COLOUR[lm["type"]]
                cv2.circle(vis, lm["pixel"], 16, col, 2)

        # current planned route (magenta) — the dynamic plan
        if dynnav.route is not None and len(dynnav.route) > 1:
            for i in range(1, len(dynnav.route)):
                a = tuple(map(int, dynnav.route[i-1]))
                b = tuple(map(int, dynnav.route[i]))
                cv2.line(vis, a, b, (255, 0, 255), 2)
                cv2.circle(vis, b, 5, (255, 0, 255), -1)

        # true path (green) + estimated path (cyan)
        for i in range(1, len(true_path)):
            cv2.line(vis, tuple(map(int,true_path[i-1])),
                     tuple(map(int,true_path[i])), (0,200,0), 2)
        for i in range(1, len(est_path)):
            cv2.line(vis, tuple(map(int,est_path[i-1])),
                     tuple(map(int,est_path[i])), (255,200,0), 2)

        # target + arrival radius
        tx, ty = map(int, target)
        cv2.drawMarker(vis, (tx,ty), (0,0,255), cv2.MARKER_STAR, 46, 4)
        cv2.circle(vis, (tx,ty), int(self.guide.arrive_radius/self.gsd),
                   (0,0,255), 2)

        # drone (estimated)
        ex, ey = map(int, pose.get_pos())
        cv2.circle(vis, (ex,ey), 11, (255,200,0), -1)

        # HUD
        gt = self.guide.to_target(pose.get_pos(), target)
        hud = (f"{gt['distance_m']:.0f}m to target | conf {pose.conf:.2f} "
               f"| re-plans {dynnav.replan_count} | LM {len(confirmed_all)}")
        cv2.rectangle(vis, (0,0), (self.W,50), (0,0,0), -1)
        cv2.putText(vis, hud, (10,33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)
        # legend for route
        cv2.putText(vis, "magenta = live planned route   green = true   cyan = estimated",
                    (10, self.H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,0,255), 1)
        return vis


def main():
    ap = argparse.ArgumentParser(description="Dynamic landmark-based navigation")
    ap.add_argument("--target", required=True, help="target landmark photo")
    ap.add_argument("--gsd",    type=float, default=0.3)
    ap.add_argument("--speed",  type=float, default=9.0)
    ap.add_argument("--start",  default=None,
                    help="optional 'x,y' true start pixel (else default corner)")
    a = ap.parse_args()

    nav = DynamicMapNavigator(gsd=a.gsd)
    target_xy, target_gps = nav.localize_target(a.target)
    start = None
    if a.start:
        sx, sy = a.start.split(",")
        start = (int(sx), int(sy))
    nav.run(target_xy, target_gps, true_start=start, speed_px=a.speed)


if __name__ == "__main__":
    main()
