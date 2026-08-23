"""
core/route_planner.py
=====================
Dynamic, landmark-driven route planning — NO precomputed fixed path.

The campus landmarks (junctions, buildings, structures, treelines) form a
GRAPH. Nodes = landmarks (+ the drone's current position + the target).
Edges connect nearby landmarks that are mutually "reachable" (close enough and
not blocked by a large obstacle region).

At every re-plan the planner runs A* from the drone's CURRENT estimated
position to the target through this landmark graph, producing the next
way-landmark to head toward. Because it re-plans from wherever the drone
actually is, the route emerges dynamically from the map — it is never a
fixed line computed once at the start.

Re-planning is triggered when:
  • the drone reaches (or passes) its current way-landmark, OR
  • localisation confidence changes significantly, OR
  • a better route becomes available (shorter A* cost), OR
  • every `replan_interval` steps as a safety refresh.

This is the "navigation decisions driven by map landmarks" core.
"""

import heapq
import numpy as np


class LandmarkGraph:
    """Builds a navigation graph from landmarks and runs A* over it."""

    def __init__(self, landmarks, map_w, map_h,
                 connect_radius=320, obstacle_mask=None):
        """
        landmarks      : list of dicts with 'pixel', 'type', 'label'
        connect_radius : max pixel distance for two landmarks to be linked
        obstacle_mask  : optional HxW uint8 array (255 = blocked) to penalise
                         edges that cross obstacles
        """
        self.lms   = landmarks
        self.map_w = map_w
        self.map_h = map_h
        self.connect_radius = connect_radius
        self.obstacle_mask  = obstacle_mask

        # node list = landmark pixel positions
        self.nodes = [tuple(lm["pixel"]) for lm in landmarks]
        self.labels = [lm["label"] for lm in landmarks]
        self.types  = [lm["type"]  for lm in landmarks]

        # precompute static landmark-to-landmark adjacency
        self._build_adjacency()

    # ── edge cost between two pixel points ─────────────────────────────────
    def _edge_cost(self, a, b):
        d = float(np.hypot(a[0]-b[0], a[1]-b[1]))
        if d > self.connect_radius:
            return None  # not connected
        # obstacle penalty: sample along the segment
        if self.obstacle_mask is not None:
            n = max(2, int(d // 10))
            blocked = 0
            for t in np.linspace(0, 1, n):
                px = int(a[0] + (b[0]-a[0])*t)
                py = int(a[1] + (b[1]-a[1])*t)
                if 0 <= px < self.map_w and 0 <= py < self.map_h:
                    if self.obstacle_mask[py, px] > 0:
                        blocked += 1
            penalty = 1.0 + 4.0 * (blocked / n)   # up to 5x cost if fully blocked
            return d * penalty
        return d

    def _build_adjacency(self):
        n = len(self.nodes)
        self.adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                c = self._edge_cost(self.nodes[i], self.nodes[j])
                if c is not None:
                    self.adj[i].append((j, c))
                    self.adj[j].append((i, c))

    # ── A* from an arbitrary start point to an arbitrary goal point ───────
    def plan(self, start_xy, goal_xy):
        """
        Plan a route from start_xy to goal_xy through the landmark graph.
        start and goal are connected to all landmarks within connect_radius.
        Returns a list of (x, y) waypoints: [start, lm, lm, ..., goal].
        Falls back to a direct [start, goal] if no graph route exists.
        """
        n = len(self.nodes)
        START, GOAL = n, n + 1
        # temporary connections from start/goal to nearby landmarks
        extra = {START: [], GOAL: []}
        for i, node in enumerate(self.nodes):
            cs = self._edge_cost(start_xy, node)
            if cs is not None:
                extra[START].append((i, cs))
            cg = self._edge_cost(goal_xy, node)
            if cg is not None:
                extra[GOAL].append((i, cg))
        # also allow a direct start→goal edge ONLY if reasonably short,
        # so longer trips are forced to route through landmarks
        direct = self._edge_cost(start_xy, goal_xy)
        if direct is not None and np.hypot(start_xy[0]-goal_xy[0],
                                           start_xy[1]-goal_xy[1]) > self.connect_radius * 0.8:
            direct = None

        def neighbours(node):
            if node == START:
                yield from extra[START]
                if direct is not None:
                    yield (GOAL, direct)
            elif node == GOAL:
                return
            else:
                for j, c in self.adj[node]:
                    yield (j, c)
                # landmark → goal if close
                cg = self._edge_cost(self.nodes[node], goal_xy)
                if cg is not None:
                    yield (GOAL, cg)

        def pos(node):
            if node == START: return start_xy
            if node == GOAL:  return goal_xy
            return self.nodes[node]

        def heur(node):
            p = pos(node)
            return float(np.hypot(p[0]-goal_xy[0], p[1]-goal_xy[1]))

        # A*
        open_set = [(heur(START), 0.0, START, [START])]
        best_g = {START: 0.0}
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node == GOAL:
                return [pos(p) for p in path]
            for nxt, cost in neighbours(node):
                ng = g + cost
                if nxt not in best_g or ng < best_g[nxt]:
                    best_g[nxt] = ng
                    heapq.heappush(open_set,
                                   (ng + heur(nxt), ng, nxt, path + [nxt]))

        # no route found → direct line
        return [start_xy, goal_xy]


class DynamicNavigator:
    """
    Wraps the graph and decides, at each step, the next way-landmark to steer
    toward — re-planning whenever conditions change.
    """

    def __init__(self, graph: LandmarkGraph, target_xy,
                 reach_radius=70, replan_interval=25):
        self.graph = graph
        self.target = tuple(target_xy)
        self.reach_radius = reach_radius        # px to consider a waypoint reached
        self.replan_interval = replan_interval  # steps between forced re-plans

        self.route = None          # current planned list of (x,y)
        self.route_idx = 1         # index of the current way-point we steer to
        self.last_conf = None
        self.steps_since_replan = 0
        self.replan_count = 0
        self.replan_reasons = []
        self._last_replan_pos = None   # position at last reached-waypoint replan
        self._stall_count = 0          # consecutive stalled replans

    def _should_replan(self, cur_xy, conf):
        if self.route is None:
            return True, "initial"
        # reached current waypoint?
        wp = self.route[self.route_idx]
        if np.hypot(cur_xy[0]-wp[0], cur_xy[1]-wp[1]) <= self.reach_radius:
            # if this is the LAST waypoint before the goal, don't re-plan
            # repeatedly — just hold heading to the goal (prevents spam when
            # localization stalls near a node)
            if self.route_idx >= len(self.route) - 1:
                return False, None
            return True, "reached_waypoint"
        # confidence changed a lot?
        if self.last_conf is not None and abs(conf - self.last_conf) > 0.25:
            return True, "confidence_change"
        # periodic refresh (lets a better route be found as we move)
        if self.steps_since_replan >= self.replan_interval:
            return True, "periodic_refresh"
        return False, None

    def step(self, cur_xy, conf):
        """
        Given the drone's current estimated position and confidence,
        return the next waypoint (x, y) to steer toward, re-planning if needed.
        """
        replan, reason = self._should_replan(cur_xy, conf)

        # dampener: if we've re-planned for the same reason many times in a row
        # without the position changing (localization stalled), stop spamming
        if replan and reason == "reached_waypoint":
            if (self._last_replan_pos is not None and
                np.hypot(cur_xy[0]-self._last_replan_pos[0],
                         cur_xy[1]-self._last_replan_pos[1]) < 3):
                self._stall_count += 1
            else:
                self._stall_count = 0
            if self._stall_count > 3:
                # position not advancing — hold current waypoint, skip re-plan
                replan, reason = False, None

        if replan:
            self.route = self.graph.plan(cur_xy, self.target)
            self.route_idx = 1 if len(self.route) > 1 else 0
            self.steps_since_replan = 0
            self.replan_count += 1
            self.replan_reasons.append(reason)
            self._last_replan_pos = tuple(cur_xy)
        else:
            # advance through any waypoints we've already passed
            while (self.route_idx < len(self.route) - 1 and
                   np.hypot(cur_xy[0]-self.route[self.route_idx][0],
                            cur_xy[1]-self.route[self.route_idx][1])
                   <= self.reach_radius):
                self.route_idx += 1
            self.steps_since_replan += 1

        self.last_conf = conf
        wp = self.route[min(self.route_idx, len(self.route)-1)]
        return wp, (reason if replan else None)

    def reached_target(self, cur_xy, threshold):
        return np.hypot(cur_xy[0]-self.target[0],
                        cur_xy[1]-self.target[1]) <= threshold
