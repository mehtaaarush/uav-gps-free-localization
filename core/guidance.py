"""
core/guidance.py
================
Turns "where I am" + "where the target is" into navigation commands.

Outputs three forms (all derived from the same vector to target):
  1. Human-readable     : "Target 340 m NE — head forward-right"
  2. Heading + distance : bearing in degrees, distance in metres
  3. MAVLink-style cmd   : a velocity setpoint dict — LOGGED, never sent,
                           unless an explicit ARM flag is enabled (it is not).

SAFETY: this module never talks to a real flight controller. The send_command()
method is a stub that only prints unless armed=True AND a real link is wired in
later. For a final-year Stage-A demo you read the printed commands.
"""

import math


# 8-point compass for human-readable bearings
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass(bearing_deg):
    idx = int(((bearing_deg + 22.5) % 360) // 45)
    return _COMPASS[idx]


def _plain_direction(bearing_deg):
    """Relative to drone facing 'up'/north on the map → forward/left/right."""
    b = bearing_deg % 360
    if   b < 22.5 or b >= 337.5: return "go forward"
    elif b < 67.5:  return "head forward-right"
    elif b < 112.5: return "turn right"
    elif b < 157.5: return "head back-right"
    elif b < 202.5: return "go backward"
    elif b < 247.5: return "head back-left"
    elif b < 292.5: return "turn left"
    else:           return "head forward-left"


class Guidance:
    """
    metres_per_pixel : ground sample distance of the satellite map.
                       At a known altitude this is fixed; for the Stage-A
                       simulation we pass the map's gsd directly.
    arrive_radius_m  : how close (metres) counts as "arrived".
    """

    def __init__(self, metres_per_pixel=0.3, arrive_radius_m=15.0,
                 armed=False):
        self.mpp = metres_per_pixel
        self.arrive_radius = arrive_radius_m
        self.armed = armed            # MUST stay False until SITL-tested

    def to_target(self, cur_xy, target_xy):
        """
        Compute guidance from current pixel position to target pixel position.
        Returns a dict with bearing, distance, plain text, and a command.
        """
        cx, cy = cur_xy
        tx, ty = target_xy
        dpx, dpy = (tx - cx), (ty - cy)

        dist_px = math.hypot(dpx, dpy)
        dist_m  = dist_px * self.mpp

        # bearing: 0° = north (up on map), clockwise. image y grows downward.
        bearing = (math.degrees(math.atan2(dpx, -dpy))) % 360

        arrived = dist_m <= self.arrive_radius

        # normalised velocity setpoint (unit vector toward target)
        if dist_px > 1e-6:
            vx, vy = dpx / dist_px, dpy / dist_px
        else:
            vx, vy = 0.0, 0.0

        cmd = {
            # MAVLink SET_POSITION_TARGET_LOCAL_NED style (body frame)
            "vx_north":  round(-vy, 3),     # +north
            "vy_east":   round( vx, 3),     # +east
            "vz_down":   0.0,               # hold altitude
            "yaw_deg":   round(bearing, 1),
        }

        return {
            "bearing_deg":  round(bearing, 1),
            "compass":      _compass(bearing),
            "distance_m":   round(dist_m, 1),
            "direction":    "ARRIVED" if arrived else _plain_direction(bearing),
            "arrived":      arrived,
            "command":      cmd,
        }

    def send_command(self, cmd):
        """
        SAFETY STUB. Prints the command. Only a real MAVLink link (added later,
        after SITL testing) and armed=True would actually transmit.
        """
        if not self.armed:
            # default path for the whole project until real flight
            return f"[SIM] would send: {cmd}"
        # ── real flight path (intentionally not implemented here) ───────────
        raise NotImplementedError(
            "Real flight-controller link is not enabled. Test in SITL first, "
            "then implement the MAVLink transmit here.")
