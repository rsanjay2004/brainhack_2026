"""
RoboVerse 2026 Qualifier — Autonomous Barrel Search Mission
============================================================

Algorithm: "Virtual Target" (Goal + Avoidance vector blending)
-------------------------------------------------------------
At every 20 Hz tick the code:
  1. Computes a Goal Vector   → unit vector from current pos toward waypoint.
  2. Computes an Avoidance Vector → from AvoidancePlanner depth histogram.
  3. Blends them into a Resultant Vector.
  4. Projects a Virtual Target (LOOK_AHEAD metres) along that resultant.
  5. Sends the Virtual Target as an offboard PositionNedYaw setpoint.

This is more robust than purely reactive avoidance: the drone "bends" its
path around obstacles while maintaining forward progress toward the goal.
Reference: Learning Material 3 — "Goal + Avoidance Approach".

Two-phase lawnmower sweep
--------------------------
Phase 1  2 m altitude  → ground-level yellow barrels.
Phase 2  4.5 m altitude → elevated red barrels.

Stuck recovery
--------------
If the drone hasn't advanced > STUCK_DIST_M in STUCK_TIMEOUT_S seconds, it
rotates 90° and moves forward to escape.

IMPORTANT — run before starting this script
-------------------------------------------
In the PX4 terminal (px4>) type:
    commander set_ekf_origin 47.397742 8.545594 488.0
Or use SET ESTIMATOR ORIGIN in QGroundControl.
This is required so PX4 fuses the VIO and allows arming.

Usage
-----
    python qualifier_main.py [path/to/yolo_model.pt]

If no model path is given the script falls back to HSV colour detection.
The Qualifier YOLO model is posted on the RoboVerse Discord channel.
"""

import asyncio
import math
import sys
import time

import numpy as np

from drone_control import Drone
from depth_receiver import DepthReceiver
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task
from barrel_detector import BarrelDetector, DetectionTracker

# ===========================================================================
# TUNABLE CONSTANTS
# ===========================================================================

DEPTH_TOPIC = "/depth_camera"

# Arena
ARENA_W       = 40.0   # m (east axis)
ARENA_D       = 40.0   # m (north axis)
WALL_MARGIN   = 2.0    # m — keep this far from walls

# Flight altitudes
ALT_YELLOW = 2.0       # m — ground-level barrels
ALT_RED    = 4.5       # m — elevated barrels

# Lawnmower row spacing
#   Horizontal FOV ≈ 72° → swath ≈ 2·alt·tan(36°)
#   2 m → ~2.9 m swath; 3.5 m rows give ~20 % overlap
#   4.5 m → ~6.5 m swath; 5.0 m rows give ~25 % overlap
ROW_SPACING_LOW  = 3.5   # m
ROW_SPACING_HIGH = 5.0   # m

# Control loop
CONTROL_HZ      = 20.0   # Hz  (PX4 offboard needs setpoints ≥ every 0.5 s)
ARRIVAL_RADIUS  = 1.2    # m   — waypoint considered reached within this
MISSION_LIMIT   = 600.0  # s   — hard 10-min cap

# Virtual-target blending  (Learning Material 3)
LOOK_AHEAD      = 2.0    # m   — how far ahead the virtual target is projected
W_AVOID         = 0.6    # 0–1 — how strongly avoidance bends the path
                         #       1.0 = full avoidance, 0.0 = straight to goal

# Obstacle avoidance
SAFE_DIST  = 3.5   # m
CRIT_DIST  = 1.2   # m

# Stuck detection  (Learning Material 3)
STUCK_TIMEOUT_S = 12.0   # s  — declare stuck after this many seconds without progress
STUCK_DIST_M    = 0.5    # m  — minimum forward progress to NOT be stuck
STUCK_ESCAPE_M  = 3.0    # m  — how far to fly when escaping

# Detection
MERGE_DIST = 3.0   # m — same barrel if detections within this radius

# Camera intrinsics  (x500 IMX214 Gazebo)
CAM_K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])


# ===========================================================================
class QualifierMission:
    """Autonomous barrel search using virtual-target obstacle avoidance."""

    # -----------------------------------------------------------------------
    def __init__(self, model_path: str = ""):
        self.drone    = Drone()
        self.depth_rx = DepthReceiver(DEPTH_TOPIC)
        self.detector = BarrelDetector(model_path)
        self.tracker  = DetectionTracker(MERGE_DIST)
        self.state    = SharedState()
        self.stop_evt = asyncio.Event()

        self.planner = AvoidancePlanner(
            K=CAM_K, width=640, height=480,
            safe_distance=SAFE_DIST,
            critical_distance=CRIT_DIST,
        )

        self._start_time: float | None = None
        self._waypoints: list[tuple] = []
        self._wp_idx   = 0
        self._phase    = "YELLOW"

        # Stuck detection state
        self._stuck_timer  = time.monotonic()
        self._stuck_ref_n  = 0.0
        self._stuck_ref_e  = 0.0

    # -----------------------------------------------------------------------
    # Time helpers
    # -----------------------------------------------------------------------
    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self) -> float:
        return max(0.0, MISSION_LIMIT - self._elapsed())

    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------
    def _pose(self) -> dict:
        if self.state.latest_position is None:
            return {"north": 0.0, "east": 0.0, "down": 0.0,
                    "yaw": 0.0, "yaw_deg": 0.0}
        return {
            "north":   float(self.state.latest_position.north_m),
            "east":    float(self.state.latest_position.east_m),
            "down":    float(self.state.latest_position.down_m),
            "yaw":     math.radians(float(self.state.latest_yaw or 0.0)),
            "yaw_deg": float(self.state.latest_yaw or 0.0),
        }

    # -----------------------------------------------------------------------
    # Waypoint generation — lawnmower sweep
    # -----------------------------------------------------------------------
    def _build_sweep(self, altitude: float, row_spacing: float) -> list:
        """
        Return (north, east, down) waypoints for a full arena lawnmower.
        Rows alternate north-bound / south-bound so the drone snakes
        continuously without retracing.
        """
        down = -altitude   # NED: negative = upward
        lo   = WALL_MARGIN
        hi_n = ARENA_D - WALL_MARGIN
        hi_e = ARENA_W - WALL_MARGIN

        east_cols = list(np.arange(lo, hi_e + 1e-6, row_spacing))
        wps = []
        for i, east in enumerate(east_cols):
            if i % 2 == 0:
                wps += [(lo,   east, down), (hi_n, east, down)]
            else:
                wps += [(hi_n, east, down), (lo,   east, down)]
        return wps

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------
    def _run_detection(self) -> None:
        result = self.detector.detect()
        if not (result["yellow"] or result["red"]):
            return
        p = self._pose()
        n, e = p["north"], p["east"]
        if result["yellow"] and self.tracker.try_add_yellow(n, e):
            print(f"[DETECT] *** YELLOW #{self.tracker.yellow_count} ***  "
                  f"N={n:.1f} E={e:.1f}  |  {self.tracker.summary()}")
        if result["red"] and self.tracker.try_add_red(n, e):
            print(f"[DETECT] *** RED #{self.tracker.red_count} ***  "
                  f"N={n:.1f} E={e:.1f}  |  {self.tracker.summary()}")

    # -----------------------------------------------------------------------
    # Waypoint helpers
    # -----------------------------------------------------------------------
    def _current_wp(self):
        return self._waypoints[self._wp_idx] if self._wp_idx < len(self._waypoints) else None

    def _arrived(self, wp) -> bool:
        p = self._pose()
        return math.hypot(p["north"] - wp[0], p["east"] - wp[1]) < ARRIVAL_RADIUS

    def _advance_wp(self) -> None:
        self._wp_idx += 1
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP {self._wp_idx}/{len(self._waypoints)} → "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m  "
                  f"T={self._time_left():.0f}s  {self.tracker.summary()}")

    def _start_phase(self, phase: str) -> None:
        self._phase = phase
        if phase == "YELLOW":
            self._waypoints = self._build_sweep(ALT_YELLOW, ROW_SPACING_LOW)
            print(f"\n[PHASE 1] Yellow sweep — {ALT_YELLOW}m, "
                  f"{len(self._waypoints)} waypoints")
        else:
            self._waypoints = self._build_sweep(ALT_RED, ROW_SPACING_HIGH)
            print(f"\n[PHASE 2] Red sweep — {ALT_RED}m, "
                  f"{len(self._waypoints)} waypoints")
        self._wp_idx = 0
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP 1/{len(self._waypoints)} → "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m")

    # -----------------------------------------------------------------------
    # Stuck detection  (Learning Material 3)
    # -----------------------------------------------------------------------
    def _reset_stuck(self) -> None:
        p = self._pose()
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = p["north"]
        self._stuck_ref_e = p["east"]

    def _is_stuck(self) -> bool:
        if time.monotonic() - self._stuck_timer < STUCK_TIMEOUT_S:
            return False
        p = self._pose()
        moved = math.hypot(p["north"] - self._stuck_ref_n,
                           p["east"]  - self._stuck_ref_e)
        return moved < STUCK_DIST_M

    async def _escape_stuck(self) -> None:
        """Rotate 90° and push forward to get unstuck."""
        p = self._pose()
        escape_yaw = (p["yaw_deg"] + 90.0) % 360.0
        print(f"[STUCK] Escaping — rotating to {escape_yaw:.0f}° and pushing forward")
        await self.drone.rotate_to_yaw(escape_yaw)
        # Push STUCK_ESCAPE_M forward in the new heading
        esc_n = p["north"] + STUCK_ESCAPE_M * math.cos(math.radians(escape_yaw))
        esc_e = p["east"]  + STUCK_ESCAPE_M * math.sin(math.radians(escape_yaw))
        wp = self._current_wp()
        esc_d = wp[2] if wp else p["down"]
        await self.drone.send_position_setpoint(esc_n, esc_e, esc_d, escape_yaw)
        await asyncio.sleep(3.0)
        self._reset_stuck()

    # -----------------------------------------------------------------------
    # Virtual Target computation  (Learning Material 3)
    # -----------------------------------------------------------------------
    def _compute_virtual_target(
        self,
        pose: dict,
        target_n: float,
        target_e: float,
        target_d: float,
        depth,
    ) -> tuple[float, float, float, float]:
        """
        Blend goal direction with avoidance direction to produce a virtual
        target (absolute NED setpoint) and a yaw to face toward it.

        Returns: (send_n, send_e, send_d, yaw_deg)
        """
        cur_n = pose["north"]
        cur_e = pose["east"]

        # ------------------------------------------------------------------
        # 1. Goal vector — unit vector toward waypoint
        # ------------------------------------------------------------------
        dn = target_n - cur_n
        de = target_e - cur_e
        dist_to_wp = math.hypot(dn, de)

        if dist_to_wp > 1e-3:
            goal_n = dn / dist_to_wp
            goal_e = de / dist_to_wp
        else:
            goal_n, goal_e = 1.0, 0.0   # already there — default north

        # ------------------------------------------------------------------
        # 2. Avoidance vector — derived from AvoidancePlanner
        # ------------------------------------------------------------------
        avoid_n, avoid_e = 0.0, 0.0
        blocked = False

        if depth is not None:
            av_n, av_e, _, info = self.planner.compute_position_ned(
                depth, pose, step_size=1.0
            )
            blocked = info["blocked"]
            # Direction the planner wants the drone to go
            av_dn = av_n - cur_n
            av_de = av_e - cur_e
            av_dist = math.hypot(av_dn, av_de)
            if av_dist > 1e-3:
                avoid_n = av_dn / av_dist
                avoid_e = av_de / av_dist
                if blocked:
                    cl = info["clearance"]
                    print(f"[AVOID] Blocked  L={cl['left']:.1f} "
                          f"C={cl['center']:.1f} R={cl['right']:.1f}")

        # ------------------------------------------------------------------
        # 3. Blend  (Learning Material 3 — "Resultant Vector")
        # ------------------------------------------------------------------
        if blocked:
            # Fully blocked: trust avoidance completely
            blend_n = avoid_n
            blend_e = avoid_e
        else:
            blend_n = goal_n + W_AVOID * avoid_n
            blend_e = goal_e + W_AVOID * avoid_e

        # Normalise
        blend_mag = math.hypot(blend_n, blend_e)
        if blend_mag > 1e-3:
            blend_n /= blend_mag
            blend_e /= blend_mag
        else:
            blend_n, blend_e = goal_n, goal_e

        # ------------------------------------------------------------------
        # 4. Project virtual target LOOK_AHEAD metres along resultant
        # ------------------------------------------------------------------
        send_n = cur_n + LOOK_AHEAD * blend_n
        send_e = cur_e + LOOK_AHEAD * blend_e
        send_d = target_d   # maintain mission altitude

        # 5. Yaw to face direction of travel
        yaw_deg = math.degrees(math.atan2(blend_e, blend_n))

        return send_n, send_e, send_d, yaw_deg

    # -----------------------------------------------------------------------
    # 20 Hz control loop
    # -----------------------------------------------------------------------
    async def _control_loop(self) -> None:
        dt = 1.0 / CONTROL_HZ
        self._start_phase("YELLOW")

        while True:
            t0 = time.monotonic()

            # Hard time limit
            if self._time_left() < 10.0:
                print("[MISSION] Time limit reached.")
                break

            # Detection
            self._run_detection()

            # Waypoint management
            wp = self._current_wp()
            if wp is None:
                if self._phase == "YELLOW":
                    print(f"\n[PHASE 1 DONE] {self.tracker.summary()}")
                    if self._time_left() > 90:
                        self._start_phase("RED")
                        wp = self._current_wp()
                    else:
                        print("[MISSION] Too little time for red sweep.")
                        break
                else:
                    print(f"\n[PHASE 2 DONE] {self.tracker.summary()}")
                    break
            if wp is None:
                break

            pose = self._pose()

            # Arrival check
            if self._arrived(wp):
                self._advance_wp()
                wp = self._current_wp()
                if wp is None:
                    continue

            # Stuck recovery  (Learning Material 3)
            if self._is_stuck():
                await self._escape_stuck()
                await asyncio.sleep(0)   # yield to event loop
                continue

            target_n, target_e, target_d = wp

            # Virtual target (Goal + Avoidance blend)
            depth = self.depth_rx.get_frame()
            send_n, send_e, send_d, yaw_deg = self._compute_virtual_target(
                pose, target_n, target_e, target_d, depth
            )

            # Offboard setpoint (must be sent every ≤ 0.5 s per PX4 spec)
            await self.drone.send_position_setpoint(
                north=send_n,
                east=send_e,
                down=send_d,
                yaw_deg=yaw_deg,
            )

            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------
    async def run(self) -> None:
        _banner("RoboVerse 2026 Qualifier Mission")
        print("IMPORTANT: Before starting, set EKF origin in PX4 terminal:")
        print("  commander set_ekf_origin 47.397742 8.545594 488.0")
        print("OR use SET ESTIMATOR ORIGIN in QGroundControl.\n")

        await self.drone.connect()
        print("[INIT] Drone connected.")
        await asyncio.sleep(3)

        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )
        print("[INIT] Position monitor running.")
        await asyncio.sleep(2)

        print("[INIT] Arming and taking off …")
        await self.drone.arm_and_takeoff()
        self._start_time = time.monotonic()
        self._reset_stuck()
        print(f"[START] Airborne — 10:00 countdown begins.\n")

        try:
            await self._control_loop()
        except asyncio.CancelledError:
            print("\n[ABORT] Mission cancelled.")
        finally:
            self.stop_evt.set()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

            elapsed = self._elapsed()
            _banner("MISSION RESULTS")
            print(f"  Time elapsed  : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
            print(f"  Yellow barrels: {self.tracker.yellow_count}  ×  50 = "
                  f"{self.tracker.yellow_count * 50} pts")
            print(f"  Red barrels   : {self.tracker.red_count}  × 100 = "
                  f"{self.tracker.red_count * 100} pts")
            print(f"  Total score   : {self.tracker.score()} pts")
            _banner()

            print("[LAND] Landing …")
            await self.drone.land()
            print("[DONE]")


# ===========================================================================
def _banner(title: str = "") -> None:
    bar = "=" * 54
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


async def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_path:
        print(f"[CONFIG] Using YOLO model: {model_path}")
    else:
        print("[CONFIG] No YOLO model specified — using HSV colour detection.")
        print("         Pass path to trained model as first argument, e.g.:")
        print("           python qualifier_main.py barrels.pt\n")

    mission = QualifierMission(model_path)
    try:
        await mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] Keyboard interrupt.")
        try:
            await mission.drone.land()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
