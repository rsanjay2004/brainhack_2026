"""
RoboVerse 2026 Qualifier — Autonomous Barrel Search Mission
============================================================

Strategy
--------
Phase 1  Serpentine lawnmower at 2 m altitude.
         Yellow barrels are at ground level, so they appear clearly in the
         forward-facing camera at low altitude.  Red barrels that happen to
         be in the horizontal FOV are also captured here.

Phase 2  Serpentine lawnmower at 4.5 m altitude.
         Elevated red barrels (placed on top of structures) enter the camera
         FOV only from a higher vantage point.

Throughout both phases:
  • Depth camera feeds the AvoidancePlanner to steer around obstacles.
  • The 20 Hz control loop keeps sending offboard setpoints (required by PX4).
  • DetectionTracker deduplicates by spatial proximity.

Usage
-----
    python qualifier_main.py

Requirements
------------
    pip install mavsdk ultralytics opencv-python numpy scipy
    Gazebo / PX4-SITL running with the RoboVerse world.
"""

import asyncio
import math
import time

import numpy as np

from drone_control import Drone
from depth_receiver import DepthReceiver
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task
from barrel_detector import BarrelDetector, DetectionTracker

# ===========================================================================
# TUNABLE CONSTANTS — adjust before the qualifier if the simulator map differs
# ===========================================================================

DEPTH_TOPIC = "/depth_camera"

# Arena
ARENA_W = 40.0       # metres (east)
ARENA_D = 40.0       # metres (north)
WALL_MARGIN = 2.0    # keep this far from walls

# Flight altitudes
ALT_YELLOW = 2.0     # m — ground-level barrels
ALT_RED    = 4.5     # m — elevated barrels

# Lawnmower row spacing (camera horizontal FOV ≈ 72°)
#   At 2 m:   ground swath ≈ 2×2×tan(36°) ≈ 2.9 m  → 3.5 m gives ~20 % overlap
#   At 4.5 m: ground swath ≈ 6.6 m             → 5.0 m gives ~25 % overlap
ROW_SPACING_LOW  = 3.5   # m
ROW_SPACING_HIGH = 5.0   # m

# Control
CONTROL_HZ        = 20.0   # Hz — setpoint rate (PX4 needs ≥ 2 Hz in offboard)
ARRIVAL_RADIUS    = 1.2    # m — distance at which a waypoint is considered reached
MISSION_DURATION  = 600.0  # s — 10 min hard limit

# Obstacle avoidance
SAFE_DIST  = 3.5   # m
CRIT_DIST  = 1.2   # m

# Detection deduplication
MERGE_DIST = 3.0   # m — detections within this radius count as the same barrel

# Camera intrinsics (x500 Gazebo IMX214)
CAM_K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])


# ===========================================================================
class QualifierMission:
    """
    Top-level mission controller.

    The `run()` coroutine is the single entry point.  Everything else is
    driven by the 20 Hz `_control_loop` coroutine which runs during flight.
    """

    # -----------------------------------------------------------------------
    def __init__(self):
        self.drone     = Drone()
        self.depth_rx  = DepthReceiver(DEPTH_TOPIC)
        self.detector  = BarrelDetector()          # HSV colour + optional YOLO
        self.tracker   = DetectionTracker(MERGE_DIST)
        self.state     = SharedState()
        self.stop_evt  = asyncio.Event()

        self.planner = AvoidancePlanner(
            K=CAM_K, width=640, height=480,
            safe_distance=SAFE_DIST,
            critical_distance=CRIT_DIST,
        )

        self._start_time: float | None = None
        self._waypoints: list[tuple[float, float, float, float]] = []
        self._wp_idx   = 0
        self._phase    = "YELLOW"

    # -----------------------------------------------------------------------
    # Time helpers
    # -----------------------------------------------------------------------
    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self) -> float:
        return max(0.0, MISSION_DURATION - self._elapsed())

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
    # Waypoint generation
    # -----------------------------------------------------------------------
    def _build_sweep(
        self,
        altitude: float,
        row_spacing: float,
    ) -> list[tuple[float, float, float, float]]:
        """
        Return a list of (north, east, down, yaw_deg) waypoints for a
        lawnmower sweep of the arena at the given altitude.

        Yaw is set to face the direction of travel so the forward camera
        and depth sensor always look ahead.
        """
        down = -altitude          # NED convention: negative = upward
        lo   = WALL_MARGIN
        hi_e = ARENA_W - WALL_MARGIN
        hi_n = ARENA_D - WALL_MARGIN

        east_cols = list(np.arange(lo, hi_e + 1e-6, row_spacing))
        waypoints = []

        for i, east in enumerate(east_cols):
            if i % 2 == 0:
                # Flying north → yaw 0°
                waypoints.append((lo,   east, down,  0.0))
                waypoints.append((hi_n, east, down,  0.0))
            else:
                # Flying south → yaw 180°
                waypoints.append((hi_n, east, down, 180.0))
                waypoints.append((lo,   east, down, 180.0))

        return waypoints

    # -----------------------------------------------------------------------
    # Detection & logging
    # -----------------------------------------------------------------------
    def _run_detection(self) -> None:
        result = self.detector.detect()
        if not (result["yellow"] or result["red"]):
            return

        pose = self._pose()
        n, e = pose["north"], pose["east"]

        if result["yellow"]:
            if self.tracker.try_add_yellow(n, e):
                print(
                    f"[DETECT] *** YELLOW barrel #{self.tracker.yellow_count} ***  "
                    f"N={n:.1f} E={e:.1f}  |  {self.tracker.summary()}"
                )

        if result["red"]:
            if self.tracker.try_add_red(n, e):
                print(
                    f"[DETECT] *** RED barrel #{self.tracker.red_count} ***  "
                    f"N={n:.1f} E={e:.1f}  |  {self.tracker.summary()}"
                )

    # -----------------------------------------------------------------------
    # Waypoint helpers
    # -----------------------------------------------------------------------
    def _current_wp(self) -> tuple | None:
        if self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None

    def _arrived(self, wp: tuple) -> bool:
        p = self._pose()
        return math.hypot(p["north"] - wp[0], p["east"] - wp[1]) < ARRIVAL_RADIUS

    def _advance_wp(self) -> None:
        self._wp_idx += 1
        wp = self._current_wp()
        if wp:
            print(
                f"[NAV] WP {self._wp_idx}/{len(self._waypoints)} → "
                f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m  "
                f"T-left={self._time_left():.0f}s  {self.tracker.summary()}"
            )

    # -----------------------------------------------------------------------
    # Switch to next phase
    # -----------------------------------------------------------------------
    def _start_phase(self, phase: str) -> None:
        self._phase = phase
        if phase == "YELLOW":
            self._waypoints = self._build_sweep(ALT_YELLOW, ROW_SPACING_LOW)
            print(
                f"\n[PHASE 1] Yellow sweep — {ALT_YELLOW} m altitude, "
                f"{len(self._waypoints)} waypoints"
            )
        elif phase == "RED":
            self._waypoints = self._build_sweep(ALT_RED, ROW_SPACING_HIGH)
            print(
                f"\n[PHASE 2] Red sweep — {ALT_RED} m altitude, "
                f"{len(self._waypoints)} waypoints"
            )
        self._wp_idx = 0
        wp = self._current_wp()
        if wp:
            print(
                f"[NAV] WP 1/{len(self._waypoints)} → "
                f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m"
            )

    # -----------------------------------------------------------------------
    # Main 20 Hz control loop
    # -----------------------------------------------------------------------
    async def _control_loop(self) -> None:
        dt = 1.0 / CONTROL_HZ
        self._start_phase("YELLOW")

        while True:
            t0 = time.monotonic()

            # Hard time limit
            if self._time_left() < 10.0:
                print("[MISSION] Time limit reached — exiting control loop.")
                break

            # Detection
            self._run_detection()

            # Waypoint management
            wp = self._current_wp()
            if wp is None:
                # Current phase exhausted
                if self._phase == "YELLOW":
                    print(
                        f"\n[PHASE 1 DONE] Yellow sweep complete.  "
                        f"{self.tracker.summary()}"
                    )
                    if self._time_left() > 90:
                        self._start_phase("RED")
                        wp = self._current_wp()
                    else:
                        print("[MISSION] Too little time for red sweep — landing.")
                        break
                else:
                    print(
                        f"\n[PHASE 2 DONE] Red sweep complete.  "
                        f"{self.tracker.summary()}"
                    )
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

            target_n, target_e, target_d, target_yaw = wp

            # ----------------------------------------------------------------
            # Obstacle avoidance
            # ----------------------------------------------------------------
            depth = self.depth_rx.get_frame()
            send_n, send_e, send_d = target_n, target_e, target_d

            if depth is not None:
                av_n, av_e, av_d, info = self.planner.compute_position_ned(
                    depth, pose, step_size=1.5
                )
                if info["blocked"]:
                    cl = info["clearance"]
                    print(
                        f"[AVOID] Obstacle  L={cl['left']:.1f}  "
                        f"C={cl['center']:.1f}  R={cl['right']:.1f}  "
                        f"→ stepping N={av_n:.1f} E={av_e:.1f}"
                    )
                    # Accept avoidance lateral redirect, keep mission altitude
                    send_n, send_e = av_n, av_e

            # ----------------------------------------------------------------
            # Send offboard setpoint (must be sent continuously for PX4)
            # ----------------------------------------------------------------
            await self.drone.send_position_setpoint(
                north=send_n,
                east=send_e,
                down=send_d,
                yaw_deg=target_yaw,
            )

            # Maintain loop rate
            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

    # -----------------------------------------------------------------------
    # Mission entry point
    # -----------------------------------------------------------------------
    async def run(self) -> None:
        _banner("RoboVerse 2026 Qualifier Mission")

        # Connect
        await self.drone.connect()
        print("[INIT] Drone connected.")
        await asyncio.sleep(3)

        # Start background telemetry monitor
        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )
        print("[INIT] Position monitor running.")
        await asyncio.sleep(2)

        # Arm and take off
        print("[INIT] Arming and taking off …")
        await self.drone.arm_and_takeoff()
        self._start_time = time.monotonic()
        print(f"[START] Airborne — 10:00 countdown begins.\n")

        try:
            await self._control_loop()

        except asyncio.CancelledError:
            print("\n[ABORT] Mission cancelled.")

        finally:
            # Shut down telemetry
            self.stop_evt.set()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

            # Print results
            elapsed = self._elapsed()
            _banner("MISSION RESULTS")
            print(f"  Time elapsed  : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
            print(f"  Yellow barrels: {self.tracker.yellow_count}  ×  50 = "
                  f"{self.tracker.yellow_count * 50} pts")
            print(f"  Red barrels   : {self.tracker.red_count}  × 100 = "
                  f"{self.tracker.red_count * 100} pts")
            print(f"  Total score   : {self.tracker.score()} pts")
            _banner()

            # Land
            print("[LAND] Initiating landing …")
            await self.drone.land()
            print("[DONE]")


# ===========================================================================
# Helpers
# ===========================================================================

def _banner(title: str = "") -> None:
    bar = "=" * 52
    if title:
        print(f"\n{bar}")
        print(f"  {title}")
        print(f"{bar}")
    else:
        print(bar)


# ===========================================================================
# Entry point
# ===========================================================================

async def main() -> None:
    mission = QualifierMission()
    try:
        await mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] Keyboard interrupt — landing.")
        try:
            await mission.drone.land()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
