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
from GlobalMapper import GlobalMapper

DEPTH_TOPIC = "/depth_camera"

# L-shaped arena zone offsets from spawn NED (m)
# Map: world origin bottom-left; drone spawns ~N=4 E=12
# Zone 1 — main floor (bottom section of L)
ZONE1_N = (-4.0,  16.0)
ZONE1_E = (-12.0, 16.0)
# Zone 2 — upper-left arm
ZONE2_N = (16.0,  32.0)
ZONE2_E = (-12.0,  0.0)
# Zone 3 — upper-right arm
ZONE3_N = (16.0,  32.0)
ZONE3_E = (12.0,  28.0)
# Inaccessible cutout between the two upper arms
FORBIDDEN_N = (16.0, 36.0)
FORBIDDEN_E = ( 0.0, 12.0)

WALL_MARGIN = 2.5   # stay this far from walls at all times

# Altitudes
ALT_YELLOW = 1.8    # low — ground-level yellow barrels visible in lower frame
ALT_RED    = 4.5    # high — elevated red barrels come into camera FOV

# Lawnmower row spacing
# Camera horiz FOV ~72° → swath ~2*alt*tan(36°)
# 1.8 m alt → ~2.6 m swath; 3.0 m rows give ~15% overlap
# 4.5 m alt → ~6.5 m swath; 5.0 m rows give ~25% overlap
ROW_SPACING_LOW  = 3.0
ROW_SPACING_HIGH = 5.0

CONTROL_HZ     = 20.0
ARRIVAL_RADIUS = 1.0   # horizontal arrival threshold (m)
ARRIVAL_ALT    = 0.5   # vertical arrival threshold (m)
MISSION_LIMIT  = 600.0 # s — 10 min hard cap

# Virtual target blending
LOOK_AHEAD  = 1.5  # m — smaller = tighter path following
W_AVOID     = 0.5  # depth-camera avoidance weight
W_MEM_AVOID = 0.3  # memory map avoidance weight

# Avoidance
SAFE_DIST = 3.0
CRIT_DIST = 1.5

# Map memory
MAP_RETENTION_M  = 15.0  # prune obstacles beyond this radius (m)
MAP_INFLUENCE_M  = 4.5   # repulsion influence radius (m)
MAP_Z_MAX        = 10.0  # max depth to trust in mapper (m)

# Stuck detection
STUCK_TIMEOUT_S = 10.0
STUCK_DIST_M    = 0.4
STUCK_ESCAPE_M  = 2.5

# Detection confirmation: barrel must appear in this many consecutive frames
DETECT_CONFIRM = 2

MERGE_DIST = 3.0

CAM_K = np.array([[433.0, 0.0, 320.0],
                  [0.0,   433.0, 240.0],
                  [0.0,   0.0,   1.0]])


class QualifierMission:

    def __init__(self, model_path=""):
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

        self.mapper = GlobalMapper(
            K=CAM_K,
            cam_height=ALT_YELLOW,
            obs_h_min=0.1, obs_h_max=2.0,
            z_min=0.3, z_max=MAP_Z_MAX,
            yaw_in_degrees=True, yaw_clockwise=True,
            yaw_smoothing=0.8,
        )

        self._start_time = None
        self._waypoints  = []
        self._wp_idx     = 0
        self._phase      = "YELLOW"

        # NED origin recorded at takeoff — arena is offset from here
        self._origin_n = 0.0
        self._origin_e = 0.0

        # Stuck tracking
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = 0.0
        self._stuck_ref_e = 0.0

        # Consecutive detection counters
        self._yellow_streak = 0
        self._red_streak    = 0

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def _elapsed(self):
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self):
        return max(0.0, MISSION_LIMIT - self._elapsed())

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------
    def _pose(self):
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

    # ------------------------------------------------------------------
    # Arena boundary helpers — L-shaped arena aware
    # All setpoints MUST be clamped through here before being sent.
    # Offsets from initial spawn NED so the sweep works regardless of
    # where in the world the drone spawns.
    # ------------------------------------------------------------------
    @property
    def _n_min(self): return self._origin_n + ZONE1_N[0] + WALL_MARGIN
    @property
    def _n_max(self): return self._origin_n + ZONE2_N[1] - WALL_MARGIN
    @property
    def _e_min(self): return self._origin_e + ZONE1_E[0] + WALL_MARGIN
    @property
    def _e_max(self): return self._origin_e + ZONE3_E[1] - WALL_MARGIN

    def _in_forbidden(self, n, e, margin=0.0):
        fn0 = self._origin_n + FORBIDDEN_N[0] - margin
        fn1 = self._origin_n + FORBIDDEN_N[1] + margin
        fe0 = self._origin_e + FORBIDDEN_E[0] - margin
        fe1 = self._origin_e + FORBIDDEN_E[1] + margin
        return fn0 <= n <= fn1 and fe0 <= e <= fe1

    def _clamp(self, n, e):
        n = max(self._n_min, min(self._n_max, n))
        e = max(self._e_min, min(self._e_max, e))
        # If clamped into the forbidden cutout, nudge toward nearest safe arm
        if self._in_forbidden(n, e):
            # Push east toward zone 3 or west toward zone 2
            mid_e = self._origin_e + (FORBIDDEN_E[0] + FORBIDDEN_E[1]) / 2.0
            if e >= mid_e:
                e = self._origin_e + ZONE3_E[0] + WALL_MARGIN
            else:
                e = self._origin_e + ZONE2_E[1] - WALL_MARGIN
        return n, e

    def _is_near_wall(self, n, e, extra=0.0):
        m = WALL_MARGIN + extra
        out_of_bounds = (n < self._n_min - WALL_MARGIN + m or
                         n > self._n_max + WALL_MARGIN - m or
                         e < self._e_min - WALL_MARGIN + m or
                         e > self._e_max + WALL_MARGIN - m)
        return out_of_bounds or self._in_forbidden(n, e, margin=m)

    # ------------------------------------------------------------------
    # Sweep waypoint generation — covers all three L-shaped zones
    # ------------------------------------------------------------------
    def _build_zone_sweep(self, altitude, row_spacing):
        down  = -altitude
        zones = [
            (ZONE1_N, ZONE1_E),
            (ZONE2_N, ZONE2_E),
            (ZONE3_N, ZONE3_E),
        ]
        wps = []
        for (n_lo, n_hi), (e_lo, e_hi) in zones:
            n0 = self._origin_n + n_lo + WALL_MARGIN
            n1 = self._origin_n + n_hi - WALL_MARGIN
            e0 = self._origin_e + e_lo + WALL_MARGIN
            e1 = self._origin_e + e_hi - WALL_MARGIN
            if n1 <= n0 or e1 <= e0:
                continue
            east_cols = list(np.arange(e0, e1 + 1e-6, row_spacing))
            base = len(wps)
            for i, east in enumerate(east_cols):
                east = max(e0, min(e1, east))
                if (i + base) % 2 == 0:
                    wps += [(n0, east, down), (n1, east, down)]
                else:
                    wps += [(n1, east, down), (n0, east, down)]
        return wps

    # ------------------------------------------------------------------
    # Startup 360° scan — rotate in place (position-locked) and seed map
    # ------------------------------------------------------------------
    async def _startup_scan(self):
        p = self._pose()
        hold_n, hold_e, hold_d = p["north"], p["east"], p["down"]
        print("[SCAN] Starting 360° horizon scan — holding position")

        for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
            await self.drone.rotate_to_yaw(float(yaw))
            await asyncio.sleep(1.2)   # let frame settle after rotation stops
            depth = self.depth_rx.get_frame()
            pose  = self._pose()
            if depth is not None:
                self.mapper.update_frame(depth, pose)
            # Keep position locked throughout the dwell
            await self.drone.send_position_setpoint(hold_n, hold_e, hold_d, float(yaw))

        # Return to north-facing heading
        await self.drone.rotate_to_yaw(0.0)
        await asyncio.sleep(0.5)
        pts = self.mapper.get_global_points()
        print(f"[SCAN] Done — {len(pts)} obstacle points in initial map")

    # ------------------------------------------------------------------
    # Detection with consecutive-frame confirmation
    # ------------------------------------------------------------------
    def _run_detection(self):
        result = self.detector.detect(phase=self._phase)

        self._yellow_streak = (self._yellow_streak + 1) if result["yellow"] else 0
        self._red_streak    = (self._red_streak    + 1) if result["red"]    else 0

        p = self._pose()
        n, e = p["north"], p["east"]

        if self._yellow_streak >= DETECT_CONFIRM:
            if self.tracker.try_add_yellow(n, e):
                print(f"[DETECT] YELLOW #{self.tracker.yellow_count}  "
                      f"N={n:.1f} E={e:.1f}  {self.tracker.summary()}")
            self._yellow_streak = 0

        if self._red_streak >= DETECT_CONFIRM:
            if self.tracker.try_add_red(n, e):
                print(f"[DETECT] RED #{self.tracker.red_count}  "
                      f"N={n:.1f} E={e:.1f}  {self.tracker.summary()}")
            self._red_streak = 0

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------
    def _current_wp(self):
        return self._waypoints[self._wp_idx] if self._wp_idx < len(self._waypoints) else None

    def _arrived(self, wp):
        p = self._pose()
        horiz = math.hypot(p["north"] - wp[0], p["east"] - wp[1])
        vert  = abs(p["down"] - wp[2])
        return horiz < ARRIVAL_RADIUS and vert < ARRIVAL_ALT

    def _advance_wp(self):
        self._wp_idx += 1
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP {self._wp_idx}/{len(self._waypoints)}  "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m  "
                  f"T={self._time_left():.0f}s  {self.tracker.summary()}")

    def _start_phase(self, phase):
        self._phase = phase
        if phase == "YELLOW":
            self._waypoints = self._build_zone_sweep(ALT_YELLOW, ROW_SPACING_LOW)
            print(f"\n[PHASE 1] Yellow sweep  alt={ALT_YELLOW}m  "
                  f"{len(self._waypoints)} waypoints")
        else:
            self._waypoints = self._build_zone_sweep(ALT_RED, ROW_SPACING_HIGH)
            print(f"\n[PHASE 2] Red sweep  alt={ALT_RED}m  "
                  f"{len(self._waypoints)} waypoints")
        self._wp_idx        = 0
        self._yellow_streak = 0
        self._red_streak    = 0
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP 1/{len(self._waypoints)}  "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m")

    # ------------------------------------------------------------------
    # Stuck detection — resets automatically when progress is made
    # ------------------------------------------------------------------
    def _reset_stuck(self):
        p = self._pose()
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = p["north"]
        self._stuck_ref_e = p["east"]

    def _check_stuck(self):
        p = self._pose()
        moved = math.hypot(p["north"] - self._stuck_ref_n,
                           p["east"]  - self._stuck_ref_e)
        if moved >= STUCK_DIST_M:
            # Made progress — slide the reference window forward
            self._stuck_ref_n = p["north"]
            self._stuck_ref_e = p["east"]
            self._stuck_timer = time.monotonic()
            return False
        return time.monotonic() - self._stuck_timer > STUCK_TIMEOUT_S

    async def _escape_stuck(self):
        p = self._pose()
        print(f"[STUCK] at N={p['north']:.1f} E={p['east']:.1f} — trying escape directions")
        wp = self._current_wp()

        # Try 4 escape angles; pick first that keeps drone in bounds
        for delta in [90, -90, 180, 45]:
            yaw = (p["yaw_deg"] + delta) % 360.0
            cn  = p["north"] + STUCK_ESCAPE_M * math.cos(math.radians(yaw))
            ce  = p["east"]  + STUCK_ESCAPE_M * math.sin(math.radians(yaw))
            cn, ce = self._clamp(cn, ce)
            if not self._is_near_wall(cn, ce):
                print(f"[STUCK] escaping → yaw={yaw:.0f}°")
                alt_d = wp[2] if wp else p["down"]
                await self.drone.rotate_to_yaw(yaw)
                await self.drone.send_position_setpoint(cn, ce, alt_d, yaw)
                await asyncio.sleep(3.0)
                break

        self._reset_stuck()

    # ------------------------------------------------------------------
    # Virtual target: goal + avoidance blend, clamped to arena
    # ------------------------------------------------------------------
    def _compute_setpoint(self, pose, target_n, target_e, target_d, depth):
        cur_n, cur_e = pose["north"], pose["east"]

        # Emergency: drone is outside/near wall or in forbidden zone — push to zone 1 centre
        if self._is_near_wall(cur_n, cur_e, extra=0.5):
            safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
            safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0
            yaw = math.degrees(math.atan2(safe_e - cur_e, safe_n - cur_n))
            print(f"[BOUNDARY] Near wall/forbidden at N={cur_n:.1f} E={cur_e:.1f} — recovering")
            return safe_n, safe_e, target_d, yaw

        # Goal vector toward current waypoint
        dn   = target_n - cur_n
        de   = target_e - cur_e
        dist = math.hypot(dn, de)
        goal_n = (dn / dist) if dist > 1e-3 else 1.0
        goal_e = (de / dist) if dist > 1e-3 else 0.0

        # Avoidance vector from depth camera
        avoid_n, avoid_e, blocked = 0.0, 0.0, False
        if depth is not None:
            av_n, av_e, _, info = self.planner.compute_position_ned(
                depth, pose, step_size=1.0
            )
            blocked = info["blocked"]
            av_dist = math.hypot(av_n - cur_n, av_e - cur_e)
            if av_dist > 1e-3:
                avoid_n = (av_n - cur_n) / av_dist
                avoid_e = (av_e - cur_e) / av_dist
            if blocked:
                cl = info["clearance"]
                print(f"[AVOID] L={cl['left']:.1f} C={cl['center']:.1f} "
                      f"R={cl['right']:.1f}")

        # Memory-map repulsion — obstacles remembered from past frames
        mem_n, mem_e = self.mapper.get_repulsion_vector(cur_n, cur_e, MAP_INFLUENCE_M)

        # Blend goal + depth avoidance + memory avoidance
        if blocked:
            blend_n = avoid_n + W_MEM_AVOID * mem_n
            blend_e = avoid_e + W_MEM_AVOID * mem_e
        else:
            blend_n = goal_n + W_AVOID * avoid_n + W_MEM_AVOID * mem_n
            blend_e = goal_e + W_AVOID * avoid_e + W_MEM_AVOID * mem_e

        mag = math.hypot(blend_n, blend_e)
        if mag > 1e-3:
            blend_n /= mag
            blend_e /= mag
        else:
            blend_n, blend_e = goal_n, goal_e

        # Project virtual target, then clamp — prevents flying outside arena
        raw_n = cur_n + LOOK_AHEAD * blend_n
        raw_e = cur_e + LOOK_AHEAD * blend_e
        send_n, send_e = self._clamp(raw_n, raw_e)

        yaw_deg = math.degrees(math.atan2(blend_e, blend_n))
        return send_n, send_e, target_d, yaw_deg

    # ------------------------------------------------------------------
    # 20 Hz control loop
    # ------------------------------------------------------------------
    async def _control_loop(self):
        dt = 1.0 / CONTROL_HZ
        await self._startup_scan()
        self._start_phase("YELLOW")

        while True:
            t0 = time.monotonic()

            if self._time_left() < 10.0:
                print("[MISSION] Time limit reached.")
                break

            self._run_detection()

            wp = self._current_wp()
            if wp is None:
                if self._phase == "YELLOW":
                    print(f"\n[PHASE 1 DONE] {self.tracker.summary()}")
                    if self._time_left() > 90:
                        self._start_phase("RED")
                        wp = self._current_wp()
                    else:
                        print("[MISSION] Not enough time for red sweep.")
                        break
                else:
                    print(f"\n[PHASE 2 DONE] {self.tracker.summary()}")
                    break
            if wp is None:
                break

            pose  = self._pose()
            cur_n = pose["north"]
            cur_e = pose["east"]

            # Emergency boundary recovery — push to centre of zone 1 (always safe)
            if self._is_near_wall(cur_n, cur_e, extra=0.5):
                safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
                safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0
                await self.drone.send_position_setpoint(safe_n, safe_e, wp[2], 0.0)
                await asyncio.sleep(0.05)
                continue

            if self._arrived(wp):
                self._advance_wp()
                wp = self._current_wp()
                if wp is None:
                    continue

            if self._check_stuck():
                await self._escape_stuck()
                await asyncio.sleep(0)
                continue

            target_n, target_e, target_d = wp
            depth = self.depth_rx.get_frame()

            # Feed depth into sliding-window map
            if depth is not None:
                self.mapper.update_frame(depth, pose)
                self.mapper.prune(pose["north"], pose["east"], MAP_RETENTION_M)

            send_n, send_e, send_d, yaw_deg = self._compute_setpoint(
                pose, target_n, target_e, target_d, depth
            )

            await self.drone.send_position_setpoint(
                north=send_n, east=send_e, down=send_d, yaw_deg=yaw_deg
            )

            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def run(self):
        print("=" * 50)
        print("  RoboVerse 2026 Qualifier")
        print("=" * 50)
        print("Pre-flight checklist:")
        print("  1) Start simulator: ./start_px4.sh  (choose x500_vision)")
        print("  2) In PX4 terminal: commander set_ekf_origin 47.397742 8.545594 488.0")
        print("  3) Then run this script\n")

        await self.drone.connect()
        print("[INIT] Connected")
        await asyncio.sleep(3)

        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )
        await asyncio.sleep(2)

        print("[INIT] Arming and taking off...")
        await self.drone.arm_and_takeoff()

        # Record NED spawn position — all waypoints offset from here
        p = self._pose()
        self._origin_n = p["north"]
        self._origin_e = p["east"]
        print(f"[INIT] Spawn NED: N={self._origin_n:.2f} E={self._origin_e:.2f}")
        print(f"[INIT] Zone 1 (main floor)    "
              f"N [{self._origin_n+ZONE1_N[0]:.1f} → {self._origin_n+ZONE1_N[1]:.1f}]  "
              f"E [{self._origin_e+ZONE1_E[0]:.1f} → {self._origin_e+ZONE1_E[1]:.1f}]")
        print(f"[INIT] Zone 2 (upper-left)    "
              f"N [{self._origin_n+ZONE2_N[0]:.1f} → {self._origin_n+ZONE2_N[1]:.1f}]  "
              f"E [{self._origin_e+ZONE2_E[0]:.1f} → {self._origin_e+ZONE2_E[1]:.1f}]")
        print(f"[INIT] Zone 3 (upper-right)   "
              f"N [{self._origin_n+ZONE3_N[0]:.1f} → {self._origin_n+ZONE3_N[1]:.1f}]  "
              f"E [{self._origin_e+ZONE3_E[0]:.1f} → {self._origin_e+ZONE3_E[1]:.1f}]")

        self._start_time = time.monotonic()
        self._reset_stuck()
        print("[START] Airborne — 10:00 countdown\n")

        try:
            await self._control_loop()
        except asyncio.CancelledError:
            print("\n[ABORT] Cancelled")
        finally:
            self.stop_evt.set()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

            elapsed = self._elapsed()
            print("\n" + "=" * 50)
            print("  RESULTS")
            print("=" * 50)
            print(f"  Time    : {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Yellow  : {self.tracker.yellow_count} x 50 = "
                  f"{self.tracker.yellow_count * 50} pts")
            print(f"  Red     : {self.tracker.red_count} x 100 = "
                  f"{self.tracker.red_count * 100} pts")
            print(f"  Total   : {self.tracker.score()} pts")
            print("=" * 50)
            print("[LAND] Landing...")
            await self.drone.land()
            print("[DONE]")


async def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_path:
        print(f"[CONFIG] YOLO model: {model_path}")
    else:
        print("[CONFIG] No model — using HSV colour detection")
        print("         Usage: python3 qualifier_main.py barrels.pt\n")

    mission = QualifierMission(model_path)
    try:
        await mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] Keyboard interrupt")
        try:
            await mission.drone.land()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
