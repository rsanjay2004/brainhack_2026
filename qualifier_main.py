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

DEPTH_TOPIC = "/depth_camera"

ARENA_W          = 40.0
ARENA_D          = 40.0
WALL_MARGIN      = 2.0

ALT_YELLOW       = 2.0    # m — phase 1: ground-level barrels
ALT_RED          = 4.5    # m — phase 2: elevated barrels

ROW_SPACING_LOW  = 3.5    # m
ROW_SPACING_HIGH = 5.0    # m

CONTROL_HZ       = 20.0
ARRIVAL_RADIUS   = 1.2    # m
MISSION_LIMIT    = 600.0  # s

LOOK_AHEAD       = 2.0    # m — virtual target projection distance
W_AVOID          = 0.6    # blend weight: 1.0 = full avoidance, 0.0 = pure goal

SAFE_DIST        = 3.5    # m
CRIT_DIST        = 1.2    # m

STUCK_TIMEOUT_S  = 12.0   # s
STUCK_DIST_M     = 0.5    # m
STUCK_ESCAPE_M   = 3.0    # m

MERGE_DIST       = 3.0    # m — same barrel dedup radius

CAM_K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])


class QualifierMission:

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

        self._start_time = None
        self._waypoints  = []
        self._wp_idx     = 0
        self._phase      = "YELLOW"

        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = 0.0
        self._stuck_ref_e = 0.0

    def _elapsed(self):
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self):
        return max(0.0, MISSION_LIMIT - self._elapsed())

    def _pose(self):
        if self.state.latest_position is None:
            return {"north": 0.0, "east": 0.0, "down": 0.0, "yaw": 0.0, "yaw_deg": 0.0}
        return {
            "north":   float(self.state.latest_position.north_m),
            "east":    float(self.state.latest_position.east_m),
            "down":    float(self.state.latest_position.down_m),
            "yaw":     math.radians(float(self.state.latest_yaw or 0.0)),
            "yaw_deg": float(self.state.latest_yaw or 0.0),
        }

    def _build_sweep(self, altitude, row_spacing):
        down = -altitude
        lo   = WALL_MARGIN
        hi_n = ARENA_D - WALL_MARGIN
        hi_e = ARENA_W - WALL_MARGIN
        cols = list(np.arange(lo, hi_e + 1e-6, row_spacing))
        wps  = []
        for i, east in enumerate(cols):
            if i % 2 == 0:
                wps += [(lo, east, down), (hi_n, east, down)]
            else:
                wps += [(hi_n, east, down), (lo, east, down)]
        return wps

    def _run_detection(self):
        result = self.detector.detect()
        if not (result["yellow"] or result["red"]):
            return
        p = self._pose()
        n, e = p["north"], p["east"]
        if result["yellow"] and self.tracker.try_add_yellow(n, e):
            print(f"[DETECT] YELLOW #{self.tracker.yellow_count}  N={n:.1f} E={e:.1f}  {self.tracker.summary()}")
        if result["red"] and self.tracker.try_add_red(n, e):
            print(f"[DETECT] RED #{self.tracker.red_count}  N={n:.1f} E={e:.1f}  {self.tracker.summary()}")

    def _current_wp(self):
        return self._waypoints[self._wp_idx] if self._wp_idx < len(self._waypoints) else None

    def _arrived(self, wp):
        p = self._pose()
        return math.hypot(p["north"] - wp[0], p["east"] - wp[1]) < ARRIVAL_RADIUS

    def _advance_wp(self):
        self._wp_idx += 1
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP {self._wp_idx}/{len(self._waypoints)}  N={wp[0]:.1f} E={wp[1]:.1f} "
                  f"Alt={-wp[2]:.1f}m  T={self._time_left():.0f}s  {self.tracker.summary()}")

    def _start_phase(self, phase):
        self._phase = phase
        if phase == "YELLOW":
            self._waypoints = self._build_sweep(ALT_YELLOW, ROW_SPACING_LOW)
            print(f"\n[PHASE 1] Yellow sweep  alt={ALT_YELLOW}m  {len(self._waypoints)} waypoints")
        else:
            self._waypoints = self._build_sweep(ALT_RED, ROW_SPACING_HIGH)
            print(f"\n[PHASE 2] Red sweep  alt={ALT_RED}m  {len(self._waypoints)} waypoints")
        self._wp_idx = 0
        self._reset_stuck()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP 1/{len(self._waypoints)}  N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m")

    def _reset_stuck(self):
        p = self._pose()
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = p["north"]
        self._stuck_ref_e = p["east"]

    def _is_stuck(self):
        if time.monotonic() - self._stuck_timer < STUCK_TIMEOUT_S:
            return False
        p = self._pose()
        return math.hypot(p["north"] - self._stuck_ref_n, p["east"] - self._stuck_ref_e) < STUCK_DIST_M

    async def _escape_stuck(self):
        p = self._pose()
        escape_yaw = (p["yaw_deg"] + 90.0) % 360.0
        print(f"[STUCK] Rotating to {escape_yaw:.0f}° and pushing forward")
        await self.drone.rotate_to_yaw(escape_yaw)
        esc_n = p["north"] + STUCK_ESCAPE_M * math.cos(math.radians(escape_yaw))
        esc_e = p["east"]  + STUCK_ESCAPE_M * math.sin(math.radians(escape_yaw))
        wp    = self._current_wp()
        esc_d = wp[2] if wp else p["down"]
        await self.drone.send_position_setpoint(esc_n, esc_e, esc_d, escape_yaw)
        await asyncio.sleep(3.0)
        self._reset_stuck()

    def _compute_virtual_target(self, pose, target_n, target_e, target_d, depth):
        cur_n, cur_e = pose["north"], pose["east"]

        dn = target_n - cur_n
        de = target_e - cur_e
        dist = math.hypot(dn, de)
        goal_n = (dn / dist) if dist > 1e-3 else 1.0
        goal_e = (de / dist) if dist > 1e-3 else 0.0

        avoid_n, avoid_e, blocked = 0.0, 0.0, False
        if depth is not None:
            av_n, av_e, _, info = self.planner.compute_position_ned(depth, pose, step_size=1.0)
            blocked = info["blocked"]
            av_dist = math.hypot(av_n - cur_n, av_e - cur_e)
            if av_dist > 1e-3:
                avoid_n = (av_n - cur_n) / av_dist
                avoid_e = (av_e - cur_e) / av_dist
            if blocked:
                cl = info["clearance"]
                print(f"[AVOID] L={cl['left']:.1f} C={cl['center']:.1f} R={cl['right']:.1f}")

        if blocked:
            blend_n, blend_e = avoid_n, avoid_e
        else:
            blend_n = goal_n + W_AVOID * avoid_n
            blend_e = goal_e + W_AVOID * avoid_e

        mag = math.hypot(blend_n, blend_e)
        if mag > 1e-3:
            blend_n /= mag
            blend_e /= mag
        else:
            blend_n, blend_e = goal_n, goal_e

        send_n  = cur_n + LOOK_AHEAD * blend_n
        send_e  = cur_e + LOOK_AHEAD * blend_e
        yaw_deg = math.degrees(math.atan2(blend_e, blend_n))
        return send_n, send_e, target_d, yaw_deg

    async def _control_loop(self):
        dt = 1.0 / CONTROL_HZ
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

            pose = self._pose()

            if self._arrived(wp):
                self._advance_wp()
                wp = self._current_wp()
                if wp is None:
                    continue

            if self._is_stuck():
                await self._escape_stuck()
                await asyncio.sleep(0)
                continue

            target_n, target_e, target_d = wp
            depth = self.depth_rx.get_frame()
            send_n, send_e, send_d, yaw_deg = self._compute_virtual_target(
                pose, target_n, target_e, target_d, depth
            )

            await self.drone.send_position_setpoint(
                north=send_n, east=send_e, down=send_d, yaw_deg=yaw_deg
            )

            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

    async def run(self):
        print("=" * 50)
        print("  RoboVerse 2026 Qualifier")
        print("=" * 50)
        print("Before running: set EKF origin in PX4 terminal:")
        print("  px4> commander set_ekf_origin 47.397742 8.545594 488.0")
        print()

        await self.drone.connect()
        print("[INIT] Connected")
        await asyncio.sleep(3)

        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )
        await asyncio.sleep(2)

        print("[INIT] Arming and taking off...")
        await self.drone.arm_and_takeoff()
        self._start_time = time.monotonic()
        self._reset_stuck()
        print("[START] Airborne — 10 min countdown\n")

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
            print(f"  Time       : {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Yellow     : {self.tracker.yellow_count} x 50 = {self.tracker.yellow_count * 50} pts")
            print(f"  Red        : {self.tracker.red_count} x 100 = {self.tracker.red_count * 100} pts")
            print(f"  Total      : {self.tracker.score()} pts")
            print("=" * 50)

            print("[LAND] Landing...")
            await self.drone.land()
            print("[DONE]")


async def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_path:
        print(f"[CONFIG] YOLO model: {model_path}")
    else:
        print("[CONFIG] No model given — using colour detection (HSV)")
        print("         To use YOLO: python qualifier_main.py barrels.pt\n")

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
