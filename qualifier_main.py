# qualifier_main.py

import asyncio
import math
import sys
import time
from enum import Enum

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
ZONE1_N = (-4.0, 16.0)
ZONE1_E = (-12.0, 16.0)
# Zone 2 — upper-left arm
ZONE2_N = (16.0, 32.0)
ZONE2_E = (-12.0, 0.0)
# Zone 3 — upper-right arm
ZONE3_N = (16.0, 32.0)
ZONE3_E = (12.0, 28.0)
# Inaccessible cutout between the two upper arms
FORBIDDEN_N = (16.0, 36.0)
FORBIDDEN_E = (0.0, 12.0)

WALL_MARGIN = 2.5

ALT_YELLOW = 1.8
ALT_RED = 4.5

ROW_SPACING_LOW = 3.0
ROW_SPACING_HIGH = 5.0

CONTROL_HZ = 20.0
ARRIVAL_RADIUS = 1.0
ARRIVAL_ALT = 0.5
MISSION_LIMIT = 600.0

LOOK_AHEAD = 1.5
W_AVOID = 0.5
W_MEM_AVOID = 0.3

SAFE_DIST = 3.0
CRIT_DIST = 1.5

MAP_RETENTION_M = 15.0
MAP_INFLUENCE_M = 4.5
MAP_Z_MAX = 10.0

STUCK_TIMEOUT_S = 10.0
STUCK_DIST_M = 0.4
STUCK_ESCAPE_M = 2.5

DETECT_CONFIRM = 2
MERGE_DIST = 3.0

CELL_SIZE = 2.0
GRID_N_ORIGIN = ZONE1_N[0]
GRID_E_ORIGIN = ZONE1_E[0]
GRID_N_CELLS = 18
GRID_E_CELLS = 20

STABILIZE_TIMEOUT = 30.0

CAM_K = np.array([
    [433.0, 0.0, 320.0],
    [0.0, 433.0, 240.0],
    [0.0, 0.0, 1.0],
])


class MissionState(Enum):
    INIT = "INIT"
    CONNECT = "CONNECT"
    TAKEOFF = "TAKEOFF"
    STABILIZE = "STABILIZE"
    STARTUP_SCAN = "STARTUP_SCAN"
    EXPLORE = "EXPLORE"
    SCAN = "SCAN"
    ESCAPE = "ESCAPE"
    DONE = "DONE"
    LAND = "LAND"


class GridCell:
    __slots__ = ("visited_count", "last_visit_time", "scan_done", "blocked")

    def __init__(self):
        self.visited_count = 0
        self.last_visit_time = 0.0
        self.scan_done = False
        self.blocked = False


class QualifierMission:
    def __init__(self, model_path=""):
        self.drone = Drone()
        self.depth_rx = DepthReceiver(DEPTH_TOPIC)
        self.detector = BarrelDetector(model_path)
        self.tracker = DetectionTracker(MERGE_DIST)
        self.state = SharedState()
        self.stop_evt = asyncio.Event()

        self.planner = AvoidancePlanner(
            K=CAM_K,
            width=640,
            height=480,
            safe_distance=SAFE_DIST,
            critical_distance=CRIT_DIST,
        )

        self.mapper = GlobalMapper(
            K=CAM_K,
            cam_height=ALT_YELLOW,
            obs_h_min=0.1,
            obs_h_max=2.0,
            z_min=0.3,
            z_max=MAP_Z_MAX,
            yaw_in_degrees=True,
            yaw_clockwise=True,
            yaw_smoothing=0.8,
        )

        self._state = MissionState.INIT
        self._phase = "YELLOW"

        self._start_time = None
        self._waypoints = []
        self._wp_idx = 0

        self._origin_n = 0.0
        self._origin_e = 0.0

        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = 0.0
        self._stuck_ref_e = 0.0

        self._yellow_streak = 0
        self._red_streak = 0

        self._grid = None
        self._last_cell = (-1, -1)

    def _elapsed(self):
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self):
        return max(0.0, MISSION_LIMIT - self._elapsed())

    def _pose(self):
        if self.state.latest_position is None:
            return {
                "north": 0.0,
                "east": 0.0,
                "down": 0.0,
                "yaw": 0.0,
                "yaw_deg": 0.0,
            }
        return {
            "north": float(self.state.latest_position.north_m),
            "east": float(self.state.latest_position.east_m),
            "down": float(self.state.latest_position.down_m),
            "yaw": math.radians(float(self.state.latest_yaw or 0.0)),
            "yaw_deg": float(self.state.latest_yaw or 0.0),
        }

    @property
    def _n_min(self):
        return self._origin_n + ZONE1_N[0] + WALL_MARGIN

    @property
    def _n_max(self):
        return self._origin_n + ZONE2_N[1] - WALL_MARGIN

    @property
    def _e_min(self):
        return self._origin_e + ZONE1_E[0] + WALL_MARGIN

    @property
    def _e_max(self):
        return self._origin_e + ZONE3_E[1] - WALL_MARGIN

    def _in_forbidden(self, n, e, margin=0.0):
        fn0 = self._origin_n + FORBIDDEN_N[0] - margin
        fn1 = self._origin_n + FORBIDDEN_N[1] + margin
        fe0 = self._origin_e + FORBIDDEN_E[0] - margin
        fe1 = self._origin_e + FORBIDDEN_E[1] + margin
        return fn0 <= n <= fn1 and fe0 <= e <= fe1

    def _clamp(self, n, e):
        n = max(self._n_min, min(self._n_max, n))
        e = max(self._e_min, min(self._e_max, e))
        if self._in_forbidden(n, e):
            if e < self._origin_e + FORBIDDEN_E[0] + (FORBIDDEN_E[1] - FORBIDDEN_E[0]) / 2.0:
                e = self._origin_e + FORBIDDEN_E[0] - WALL_MARGIN
            else:
                e = self._origin_e + FORBIDDEN_E[1] + WALL_MARGIN
        return n, e

    def _in_arena(self, n, e):
        in_z1 = (
            self._origin_n + ZONE1_N[0] <= n <= self._origin_n + ZONE1_N[1]
            and self._origin_e + ZONE1_E[0] <= e <= self._origin_e + ZONE1_E[1]
        )
        in_z2 = (
            self._origin_n + ZONE2_N[0] <= n <= self._origin_n + ZONE2_N[1]
            and self._origin_e + ZONE2_E[0] <= e <= self._origin_e + ZONE2_E[1]
        )
        in_z3 = (
            self._origin_n + ZONE3_N[0] <= n <= self._origin_n + ZONE3_N[1]
            and self._origin_e + ZONE3_E[0] <= e <= self._origin_e + ZONE3_E[1]
        )
        return in_z1 or in_z2 or in_z3

    def _valid_cell(self, ci, cj):
        return 0 <= ci < GRID_N_CELLS and 0 <= cj < GRID_E_CELLS

    def _ned_to_cell(self, north, east):
        ci = int((north - self._origin_n - GRID_N_ORIGIN) // CELL_SIZE)
        cj = int((east - self._origin_e - GRID_E_ORIGIN) // CELL_SIZE)
        return ci, cj

    def _cell_to_ned(self, ci, cj):
        north = self._origin_n + GRID_N_ORIGIN + (ci + 0.5) * CELL_SIZE
        east = self._origin_e + GRID_E_ORIGIN + (cj + 0.5) * CELL_SIZE
        return north, east

    def _init_grid(self):
        self._grid = [[GridCell() for _ in range(GRID_E_CELLS)] for _ in range(GRID_N_CELLS)]
        blocked = 0
        for ci in range(GRID_N_CELLS):
            for cj in range(GRID_E_CELLS):
                cn, ce = self._cell_to_ned(ci, cj)
                if not self._in_arena(cn, ce):
                    self._grid[ci][cj].blocked = True
                    self._grid[ci][cj].scan_done = True
                    blocked += 1
        navigable = GRID_N_CELLS * GRID_E_CELLS - blocked
        print(
            f"[GRID] {GRID_N_CELLS}×{GRID_E_CELLS} grid  "
            f"cell={CELL_SIZE}m  navigable={navigable}  blocked={blocked}"
        )

    def _update_grid(self, pose):
        if self._grid is None:
            return False
        ci, cj = self._ned_to_cell(pose["north"], pose["east"])
        if not self._valid_cell(ci, cj):
            return False
        cell = self._grid[ci][cj]
        if cell.blocked:
            return False

        prev_cell = self._last_cell
        self._last_cell = (ci, cj)

        first_entry = (prev_cell != (ci, cj)) and (cell.visited_count == 0)
        cell.visited_count += 1
        cell.last_visit_time = time.monotonic()

        if first_entry and not cell.scan_done:
            if self._yellow_streak > 0 or self._red_streak > 0:
                print(
                    f"[GRID] New cell ({ci},{cj}) + streak "
                    f"Y={self._yellow_streak} R={self._red_streak} → SCAN"
                )
                return True
        return False

    def _coverage_summary(self):
        if self._grid is None:
            return "grid=uninit"
        navigable = 0
        visited = 0
        for row in self._grid:
            for cell in row:
                if not cell.blocked:
                    navigable += 1
                    if cell.visited_count > 0:
                        visited += 1
        pct = int(100 * visited / navigable) if navigable else 0
        return f"grid={visited}/{navigable}({pct}%)"

    async def _wait_stabilize(self):
        print(
            f"[STABILIZE] Waiting for pose, yaw, and depth frame "
            f"(timeout={STABILIZE_TIMEOUT:.0f}s)..."
        )
        t0 = time.monotonic()
        while True:
            pose_ok = self.state.latest_position is not None
            yaw_ok = self.state.latest_yaw is not None
            depth_ok = self.depth_rx.get_frame() is not None

            if pose_ok and yaw_ok and depth_ok:
                print("[STABILIZE] All sensors valid — proceeding.")
                return True

            elapsed = time.monotonic() - t0
            if elapsed > STABILIZE_TIMEOUT:
                print(
                    f"[STABILIZE] Timeout after {STABILIZE_TIMEOUT:.0f}s — "
                    f"pose={'ok' if pose_ok else 'MISSING'}  "
                    f"yaw={'ok' if yaw_ok else 'MISSING'}  "
                    f"depth={'ok' if depth_ok else 'MISSING'}"
                )
                return False

            await asyncio.sleep(0.1)

    # keep your existing implementations below unchanged:
    # _start_phase
    # _reset_stuck
    # _run_detection
    # _current_wp
    # _advance_wp
    # _arrived
    # _check_stuck
    # _escape_stuck
    # _is_near_wall
    # _compute_setpoint
    # _tick_explore
    # _tick_scan
    # _tick_escape
    # _control_loop
    # _startup_scan

    async def run(self):
        print("=" * 50)
        print("  RoboVerse 2026 Qualifier")
        print("=" * 50)
        print("Pre-flight checklist:")
        print("  1) Start simulator: ./start_px4.sh  (choose x500_vision)")
        print("  2) In PX4 terminal: commander set_ekf_origin 47.397742 8.545594 488.0")
        print("  3) Then run this script\n")

        monitor = None

        self._state = MissionState.CONNECT
        print(f"[FSM] {self._state.value}")
        try:
            await self.drone.connect()
        except Exception as e:
            print(f"[FSM] Connect failed: {e}")
            print("[FSM] CONNECT → LAND (abort)")
            self._state = MissionState.LAND
            return
        print("[INIT] Connected")
        await asyncio.sleep(3)

        self._state = MissionState.TAKEOFF
        print(f"[FSM] {self._state.value}")
        await asyncio.sleep(2)
        print("[INIT] Arming and taking off...")
        try:
            await self.drone.arm_and_takeoff()
            self.state.is_armed = True
        except Exception as e:
            print(f"[FSM] Takeoff failed: {e}")
            print("[FSM] TAKEOFF → LAND (aborting safely)")
            self._state = MissionState.LAND
            if self.state.is_armed:
                await self.drone.land()
                self.state.is_armed = False
            return

        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )

        # wait briefly for telemetry to populate after starting the monitor
        for _ in range(50):
            if self.state.latest_position is not None and self.state.latest_yaw is not None:
                break
            await asyncio.sleep(0.1)

        if self.state.latest_position is None or self.state.latest_yaw is None:
            print("[FSM] Telemetry did not populate after takeoff")
            self._state = MissionState.LAND
            self.stop_evt.set()
            if monitor is not None:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
            await self.drone.land()
            return

        p = self._pose()
        self._origin_n = p["north"]
        self._origin_e = p["east"]
        self._init_grid()

        print(f"[INIT] Spawn NED: N={self._origin_n:.2f} E={self._origin_e:.2f}")
        print(
            f"[INIT] Zone 1 (main floor)    "
            f"N [{self._origin_n + ZONE1_N[0]:.1f} → {self._origin_n + ZONE1_N[1]:.1f}]  "
            f"E [{self._origin_e + ZONE1_E[0]:.1f} → {self._origin_e + ZONE1_E[1]:.1f}]"
        )
        print(
            f"[INIT] Zone 2 (upper-left)    "
            f"N [{self._origin_n + ZONE2_N[0]:.1f} → {self._origin_n + ZONE2_N[1]:.1f}]  "
            f"E [{self._origin_e + ZONE2_E[0]:.1f} → {self._origin_e + ZONE2_E[1]:.1f}]"
        )
        print(
            f"[INIT] Zone 3 (upper-right)   "
            f"N [{self._origin_n + ZONE3_N[0]:.1f} → {self._origin_n + ZONE3_N[1]:.1f}]  "
            f"E [{self._origin_e + ZONE3_E[0]:.1f} → {self._origin_e + ZONE3_E[1]:.1f}]"
        )

        self._state = MissionState.STABILIZE
        print(f"[FSM] {self._state.value}")
        sensors_ok = await self._wait_stabilize()
        if not sensors_ok:
            print("[FSM] Sensor gate failed — aborting mission safely.")
            self._state = MissionState.LAND
            self.stop_evt.set()
            if monitor is not None:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
            await self.drone.land()
            return

        self._state = MissionState.STARTUP_SCAN
        print(f"[FSM] {self._state.value}")
        await self._startup_scan()

        self._start_phase("YELLOW")
        self._start_time = time.monotonic()
        self._reset_stuck()
        self._state = MissionState.EXPLORE
        print(f"[FSM] {self._state.value} — 10:00 countdown started\n")

        try:
            await self._control_loop()
        except asyncio.CancelledError:
            print("\n[ABORT] Cancelled")
        finally:
            self.stop_evt.set()
            if monitor is not None:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass

            elapsed = self._elapsed()
            print("\n" + "=" * 50)
            print("  RESULTS")
            print("=" * 50)
            print(f"  Time    : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
            print(f"  Yellow  : {self.tracker.yellow_count} x 50 = {self.tracker.yellow_count * 50} pts")
            print(f"  Red     : {self.tracker.red_count} x 100 = {self.tracker.red_count * 100} pts")
            print(f"  Total   : {self.tracker.score()} pts")
            print("=" * 50)
            self._state = MissionState.LAND
            print(f"[FSM] {self._state.value}")
            print("[LAND] Landing...")
            try:
                await self.drone.land()
            finally:
                self.state.is_armed = False
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