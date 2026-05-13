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
ROW_SPACING_LOW  = 3.0
ROW_SPACING_HIGH = 5.0

CONTROL_HZ     = 20.0
ARRIVAL_RADIUS = 1.0   # horizontal arrival threshold (m)
ARRIVAL_ALT    = 0.5   # vertical arrival threshold (m)
MISSION_LIMIT  = 600.0 # s — 10 min hard cap

# Virtual target blending
LOOK_AHEAD  = 1.5  # m
W_AVOID     = 0.5  # depth-camera avoidance weight
W_MEM_AVOID = 0.3  # memory map avoidance weight

# Avoidance
SAFE_DIST = 3.0
CRIT_DIST = 1.5

# Map memory
MAP_RETENTION_M  = 15.0
MAP_INFLUENCE_M  = 4.5
MAP_Z_MAX        = 10.0

# Stuck detection
STUCK_TIMEOUT_S = 10.0
STUCK_DIST_M    = 0.4
STUCK_ESCAPE_M  = 2.5

# Detection confirmation
DETECT_CONFIRM = 2

MERGE_DIST = 3.0

# Visited grid
CELL_SIZE     = 2.0          # m — grid cell resolution
GRID_N_ORIGIN = ZONE1_N[0]  # -4.0  m — south edge relative to spawn NED
GRID_E_ORIGIN = ZONE1_E[0]  # -12.0 m — west  edge relative to spawn NED
GRID_N_CELLS  = 18           # ceil((ZONE2_N[1] - ZONE1_N[0]) / CELL_SIZE) = 36/2
GRID_E_CELLS  = 20           # ceil((ZONE3_E[1] - ZONE1_E[0]) / CELL_SIZE) = 40/2

# Stabilize sensor gate
STABILIZE_TIMEOUT = 30.0  # s — abort if sensors not valid within this window

CAM_K = np.array([[433.0, 0.0, 320.0],
                  [0.0,   433.0, 240.0],
                  [0.0,   0.0,   1.0]])


# ---------------------------------------------------------------------------
# Mission state machine
# ---------------------------------------------------------------------------
class MissionState(Enum):
    INIT         = "INIT"
    CONNECT      = "CONNECT"
    TAKEOFF      = "TAKEOFF"
    STABILIZE    = "STABILIZE"
    STARTUP_SCAN = "STARTUP_SCAN"
    EXPLORE      = "EXPLORE"
    SCAN         = "SCAN"     # stub — full logic added in Part 2
    ESCAPE       = "ESCAPE"
    DONE         = "DONE"
    LAND         = "LAND"


# ---------------------------------------------------------------------------
# Visited grid cell
# ---------------------------------------------------------------------------
class GridCell:
    __slots__ = ("visited_count", "last_visit_time", "scan_done", "blocked")

    def __init__(self):
        self.visited_count   = 0
        self.last_visit_time = 0.0
        self.scan_done       = False
        self.blocked         = False   # permanently out of arena — never a target


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
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

        # State machine — _state controls what the drone is doing.
        # _phase controls which altitude pass (YELLOW / RED), kept separate.
        self._state  = MissionState.INIT
        self._phase  = "YELLOW"

        self._start_time = None
        self._waypoints  = []
        self._wp_idx     = 0

        # NED origin recorded at takeoff
        self._origin_n = 0.0
        self._origin_e = 0.0

        # Stuck tracking
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = 0.0
        self._stuck_ref_e = 0.0

        # Consecutive detection streak counters
        self._yellow_streak = 0
        self._red_streak    = 0

        # Visited grid — built after takeoff origin is recorded
        self._grid      = None       # GridCell[GRID_N_CELLS][GRID_E_CELLS]
        self._last_cell = (-1, -1)   # previous cell index, for entry detection

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def _elapsed(self):
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _time_left(self):
        return max(0.0, MISSION_LIMIT - self._elapsed())

    # ------------------------------------------------------------------
    # Pose snapshot
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
    # Arena boundary helpers
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
        if self._in_forbidden(n, e):
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
    # Sweep waypoint generation
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
    # Startup 360° scan — rotate in place and seed GlobalMapper
    # ------------------------------------------------------------------
    async def _startup_scan(self):
        p = self._pose()
        hold_n, hold_e, hold_d = p["north"], p["east"], p["down"]
        print("[SCAN] Starting 360° horizon scan — holding position")

        for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
            await self.drone.rotate_to_yaw(float(yaw))
            await asyncio.sleep(1.2)
            depth = self.depth_rx.get_frame()
            pose  = self._pose()
            if depth is not None:
                self.mapper.update_frame(depth, pose)
            await self.drone.send_position_setpoint(hold_n, hold_e, hold_d, float(yaw))

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
                  f"T={self._time_left():.0f}s  {self.tracker.summary()}  "
                  f"{self._coverage_summary()}")

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
    # Stuck detection
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
            self._stuck_ref_n = p["north"]
            self._stuck_ref_e = p["east"]
            self._stuck_timer = time.monotonic()
            return False
        return time.monotonic() - self._stuck_timer > STUCK_TIMEOUT_S

    async def _escape_stuck(self):
        p = self._pose()
        print(f"[ESCAPE] at N={p['north']:.1f} E={p['east']:.1f} — trying escape directions")
        wp = self._current_wp()

        for delta in [90, -90, 180, 45]:
            yaw = (p["yaw_deg"] + delta) % 360.0
            cn  = p["north"] + STUCK_ESCAPE_M * math.cos(math.radians(yaw))
            ce  = p["east"]  + STUCK_ESCAPE_M * math.sin(math.radians(yaw))
            cn, ce = self._clamp(cn, ce)
            if not self._is_near_wall(cn, ce):
                print(f"[ESCAPE] escaping → yaw={yaw:.0f}°")
                alt_d = wp[2] if wp else p["down"]
                await self.drone.rotate_to_yaw(yaw)
                await self.drone.send_position_setpoint(cn, ce, alt_d, yaw)
                await asyncio.sleep(3.0)
                break

        self._reset_stuck()

    # ------------------------------------------------------------------
    # Virtual target blending — goal + avoidance + memory
    # ------------------------------------------------------------------
    def _compute_setpoint(self, pose, target_n, target_e, target_d, depth):
        cur_n, cur_e = pose["north"], pose["east"]

        # Emergency: near wall/forbidden — push to zone 1 centre
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

        # Memory-map repulsion from GlobalMapper
        mem_n, mem_e = self.mapper.get_repulsion_vector(cur_n, cur_e, MAP_INFLUENCE_M)

        # Blend: goal + avoidance + memory (goal suppressed when fully blocked)
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

        raw_n = cur_n + LOOK_AHEAD * blend_n
        raw_e = cur_e + LOOK_AHEAD * blend_e
        send_n, send_e = self._clamp(raw_n, raw_e)

        yaw_deg = math.degrees(math.atan2(blend_e, blend_n))
        return send_n, send_e, target_d, yaw_deg

    # ------------------------------------------------------------------
    # Visited grid — 2m cells covering the full L-shaped arena
    # ------------------------------------------------------------------
    def _in_arena(self, n, e):
        """True if NED point is inside one of the three valid flight zones."""
        in_z1 = (self._origin_n + ZONE1_N[0] <= n <= self._origin_n + ZONE1_N[1] and
                  self._origin_e + ZONE1_E[0] <= e <= self._origin_e + ZONE1_E[1])
        in_z2 = (self._origin_n + ZONE2_N[0] <= n <= self._origin_n + ZONE2_N[1] and
                  self._origin_e + ZONE2_E[0] <= e <= self._origin_e + ZONE2_E[1])
        in_z3 = (self._origin_n + ZONE3_N[0] <= n <= self._origin_n + ZONE3_N[1] and
                  self._origin_e + ZONE3_E[0] <= e <= self._origin_e + ZONE3_E[1])
        return in_z1 or in_z2 or in_z3

    def _valid_cell(self, ci, cj):
        return 0 <= ci < GRID_N_CELLS and 0 <= cj < GRID_E_CELLS

    def _ned_to_cell(self, north, east):
        """Convert NED position to (row, col) grid indices."""
        ci = int((north - self._origin_n - GRID_N_ORIGIN) / CELL_SIZE)
        cj = int((east  - self._origin_e - GRID_E_ORIGIN) / CELL_SIZE)
        return ci, cj

    def _cell_to_ned(self, ci, cj):
        """Return NED centre coordinates of cell (ci, cj)."""
        north = self._origin_n + GRID_N_ORIGIN + (ci + 0.5) * CELL_SIZE
        east  = self._origin_e + GRID_E_ORIGIN + (cj + 0.5) * CELL_SIZE
        return north, east

    def _init_grid(self):
        """
        Build the visited grid after spawn NED origin is known.
        Cells outside the L-shaped arena are pre-marked blocked so the
        exploration logic never targets them.
        """
        self._grid = [[GridCell() for _ in range(GRID_E_CELLS)]
                      for _ in range(GRID_N_CELLS)]
        blocked = 0
        for ci in range(GRID_N_CELLS):
            for cj in range(GRID_E_CELLS):
                cn, ce = self._cell_to_ned(ci, cj)
                if not self._in_arena(cn, ce):
                    self._grid[ci][cj].blocked   = True
                    self._grid[ci][cj].scan_done = True
                    blocked += 1
        navigable = GRID_N_CELLS * GRID_E_CELLS - blocked
        print(f"[GRID] {GRID_N_CELLS}×{GRID_E_CELLS} grid  "
              f"cell={CELL_SIZE}m  navigable={navigable}  blocked={blocked}")

    def _update_grid(self, pose):
        """
        Called every EXPLORE tick. Marks the current cell visited.
        Returns True if a SCAN should be triggered:
          - first entry into this cell, AND
          - detector has an active streak (barrel may be nearby).
        """
        if self._grid is None:
            return False
        ci, cj = self._ned_to_cell(pose["north"], pose["east"])
        if not self._valid_cell(ci, cj):
            return False
        cell = self._grid[ci][cj]
        if cell.blocked:
            return False

        prev_cell       = self._last_cell
        self._last_cell = (ci, cj)

        first_entry           = (prev_cell != (ci, cj)) and (cell.visited_count == 0)
        cell.visited_count   += 1
        cell.last_visit_time  = time.monotonic()

        if first_entry and not cell.scan_done:
            if self._yellow_streak > 0 or self._red_streak > 0:
                print(f"[GRID] New cell ({ci},{cj}) + streak "
                      f"Y={self._yellow_streak} R={self._red_streak} → SCAN")
                return True
        return False

    def _coverage_summary(self):
        """Short string showing how many navigable cells have been visited."""
        if self._grid is None:
            return "grid=uninit"
        navigable = visited = 0
        for row in self._grid:
            for cell in row:
                if not cell.blocked:
                    navigable += 1
                    if cell.visited_count > 0:
                        visited += 1
        pct = int(100 * visited / navigable) if navigable else 0
        return f"grid={visited}/{navigable}({pct}%)"

    # ------------------------------------------------------------------
    # STABILIZE: block until pose + yaw + depth are all valid
    # ------------------------------------------------------------------
    async def _wait_stabilize(self):
        print(f"[STABILIZE] Waiting for pose, yaw, and depth frame "
              f"(timeout={STABILIZE_TIMEOUT:.0f}s)...")
        t0 = time.monotonic()
        while True:
            pose_ok  = self.state.latest_position is not None
            yaw_ok   = self.state.latest_yaw is not None
            depth_ok = self.depth_rx.get_frame() is not None

            if pose_ok and yaw_ok and depth_ok:
                print("[STABILIZE] All sensors valid — proceeding.")
                return True

            elapsed = time.monotonic() - t0
            if elapsed > STABILIZE_TIMEOUT:
                print(f"[STABILIZE] Timeout after {STABILIZE_TIMEOUT:.0f}s — "
                      f"pose={'ok' if pose_ok else 'MISSING'}  "
                      f"yaw={'ok' if yaw_ok else 'MISSING'}  "
                      f"depth={'ok' if depth_ok else 'MISSING'}")
                return False

            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # State tick: EXPLORE
    # ------------------------------------------------------------------
    async def _tick_explore(self):
        if self._time_left() < 10.0:
            print("[MISSION] Time limit reached.")
            self._state = MissionState.DONE
            return

        self._run_detection()

        wp = self._current_wp()
        if wp is None:
            if self._phase == "YELLOW":
                print(f"\n[PHASE 1 DONE] {self.tracker.summary()}")
                if self._time_left() > 90:
                    self._start_phase("RED")
                else:
                    print("[MISSION] Not enough time for red sweep.")
                    self._state = MissionState.DONE
            else:
                print(f"\n[PHASE 2 DONE] {self.tracker.summary()}")
                self._state = MissionState.DONE
            return

        pose  = self._pose()
        cur_n = pose["north"]
        cur_e = pose["east"]

        # Update visited grid — trigger SCAN on first cell entry with active streak
        if self._update_grid(pose):
            print("[FSM] EXPLORE → SCAN (new cell + active detection streak)")
            self._state = MissionState.SCAN
            return

        # Emergency boundary recovery
        if self._is_near_wall(cur_n, cur_e, extra=0.5):
            safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
            safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0
            await self.drone.send_position_setpoint(safe_n, safe_e, wp[2], 0.0)
            return

        if self._arrived(wp):
            self._advance_wp()
            return

        if self._check_stuck():
            print(f"[FSM] Stuck detected → EXPLORE → ESCAPE")
            self._state = MissionState.ESCAPE
            return

        target_n, target_e, target_d = wp
        depth = self.depth_rx.get_frame()

        if depth is not None:
            self.mapper.update_frame(depth, pose)
            self.mapper.prune(pose["north"], pose["east"], MAP_RETENTION_M)

        send_n, send_e, send_d, yaw_deg = self._compute_setpoint(
            pose, target_n, target_e, target_d, depth
        )
        await self.drone.send_position_setpoint(
            north=send_n, east=send_e, down=send_d, yaw_deg=yaw_deg
        )

    # ------------------------------------------------------------------
    # State tick: SCAN (stub — Part 2 adds visited-grid trigger)
    # ------------------------------------------------------------------
    async def _tick_scan(self):
        print("[FSM] SCAN — stub: running 360° scan then returning to EXPLORE")
        await self._startup_scan()
        # Mark current cell scan_done so we don't re-trigger on re-entry
        if self._grid is not None:
            pose = self._pose()
            ci, cj = self._ned_to_cell(pose["north"], pose["east"])
            if self._valid_cell(ci, cj):
                self._grid[ci][cj].scan_done = True
                print(f"[GRID] Cell ({ci},{cj}) marked scan_done")
        self._state = MissionState.EXPLORE

    # ------------------------------------------------------------------
    # State tick: ESCAPE
    # ------------------------------------------------------------------
    async def _tick_escape(self):
        await self._escape_stuck()
        print("[FSM] ESCAPE → EXPLORE")
        self._state = MissionState.EXPLORE

    # ------------------------------------------------------------------
    # Dispatch-table control loop
    # ------------------------------------------------------------------
    async def _control_loop(self):
        dt = 1.0 / CONTROL_HZ

        _dispatch = {
            MissionState.EXPLORE: self._tick_explore,
            MissionState.SCAN:    self._tick_scan,
            MissionState.ESCAPE:  self._tick_escape,
        }

        print(f"[FSM] Control loop started — state: {self._state.value}")

        while self._state not in (MissionState.DONE, MissionState.LAND):
            t0 = time.monotonic()

            handler = _dispatch.get(self._state)
            if handler is not None:
                await handler()
            else:
                print(f"[FSM] No handler for state {self._state.value} — idling one tick")

            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

        print(f"[FSM] Control loop exited — final state: {self._state.value}")

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

        # ── CONNECT ──────────────────────────────────────────────────
        self._state = MissionState.CONNECT
        print(f"[FSM] {self._state.value}")
        await self.drone.connect()
        print("[INIT] Connected")
        await asyncio.sleep(3)

        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )

        # ── TAKEOFF ──────────────────────────────────────────────────
        self._state = MissionState.TAKEOFF
        print(f"[FSM] {self._state.value}")
        await asyncio.sleep(2)
        print("[INIT] Arming and taking off...")
        try:
            await self.drone.arm_and_takeoff()
        except Exception as e:
            print(f"[FSM] Takeoff failed: {e}")
            print("[FSM] TAKEOFF → LAND (aborting safely)")
            self._state = MissionState.LAND
            self.stop_evt.set()
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
        self._init_grid()    # build visited grid now that origin is known
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

        # ── STABILIZE ────────────────────────────────────────────────
        self._state = MissionState.STABILIZE
        print(f"[FSM] {self._state.value}")
        sensors_ok = await self._wait_stabilize()
        if not sensors_ok:
            print("[FSM] Sensor gate failed — aborting mission safely.")
            self._state = MissionState.LAND
            self.stop_evt.set()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
            await self.drone.land()
            return

        # ── STARTUP_SCAN ─────────────────────────────────────────────
        self._state = MissionState.STARTUP_SCAN
        print(f"[FSM] {self._state.value}")
        await self._startup_scan()

        # ── EXPLORE ──────────────────────────────────────────────────
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
            self._state = MissionState.LAND
            print(f"[FSM] {self._state.value}")
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
