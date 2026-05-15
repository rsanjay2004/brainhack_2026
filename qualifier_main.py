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
from RRTStarPlanner import RRTStarPlanner

DEPTH_TOPIC = "/depth_camera"

# ── Arena geometry ── swap this entire block when official map is released ───
# All coordinates are spawn-relative NED (N=North offset, E=East offset), meters.
# Drone spawns at approx N=4, E=12 in world frame; geometry is spawn-relative.
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

# Physical wall segments as ((n1,e1),(n2,e2)) pairs, spawn-relative.
# Used for continuous geometric repulsion (works regardless of camera FOV).
ARENA_WALL_SEGS_REL = [
    ((-4.0, -12.0), (-4.0,  16.0)),   # south wall
    ((-4.0,  16.0), (16.0,  16.0)),   # Zone1 east wall
    ((16.0,  16.0), (16.0,  28.0)),   # Zone3 south step (E=16→E=28 at N=16)
    ((16.0,  28.0), (32.0,  28.0)),   # Zone3 east wall
    ((32.0,  28.0), (32.0,  12.0)),   # Zone3 north wall
    ((32.0,  12.0), (16.0,  12.0)),   # cutout right inner wall (N=32→N=16 at E=12)
    ((16.0,  12.0), (16.0,   0.0)),   # cutout bottom (N=16 at E=0→E=12)
    ((16.0,   0.0), (32.0,   0.0)),   # cutout left inner wall (N=16→N=32 at E=0)
    ((32.0,   0.0), (32.0, -12.0)),   # Zone2 north wall
    ((32.0, -12.0), (-4.0, -12.0)),   # west wall
]

WALL_MARGIN  = 2.5   # stay this far from walls at all times
W_WALL       = 0.5   # continuous geometric wall repulsion weight
WALL_INF_M   = 3.5   # wall influence radius (m) > WALL_MARGIN so repulsion starts before margin
# ─────────────────────────────────────────────────────────────────────────────

# RRT* path planning — runs once per waypoint leg in a background thread
RRT_MAX_ITER = 500    # iterations per plan (balances quality vs latency)
RRT_SAFETY_M = 1.0   # obstacle clearance margin (m)
RRT_STEP_M   = 1.5   # tree extension step size (m)

# Crash recovery (Phase 4)
MAX_RESTARTS = 2      # max re-attempts within the 10-minute window

# Altitudes
ALT_YELLOW = 1.8    # low — ground-level yellow barrels visible in lower frame
ALT_RED    = 4.5    # high — elevated red barrels come into camera FOV

# Row spacing — wider than camera sightline since detection runs during flight
ROW_SPACING_LOW  = 5.0   # was 3.0 — at 1.8m alt camera sees ~5m ahead
ROW_SPACING_HIGH = 7.0   # was 5.0 — at 4.5m alt camera sees further

CONTROL_HZ     = 20.0
ARRIVAL_RADIUS = 1.0    # horizontal arrival threshold (m)
ARRIVAL_ALT    = 0.5    # vertical arrival threshold (m)
MISSION_LIMIT  = 600.0  # s — 10 min hard cap

# Virtual target blending
W_AVOID     = 0.5   # depth-camera avoidance weight
W_MEM_AVOID = 0.3   # memory map avoidance weight

# Avoidance
SAFE_DIST = 3.0
CRIT_DIST = 1.5

# Map memory
MAP_RETENTION_M = 15.0
MAP_INFLUENCE_M = 4.5
MAP_Z_MAX = 10.0

# Velocity setpoint limits (Phase 1)
VEL_MAX      = 1.0   # m/s — open space cruise speed
VEL_MIN      = 0.3   # m/s — near obstacles / boundary recovery
YAW_RATE_MAX = 15.0  # deg per control tick — prevents snapping to face a wall
ALT_KP       = 2.0   # P-gain for altitude velocity controller
ALT_VEL_MAX  = 0.5   # m/s — max vertical correction speed

# Stuck detection
STUCK_TIMEOUT_S = 10.0
STUCK_DIST_M = 0.4
STUCK_ESCAPE_M = 2.5

# Detection confirmation
DETECT_CONFIRM = 4

# Carrot-point lookahead — drone targets a point this far ahead along the path,
# not the raw WP. Shorter bursts = more responsive to local obstacles.
LOOKAHEAD_DIST = 2.5   # m

MERGE_DIST = 3.0

# Map update throttle: update every N explore ticks (depth_to_xy_map subsamples 4x so cost is low)
MAP_THROTTLE = 5

# Visited grid
CELL_SIZE     = 2.0          # m — grid cell resolution
GRID_N_ORIGIN = ZONE1_N[0]   # -4.0  m — south edge relative to spawn NED
GRID_E_ORIGIN = ZONE1_E[0]   # -12.0 m — west  edge relative to spawn NED
GRID_N_CELLS  = 18           # ceil((ZONE2_N[1] - ZONE1_N[0]) / CELL_SIZE) = 36/2
GRID_E_CELLS  = 20           # ceil((ZONE3_E[1] - ZONE1_E[0]) / CELL_SIZE) = 40/2

# Stabilize sensor gate
STABILIZE_TIMEOUT = 30.0  # s — abort if sensors not valid within this window

CAM_K = np.array([
    [433.0, 0.0, 320.0],
    [0.0, 433.0, 240.0],
    [0.0, 0.0, 1.0],
])


# ---------------------------------------------------------------------------
# Mission state machine
# ---------------------------------------------------------------------------
class MissionState(Enum):
    INIT = "INIT"
    CONNECT = "CONNECT"
    TAKEOFF = "TAKEOFF"
    STABILIZE = "STABILIZE"
    CENTERING = "CENTERING"
    STARTUP_SCAN = "STARTUP_SCAN"
    EXPLORE = "EXPLORE"
    SCAN = "SCAN"
    ESCAPE = "ESCAPE"
    DONE = "DONE"
    LAND = "LAND"


# ---------------------------------------------------------------------------
# Visited grid cell
# ---------------------------------------------------------------------------
class GridCell:
    __slots__ = ("visited_count", "last_visit_time", "scan_done", "blocked")

    def __init__(self):
        self.visited_count = 0
        self.last_visit_time = 0.0
        self.scan_done = False
        self.blocked = False


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
class QualifierMission:
    def __init__(self, model_path=""):
        # SharedState first — Drone reads live altitude/mode from it
        self.state = SharedState()
        self.stop_evt = asyncio.Event()
        self.drone = Drone(state=self.state)
        self.depth_rx = DepthReceiver(DEPTH_TOPIC)
        self.detector = BarrelDetector(model_path)
        self.tracker = DetectionTracker(MERGE_DIST)

        self.planner = AvoidancePlanner(
            K=CAM_K,
            width=640,
            height=480,
            safe_distance=SAFE_DIST,
            critical_distance=CRIT_DIST,
        )

        self.mapper = GlobalMapper(
            K=CAM_K,
            obs_h_min=0.1,
            obs_h_max=2.0,
            z_min=0.3,
            z_max=MAP_Z_MAX,
            yaw_in_degrees=True,
            yaw_clockwise=True,
            yaw_smoothing=0.3,
            subsample=4,
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

        # Detection deduplication: track which cells have confirmed barrels per class
        self._counted_cells = {"YELLOW": set(), "RED": set()}
        self._last_detection_frame_id = -1

        # Map update throttle counter
        self._map_tick = 0

        # Time of last escape (for post-escape look-ahead reduction)
        self._last_escape_time = 0.0

        # AVOID log throttle — print at most 1/s (every 20 ticks at 20 Hz)
        self._avoid_log_tick = 0

        # BOUNDARY log throttle — print at most once per 2 s
        self._boundary_log_time = 0.0


        # OFFBOARD gate warn throttle
        self._offboard_warn_time = 0.0
        # OFFBOARD persistence watchdog (Phase B)
        self._offboard_lost_since = 0.0     # monotonic timestamp when loss began (0 = healthy)
        self._offboard_recovery_count = 0   # auto re-entry attempts; aborts after 3

        # Tiered recovery (Phase C8) — escalate escape behaviour after consecutive stucks
        self._consecutive_stuck = 0         # reset on successful WP arrival

        # Detection-adjacent WP biasing (Phase C9) — track cells inserted as priority
        self._biased_cells = set()

        # Crash recovery (Phase 4)
        self._restart_count = 0
        self._was_in_offboard = False   # set True once we've entered offboard; gates crash detection
        self._was_flipped = False       # set True while flipping; triggers re-arm check when it clears
        self._flip_count = 0            # consecutive flips at current WP; resets on WP advance
        self._flip_start_time = 0.0     # monotonic when flip first detected; used for 5s timeout

        # RRT* path planner (Phase 2)
        self._rrt = RRTStarPlanner(
            safety_margin=RRT_SAFETY_M,
            step_size=RRT_STEP_M,
            max_iter=RRT_MAX_ITER,
        )
        self._rrt_task = None    # asyncio.Task for the background plan
        self._rrt_for_wp = -1   # wp index the pending plan targets

        # Wall segments in global NED (populated after spawn origin is known)
        self._wall_segs = []

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
        if self.state.latest_position is None or self.state.latest_yaw is None:
            return None

        return {
            "north":     float(self.state.latest_position.north_m),
            "east":      float(self.state.latest_position.east_m),
            "down":      float(self.state.latest_position.down_m),
            "yaw":       math.radians(float(self.state.latest_yaw)),
            "yaw_deg":   float(self.state.latest_yaw),
            "roll_deg":  float(self.state.latest_roll  or 0.0),
            "pitch_deg": float(self.state.latest_pitch or 0.0),
        }

    # ------------------------------------------------------------------
    # Arena boundary helpers
    # ------------------------------------------------------------------

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

    def _wp_crosses_forbidden(self, cur_n, cur_e, wp_n, wp_e, samples=10) -> bool:
        """Return True if WP destination is inside the forbidden zone (no margin).
        Does NOT check the path — avoidance + wall repulsion handle routing.
        Checking the path was too aggressive: Zone 1 north-edge WPs at N=13.5
        sit inside the WALL_MARGIN-expanded forbidden zone even though they are
        valid Zone 1 navigation targets."""
        return self._in_forbidden(wp_n, wp_e, margin=0.0)

    def _is_near_wall(self, n, e, extra=0.0):
        m = WALL_MARGIN + extra
        out_of_bounds = (n < self._n_min - WALL_MARGIN + m or
                         n > self._n_max + WALL_MARGIN - m or
                         e < self._e_min - WALL_MARGIN + m or
                         e > self._e_max + WALL_MARGIN - m)
        return out_of_bounds or self._in_forbidden(n, e, margin=m)

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

    def _cell_key(self, pose):
        return self._ned_to_cell(pose["north"], pose["east"])

    def _build_wall_segments(self):
        """Translate spawn-relative wall segments to global NED using spawn origin."""
        self._wall_segs = [
            ((self._origin_n + n1, self._origin_e + e1),
             (self._origin_n + n2, self._origin_e + e2))
            for (n1, e1), (n2, e2) in ARENA_WALL_SEGS_REL
        ]

    def _init_grid(self):
        self._build_wall_segments()
        self._grid = [[GridCell() for _ in range(GRID_E_CELLS)] for _ in range(GRID_N_CELLS)]
        blocked = 0
        for ci in range(GRID_N_CELLS):
            for cj in range(GRID_E_CELLS):
                cn, ce = self._cell_to_ned(ci, cj)
                if not self._in_arena(cn, ce):
                    self._grid[ci][cj].blocked = True
                    self._grid[ci][cj].scan_done = True
                    blocked += 1
                else:
                    # Phase 3: detection runs every tick during flight — no stop-and-scan needed
                    self._grid[ci][cj].scan_done = True
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

    # ------------------------------------------------------------------
    # Depth sanitization — replace NaN/Inf/non-positive with fill_value
    # ------------------------------------------------------------------
    def _sanitize_depth(self, depth, fill_value=MAP_Z_MAX):
        if depth is None:
            return None
        d = np.array(depth, dtype=float)
        mask = ~np.isfinite(d) | (d <= 0)
        d[mask] = fill_value
        return d

    # ------------------------------------------------------------------
    # Rotate waypoint list so first WP is nearest current pose
    # ------------------------------------------------------------------
    def _is_corner_wp(self, wp):
        """True when waypoint is near two walls simultaneously (corner trap)."""
        n, e = wp[0], wp[1]
        near_s = n < self._n_min + WALL_MARGIN
        near_n = n > self._n_max - WALL_MARGIN
        near_w = e < self._e_min + WALL_MARGIN
        near_e = e > self._e_max - WALL_MARGIN
        return (near_s or near_n) and (near_w or near_e)

    def _carrot_point(self, pose, wp):
        """Return a carrot point at most LOOKAHEAD_DIST ahead along pose→wp.
        Keeps the drone moving in short bursts rather than lunging at distant WPs,
        which improves local obstacle reactivity and gives smoother motion."""
        dn = wp[0] - pose["north"]
        de = wp[1] - pose["east"]
        dist = math.hypot(dn, de)
        if dist <= LOOKAHEAD_DIST or dist < 1e-3:
            return wp[0], wp[1], wp[2]
        t = LOOKAHEAD_DIST / dist
        return (pose["north"] + t * dn, pose["east"] + t * de, wp[2])

    def _rotate_waypoints_to_most_open(self, waypoints):
        """
        Pick starting WP whose approach path (from current position) has
        the fewest obstacle points within 1.5 m of the straight line.
        Prefers longer paths when obstacle counts are equal (open-space bias).
        Falls back to nearest non-wall WP if obstacle map is empty.
        """
        p = self._pose()
        if p is None or not waypoints:
            return waypoints

        cur_n, cur_e = p["north"], p["east"]
        obs_pts = self.mapper.get_global_points()

        order = sorted(
            range(len(waypoints)),
            key=lambda i: math.hypot(waypoints[i][0] - cur_n, waypoints[i][1] - cur_e)
        )

        best_idx = None
        best_score = -1.0

        # Two-pass: first try WPs within 12 m (prefer nearby clear paths); if
        # none qualify (e.g. spawned far from all WPs) fall back to all WPs.
        for dist_cap in (12.0, float("inf")):
            for idx in order:
                wp = waypoints[idx]
                if self._is_near_wall(wp[0], wp[1], extra=0.5):
                    continue

                path_n = wp[0] - cur_n
                path_e = wp[1] - cur_e
                path_len = math.hypot(path_n, path_e) + 1e-6

                if path_len > dist_cap:
                    continue

                if obs_pts.shape[0] > 0:
                    t = np.clip(
                        ((obs_pts[:, 0] - cur_n) * path_n + (obs_pts[:, 1] - cur_e) * path_e)
                        / (path_len ** 2),
                        0.0, 1.0,
                    )
                    closest_n = cur_n + t * path_n
                    closest_e = cur_e + t * path_e
                    dists_to_path = np.hypot(obs_pts[:, 0] - closest_n, obs_pts[:, 1] - closest_e)
                    obs_count = int(np.sum(dists_to_path < 1.5))
                else:
                    obs_count = 0

                # Higher score = more open + closer
                score = 1.0 / (1.0 + obs_count + 0.05 * path_len)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                break  # found a good nearby WP — skip the uncapped pass

        if best_idx is None:
            # All WPs near wall — fall back to nearest non-wall or first
            for idx in order:
                if not self._is_near_wall(waypoints[idx][0], waypoints[idx][1], extra=0.0):
                    best_idx = idx
                    break
            else:
                best_idx = order[0]

        wp = waypoints[best_idx]
        print(f"[NAV] Open-space start: WP {best_idx} N={wp[0]:.1f} E={wp[1]:.1f} score={best_score:.1f}")
        return waypoints[best_idx:] + waypoints[:best_idx]

    # ------------------------------------------------------------------
    # Startup 360° scan — rotate relative to current yaw and seed GlobalMapper
    # ------------------------------------------------------------------
    async def _startup_scan(self):
        p = self._pose()
        if p is None:
            return

        hold_n, hold_e, hold_d = p["north"], p["east"], p["down"]
        base_yaw = p["yaw_deg"]

        print("[SCAN] Starting 360° horizon scan — holding position")

        for delta_yaw in [0, 90, 180, 270]:
            target_yaw = base_yaw + float(delta_yaw)

            await self.drone.rotate_to_yaw(target_yaw)
            await asyncio.sleep(0.8)

            depth = self.depth_rx.get_frame()
            pose = self._pose()
            if pose is None:
                print("[SCAN] Pose lost during startup scan")
                break

            depth_clean = self._sanitize_depth(depth)
            if depth_clean is not None:
                self.mapper.update_frame(depth_clean, pose)

            await self.drone.send_position_setpoint(
                hold_n, hold_e, hold_d, target_yaw
            )

        await self.drone.rotate_to_yaw(base_yaw)
        await asyncio.sleep(0.5)

        pts = self.mapper.get_global_points()
        print(f"[SCAN] Done — {len(pts)} obstacle points in initial map")

    # ------------------------------------------------------------------
    # Detection — runs every tick, phase-gated, cell-deduped
    # ------------------------------------------------------------------
    def _run_detection(self):
        p = self._pose()
        if p is None:
            return

        result = self.detector.detect(phase=self._phase)

        frame_id = result.get("frame_id", -1)
        if frame_id == self._last_detection_frame_id:
            return
        self._last_detection_frame_id = frame_id

        # Phase-gate: yellow phase counts yellow only, red phase counts red only
        yellow_seen = bool(result.get("yellow", False)) if self._phase == "YELLOW" else False
        red_seen    = bool(result.get("red",    False)) if self._phase == "RED"    else False

        self._yellow_streak = (self._yellow_streak + 1) if yellow_seen else 0
        self._red_streak    = (self._red_streak    + 1) if red_seen    else 0

        # Use cell center as barrel location — not raw drone pose
        ci, cj = self._ned_to_cell(p["north"], p["east"])
        cell_n, cell_e = self._cell_to_ned(ci, cj)
        cell_key = (ci, cj)

        if self._yellow_streak >= DETECT_CONFIRM:
            if cell_key not in self._counted_cells["YELLOW"]:
                self._counted_cells["YELLOW"].add(cell_key)
                # Mark cell highly visited so navigator won't target it as a WP
                if self._grid is not None and self._valid_cell(ci, cj):
                    self._grid[ci][cj].visited_count = 999
                if self.tracker.try_add_yellow(cell_n, cell_e):
                    print(
                        f"[DETECT] YELLOW #{self.tracker.yellow_count}  "
                        f"cell=({ci},{cj}) N={cell_n:.1f} E={cell_e:.1f}  "
                        f"{self.tracker.summary()}"
                    )
                    # Phase C9: bias search toward neighbor cells (clusters)
                    self._insert_neighbor_wps(ci, cj, ALT_YELLOW)
            self._yellow_streak = 0

        if self._red_streak >= DETECT_CONFIRM:
            if cell_key not in self._counted_cells["RED"]:
                self._counted_cells["RED"].add(cell_key)
                # Mark cell highly visited so navigator won't target it as a WP
                if self._grid is not None and self._valid_cell(ci, cj):
                    self._grid[ci][cj].visited_count = 999
                if self.tracker.try_add_red(cell_n, cell_e):
                    print(
                        f"[DETECT] RED #{self.tracker.red_count}  "
                        f"cell=({ci},{cj}) N={cell_n:.1f} E={cell_e:.1f}  "
                        f"{self.tracker.summary()}"
                    )
                    # Phase C9: bias search toward neighbor cells (clusters)
                    self._insert_neighbor_wps(ci, cj, ALT_RED)
            self._red_streak = 0

    # ------------------------------------------------------------------
    # Sweep waypoint generation
    # ------------------------------------------------------------------
    def _build_zone_sweep(self, altitude, row_spacing):
        down = -altitude
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
    # Waypoint management
    # ------------------------------------------------------------------
    def _current_wp(self):
        return self._waypoints[self._wp_idx] if self._wp_idx < len(self._waypoints) else None

    def _arrived(self, wp):
        p = self._pose()
        if p is None:
            return False
        horiz = math.hypot(p["north"] - wp[0], p["east"] - wp[1])
        vert  = abs(p["down"] - wp[2])
        return horiz < ARRIVAL_RADIUS and vert < ARRIVAL_ALT

    def _advance_wp(self):
        self._wp_idx += 1
        self._reset_stuck()
        if self._consecutive_stuck > 0:
            self._consecutive_stuck = 0   # arriving at any WP = made progress
        self._flip_count = 0
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP {self._wp_idx}/{len(self._waypoints)}  "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m  "
                  f"T={self._time_left():.0f}s  {self.tracker.summary()}  "
                  f"{self._coverage_summary()}")
        # Background plan for the leg AFTER the one we just started
        self._start_rrt_plan(self._wp_idx, self._wp_idx + 1)

    def _start_phase(self, phase):
        self._phase = phase
        if phase == "YELLOW":
            wps = self._build_zone_sweep(ALT_YELLOW, ROW_SPACING_LOW)
            self._waypoints = self._rotate_waypoints_to_most_open(wps)
            print(f"\n[PHASE 1] Yellow sweep  alt={ALT_YELLOW}m  "
                  f"{len(self._waypoints)} waypoints")
        else:
            wps = self._build_zone_sweep(ALT_RED, ROW_SPACING_HIGH)
            self._waypoints = self._rotate_waypoints_to_most_open(wps)
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
        # Pre-plan first leg in background — will be ready by the time we arrive at WP 0
        self._start_rrt_plan(0, 1)

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    def _reset_stuck(self):
        p = self._pose()
        if p is None:
            return
        self._stuck_timer = time.monotonic()
        self._stuck_ref_n = p["north"]
        self._stuck_ref_e = p["east"]

    def _check_stuck(self):
        p = self._pose()
        if p is None:
            return False
        # Only fire stuck when drone is confirmed airborne in OFFBOARD.
        # A grounded drone is motionless by definition — not "stuck".
        if not self.state.is_in_offboard:
            return False
        if p["down"] > -0.3:  # NED: down < 0 when above ground; -0.3 = ~30 cm
            return False
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
        if p is None:
            return
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
                self._last_escape_time = time.monotonic()
                break

        self._reset_stuck()

    # ------------------------------------------------------------------
    # Wall repulsion — geometric, FOV-independent (Phase 2)
    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_on_seg(n, e, n1, e1, n2, e2):
        dn, de = n2 - n1, e2 - e1
        seg_sq = dn * dn + de * de
        if seg_sq < 1e-9:
            return n1, e1
        t = max(0.0, min(1.0, ((n - n1) * dn + (e - e1) * de) / seg_sq))
        return n1 + t * dn, e1 + t * de

    def _compute_wall_repulsion(self, n, e):
        """Unit repulsion vector from all wall segments within WALL_INF_M."""
        rep_n, rep_e = 0.0, 0.0
        for (n1, e1), (n2, e2) in self._wall_segs:
            wn, we = self._nearest_on_seg(n, e, n1, e1, n2, e2)
            dist = math.hypot(n - wn, e - we)
            if 1e-3 < dist < WALL_INF_M:
                t = 1.0 - dist / WALL_INF_M   # linear falloff to 0 at influence radius
                rep_n += t * (n - wn) / dist
                rep_e += t * (e - we) / dist
        mag = math.hypot(rep_n, rep_e)
        if mag < 1e-6:
            return 0.0, 0.0
        return rep_n / mag, rep_e / mag

    # ------------------------------------------------------------------
    # RRT* background path planning (Phase 2)
    # ------------------------------------------------------------------
    def _start_rrt_plan(self, from_idx, to_idx):
        """Launch background RRT* plan for the leg from_idx → to_idx."""
        if to_idx >= len(self._waypoints):
            return
        from_wp = self._waypoints[from_idx]
        to_wp   = self._waypoints[to_idx]
        obs_snap = self.mapper.get_global_points().copy()

        n_min, n_max = self._n_min, self._n_max
        e_min, e_max = self._e_min, self._e_max
        rrt = self._rrt

        def _plan_sync():
            # KDTree needs ≥1 point; use a dummy far away when map is empty
            if obs_snap.shape[0] == 0:
                obs_pts = np.array([[n_max + 100, e_max + 100]])
            else:
                obs_pts = obs_snap
            bounds = np.array([[n_min, n_max], [e_min, e_max]])
            return rrt.plan(
                start=[from_wp[0], from_wp[1]],
                goal=[to_wp[0], to_wp[1]],
                obstacle_points=obs_pts,
                bounds=bounds,
            )

        if self._rrt_task is not None and not self._rrt_task.done():
            self._rrt_task.cancel()

        self._rrt_for_wp = to_idx
        self._rrt_task = asyncio.create_task(asyncio.to_thread(_plan_sync))

    async def _replan_via_rrt(self, target_wp):
        """
        On-demand RRT* replan from current pose to target_wp. Inserts the
        intermediate path before the current wp_idx so navigation continues
        through the replanned route. Returns True if path was inserted.
        Used by Tier 2 escape escalation (Phase C8).
        """
        p = self._pose()
        if p is None:
            return False

        obs_pts = self.mapper.get_global_points()
        if obs_pts.shape[0] == 0:
            obs_pts = np.array([[self._n_max + 100, self._e_max + 100]])

        bounds = np.array([[self._n_min, self._n_max], [self._e_min, self._e_max]])

        try:
            path = await asyncio.to_thread(
                self._rrt.plan,
                [p["north"], p["east"]],
                [target_wp[0], target_wp[1]],
                obs_pts,
                bounds,
            )
        except Exception as e:
            print(f"[REPLAN] RRT* threw: {e}")
            return False

        if path is None or len(path) <= 2:
            return False

        alt_d = target_wp[2]
        intermediate = [(float(pt[0]), float(pt[1]), alt_d) for pt in path[1:-1]]
        if not intermediate:
            return False
        n_ins = len(intermediate)
        self._waypoints = (
            self._waypoints[:self._wp_idx]
            + intermediate
            + self._waypoints[self._wp_idx:]
        )
        # Shift pending RRT* target index so the background plan still maps to the same WP
        if self._rrt_for_wp >= self._wp_idx:
            self._rrt_for_wp += n_ins
        print(f"[REPLAN] Inserted {n_ins} sub-WPs before WP {self._wp_idx}")
        return True

    def _insert_neighbor_wps(self, ci, cj, alt):
        """
        After detection at cell (ci, cj), insert the 4 cardinal neighbor cells
        as priority WPs. Skips cells that are blocked, already visited, near a
        wall, or already biased. Phase C9 — biases search around clusters.
        """
        if self._grid is None:
            return
        down = -alt
        inserted = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = ci + di, cj + dj
            cell_key = (ni, nj)
            if cell_key in self._biased_cells:
                continue
            if not self._valid_cell(ni, nj):
                continue
            cell = self._grid[ni][nj]
            if cell.blocked or cell.visited_count > 0:
                continue
            cn, ce = self._cell_to_ned(ni, nj)
            if self._is_near_wall(cn, ce, extra=1.0):
                continue
            self._biased_cells.add(cell_key)
            inserted.append((cn, ce, down))

        if inserted:
            insert_at = self._wp_idx + 1
            n_ins = len(inserted)
            self._waypoints = (
                self._waypoints[:insert_at]
                + inserted
                + self._waypoints[insert_at:]
            )
            # Shift pending RRT* target index to track the same target WP
            if self._rrt_for_wp >= insert_at:
                self._rrt_for_wp += n_ins
            print(
                f"[BIAS] Detection cluster at ({ci},{cj}) — "
                f"inserted {n_ins} neighbor cells after WP {self._wp_idx}"
            )

    def _try_apply_rrt_plan(self):
        """If a background plan is ready for wp_idx+1, insert its intermediate points."""
        if (self._rrt_task is None
                or not self._rrt_task.done()
                or self._rrt_for_wp != self._wp_idx + 1):
            return
        try:
            path = self._rrt_task.result()
        except Exception:
            self._rrt_task = None
            return
        self._rrt_task = None

        if path is None or len(path) <= 2:
            return  # RRT* failed or path is trivial — use direct leg

        # Defensive bounds check — _rrt_for_wp shifts are tracked but be safe
        if self._rrt_for_wp >= len(self._waypoints) or self._rrt_for_wp < 0:
            return
        target_wp = self._waypoints[self._rrt_for_wp]
        alt_d = target_wp[2]
        # path[0] = start (current WP, already visited), path[-1] = goal (already in list)
        intermediate = [(float(pt[0]), float(pt[1]), alt_d) for pt in path[1:-1]]
        if intermediate:
            self._waypoints = (
                self._waypoints[:self._rrt_for_wp]
                + intermediate
                + self._waypoints[self._rrt_for_wp:]
            )
            print(f"[RRT*] Inserted {len(intermediate)} sub-WPs before WP {self._rrt_for_wp}")

    # ------------------------------------------------------------------
    # Velocity setpoint — goal + avoidance + memory + wall blend → (vn, ve, vd, yaw)
    # ------------------------------------------------------------------
    def _compute_velocity_setpoint(self, pose, target_n, target_e, target_d, depth):
        cur_n, cur_e = pose["north"], pose["east"]
        cur_yaw = pose["yaw_deg"]

        # Emergency: near wall/forbidden — push toward zone 1 centre at VEL_MIN
        if self._is_near_wall(cur_n, cur_e, extra=0.0):
            safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
            safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0
            dn = safe_n - cur_n
            de = safe_e - cur_e
            mag = math.hypot(dn, de)
            if mag > 1e-3:
                dn /= mag
                de /= mag
            yaw = math.degrees(math.atan2(de, dn))
            vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))
            _now = time.monotonic()
            if _now - self._boundary_log_time >= 2.0:
                print(f"[BOUNDARY] Near wall/forbidden at N={cur_n:.1f} E={cur_e:.1f} — recovering")
                self._boundary_log_time = _now
            return VEL_MIN * dn, VEL_MIN * de, vd, yaw

        # Goal vector toward current waypoint
        dn   = target_n - cur_n
        de   = target_e - cur_e
        dist = math.hypot(dn, de)
        goal_n = (dn / dist) if dist > 1e-3 else 1.0
        goal_e = (de / dist) if dist > 1e-3 else 0.0

        # Avoidance vector from depth camera (depth already sanitized by caller)
        avoid_n, avoid_e, blocked = 0.0, 0.0, False
        center_clearance = SAFE_DIST
        left_clearance   = SAFE_DIST
        right_clearance  = SAFE_DIST
        if depth is not None:
            av_n, av_e, _, info = self.planner.compute_position_ned(
                depth, pose, step_size=1.0
            )
            blocked = info["blocked"]
            av_dist = math.hypot(av_n - cur_n, av_e - cur_e)
            if av_dist > 1e-3:
                avoid_n = (av_n - cur_n) / av_dist
                avoid_e = (av_e - cur_e) / av_dist
            center_clearance = info["clearance"]["center"]
            left_clearance   = info["clearance"]["left"]
            right_clearance  = info["clearance"]["right"]
            if blocked:
                self._avoid_log_tick += 1
                if self._avoid_log_tick % 20 == 1:  # ~1 Hz at 20 Hz control loop
                    cl = info["clearance"]
                    print(f"[AVOID] L={cl['left']:.1f} C={cl['center']:.1f} "
                          f"R={cl['right']:.1f}")

        # Emergency stop: obstacle < 0.6 m ahead OR < 0.5 m to either side.
        # Side threshold catches corner clips where center clearance looks fine.
        if center_clearance < 0.6 or min(left_clearance, right_clearance) < 0.5:
            vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))
            return 0.0, 0.0, vd, cur_yaw

        # Memory-map repulsion from GlobalMapper
        mem_n, mem_e = self.mapper.get_repulsion_vector(cur_n, cur_e, MAP_INFLUENCE_M)
        if not (math.isfinite(mem_n) and math.isfinite(mem_e)):
            mem_n, mem_e = 0.0, 0.0

        # Geometric wall repulsion — always active, no camera FOV dependency
        wall_n, wall_e = self._compute_wall_repulsion(cur_n, cur_e)

        # All three sectors critical — use memory repulsion to escape obstacle cluster
        all_critical = (left_clearance < CRIT_DIST and center_clearance < CRIT_DIST
                        and right_clearance < CRIT_DIST)

        # Use minimum clearance across all sectors for speed and avoidance weight.
        # Corner clips happen because center is clear but a side is close — using
        # center-only let the drone fly full speed into a wall corner.
        min_clearance = min(center_clearance, left_clearance, right_clearance)

        # In open space, reduce avoidance weight so goal vector dominates
        w_avoid = 0.1 if min_clearance >= SAFE_DIST else W_AVOID

        # Blend: goal + avoidance + memory + wall
        if all_critical:
            if abs(mem_n) > 1e-3 or abs(mem_e) > 1e-3:
                blend_n = mem_n + W_WALL * wall_n
                blend_e = mem_e + W_WALL * wall_e
            else:
                blend_n = avoid_n + W_WALL * wall_n
                blend_e = avoid_e + W_WALL * wall_e
        elif blocked:
            blend_n = avoid_n + W_MEM_AVOID * mem_n + W_WALL * wall_n
            blend_e = avoid_e + W_MEM_AVOID * mem_e + W_WALL * wall_e
        else:
            blend_n = goal_n + w_avoid * avoid_n + W_MEM_AVOID * mem_n + W_WALL * wall_n
            blend_e = goal_e + w_avoid * avoid_e + W_MEM_AVOID * mem_e + W_WALL * wall_e

        mag = math.hypot(blend_n, blend_e)
        if mag > 1e-3:
            blend_n /= mag
            blend_e /= mag
        else:
            blend_n, blend_e = goal_n, goal_e

        # Guard: abort to hover if blend is non-finite
        if not (math.isfinite(blend_n) and math.isfinite(blend_e)):
            return 0.0, 0.0, 0.0, cur_yaw

        # --- Speed scaling based on minimum clearance across all sectors ---
        if min_clearance >= SAFE_DIST:
            speed = VEL_MAX
        elif min_clearance > CRIT_DIST:
            t = (min_clearance - CRIT_DIST) / (SAFE_DIST - CRIT_DIST)
            speed = VEL_MIN + t * (VEL_MAX - VEL_MIN)
        else:
            speed = VEL_MIN

        # Slow down as we approach the waypoint (last 1.5m)
        if dist < 1.5:
            speed = min(speed, dist * (VEL_MAX / 1.5))
        speed = max(speed, VEL_MIN)

        # Slow down briefly after an escape maneuver
        if time.monotonic() - self._last_escape_time < 4.0:
            speed = min(speed, VEL_MIN)

        vn = speed * blend_n
        ve = speed * blend_e

        # --- Altitude P-controller ---
        # NED: down < 0 when above ground; positive vd moves toward ground
        vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))

        # --- Yaw rate limit — prevents snapping to face a wall in one tick ---
        desired_yaw = math.degrees(math.atan2(blend_e, blend_n))
        if not math.isfinite(desired_yaw):
            desired_yaw = cur_yaw
        yaw_error = ((desired_yaw - cur_yaw + 180.0) % 360.0) - 180.0
        yaw_step  = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, yaw_error))
        commanded_yaw = cur_yaw + yaw_step

        return vn, ve, vd, commanded_yaw

    # ------------------------------------------------------------------
    # STABILIZE: block until pose + yaw + depth are all valid
    # ------------------------------------------------------------------
    async def _wait_stabilize(self):
        """
        Block until pose, yaw, depth are all valid AND pass sanity checks:
        - Position: drone is airborne (down < -0.3 m, NED)
        - Depth: at least 10% of pixels are finite-positive (not all NaN/Inf/0)
        Stale (0,0,0) telemetry or all-NaN depth would otherwise sneak through.
        """
        print(
            f"[STABILIZE] Waiting for pose+yaw+depth + sanity checks "
            f"(timeout={STABILIZE_TIMEOUT:.0f}s)..."
        )
        t0 = time.monotonic()
        while True:
            pose_ok  = self.state.latest_position is not None
            yaw_ok   = self.state.latest_yaw is not None
            depth = self.depth_rx.get_frame()

            # Pose sanity: drone must be airborne (NED: down < -0.3 m means above ground)
            pose_airborne = pose_ok and self.state.latest_position.down_m < -0.3

            # Depth sanity: enough finite positive pixels to be useful
            depth_finite_frac = 0.0
            depth_valid = False
            if depth is not None:
                d = np.asarray(depth, dtype=float)
                depth_finite_frac = float(np.mean(np.isfinite(d) & (d > 0)))
                depth_valid = depth_finite_frac > 0.1

            if pose_airborne and yaw_ok and depth_valid:
                p = self.state.latest_position
                print(
                    f"[STABILIZE] OK — pos N={p.north_m:.2f} E={p.east_m:.2f} D={p.down_m:.2f}  "
                    f"yaw={self.state.latest_yaw:.1f}°  "
                    f"depth_finite={depth_finite_frac*100:.0f}%"
                )
                return True

            elapsed = time.monotonic() - t0
            if elapsed > STABILIZE_TIMEOUT:
                print(
                    f"[STABILIZE] Timeout after {STABILIZE_TIMEOUT:.0f}s — "
                    f"pose={'ok' if pose_ok else 'MISSING'}({'airborne' if pose_airborne else 'GROUNDED'})  "
                    f"yaw={'ok' if yaw_ok else 'MISSING'}  "
                    f"depth={'ok' if depth_valid else f'BAD({depth_finite_frac*100:.0f}% finite)'}"
                )
                return False

            await asyncio.sleep(0.2)

    # ------------------------------------------------------------------
    # SAFE CENTERING: move to Zone 1 geometric center before sweep
    # Avoids spawn-corner trap when spawn is near a wall.
    # ------------------------------------------------------------------
    async def _safe_centering(self, timeout=15.0):
        safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
        safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0
        safe_d = -ALT_YELLOW
        print(f"[CENTER] Moving to Zone 1 center N={safe_n:.1f} E={safe_e:.1f} alt={-safe_d:.1f}m")

        p = self._pose()
        if p is None:
            print("[CENTER] No pose — skipping")
            return
        target_yaw = p["yaw_deg"]

        deadline = time.monotonic() + timeout
        arrived = False
        while time.monotonic() < deadline:
            p = self._pose()
            if p is None:
                await asyncio.sleep(0.1)
                continue
            horiz = math.hypot(p["north"] - safe_n, p["east"] - safe_e)
            vert  = abs(p["down"] - safe_d)
            if horiz < 1.0 and vert < 0.6:
                arrived = True
                print(f"[CENTER] Arrived (horiz={horiz:.2f}m, vert={vert:.2f}m)")
                break
            await self.drone.send_position_setpoint(safe_n, safe_e, safe_d, target_yaw)
            await asyncio.sleep(0.1)

        if not arrived:
            p = self._pose()
            if p is not None:
                horiz = math.hypot(p["north"] - safe_n, p["east"] - safe_e)
                print(f"[CENTER] Timeout (horiz={horiz:.2f}m) — proceeding anyway")

        # Hold position briefly for stability before scan
        for _ in range(10):
            await self.drone.send_position_setpoint(safe_n, safe_e, safe_d, target_yaw)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # State tick: EXPLORE
    # ------------------------------------------------------------------
    async def _tick_explore(self):
        if self._time_left() < 10.0:
            print("[MISSION] Time limit reached.")
            self._state = MissionState.DONE
            return

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

        pose = self._pose()
        if pose is None:
            return
        cur_n = pose["north"]
        cur_e = pose["east"]

        self._update_grid(pose)

        if self._arrived(wp):
            self._try_apply_rrt_plan()
            self._advance_wp()
            return

        if self._check_stuck():
            print("[FSM] Stuck detected → EXPLORE → ESCAPE")
            self._state = MissionState.ESCAPE
            return

        target_n, target_e, target_d = wp   # real WP (used for arrival + forbidden check)

        # Skip WP whose destination is inside the forbidden zone.
        if self._wp_crosses_forbidden(cur_n, cur_e, target_n, target_e):
            _now = time.monotonic()
            if _now - self._boundary_log_time >= 5.0:
                print(f"[BOUNDARY] WP {self._wp_idx} destination in forbidden zone — skipping")
                self._boundary_log_time = _now
            self._advance_wp()   # natural phase-done when idx >= len
            return

        # Sanitize depth before any planner/map use
        depth = self._sanitize_depth(self.depth_rx.get_frame())

        # Throttle map updates to prevent GlobalMapper overload
        self._map_tick += 1
        if depth is not None and self._map_tick % MAP_THROTTLE == 0:
            self.mapper.update_frame(depth, pose)
            self.mapper.prune(cur_n, cur_e, MAP_RETENTION_M)

        # Detection every tick — no stop-and-scan required (Phase 3)
        self._run_detection()

        # Carrot point: target a point LOOKAHEAD_DIST ahead along path to WP.
        # Prevents lunging at distant WPs; makes movement smoother and more
        # reactive to local obstacles.
        carrot_n, carrot_e, carrot_d = self._carrot_point(pose, wp)

        vn, ve, vd, yaw_deg = self._compute_velocity_setpoint(
            pose, carrot_n, carrot_e, carrot_d, depth
        )

        await self.drone.send_velocity(vn, ve, vd, yaw_deg)

    # ------------------------------------------------------------------
    # State tick: SCAN
    # 360° scan with detection at each heading; marks cell scan_done on exit.
    # ------------------------------------------------------------------
    async def _tick_scan(self):
        print("[FSM] SCAN — running 360° scan with detection")

        p = self._pose()
        if p is None:
            self._state = MissionState.EXPLORE
            return

        hold_n, hold_e, hold_d = p["north"], p["east"], p["down"]
        base_yaw = p["yaw_deg"]

        for delta_yaw in [0, 90, 180, 270]:
            target_yaw = base_yaw + float(delta_yaw)

            await self.drone.rotate_to_yaw(target_yaw)
            await asyncio.sleep(0.8)

            pose = self._pose()
            if pose is None:
                break

            depth_clean = self._sanitize_depth(self.depth_rx.get_frame())
            if depth_clean is not None:
                self.mapper.update_frame(depth_clean, pose)

            self._run_detection()

            await self.drone.send_position_setpoint(hold_n, hold_e, hold_d, target_yaw)

        await self.drone.rotate_to_yaw(base_yaw)
        await asyncio.sleep(0.5)

        # Mark current cell as scanned (once, prevents re-scan loops)
        if self._grid is not None:
            pose = self._pose()
            if pose is not None:
                ci, cj = self._ned_to_cell(pose["north"], pose["east"])
                if self._valid_cell(ci, cj):
                    self._grid[ci][cj].scan_done = True
                    print(f"[GRID] Cell ({ci},{cj}) marked scan_done")

        pts = self.mapper.get_global_points()
        print(f"[SCAN] Done — {len(pts)} obstacle points in map")
        self._state = MissionState.EXPLORE

    # ------------------------------------------------------------------
    # State tick: ESCAPE
    # ------------------------------------------------------------------
    async def _tick_escape(self):
        """
        Tiered recovery (Phase C8). Counter resets on WP arrival via _advance_wp.
          Tier 1: reactive escape direction + skip 1 WP (original behaviour)
          Tier 2: RRT* on-demand replan from current pose to current WP
          Tier 3+: skip 3 WPs to bypass current sweep row
        """
        self._consecutive_stuck += 1
        tier = self._consecutive_stuck
        wp = self._current_wp()
        print(f"[ESCAPE] Tier {tier} (consecutive stucks at this leg)")

        if tier == 1:
            # Reactive escape — random non-wall direction
            await self._escape_stuck()
            p = self._pose()
            if p is not None:
                await self.drone.send_position_setpoint(
                    p["north"], p["east"], p["down"], p["yaw_deg"]
                )
            await asyncio.sleep(2.0)
            if self._wp_idx < len(self._waypoints) - 1:
                self._wp_idx += 1
                nwp = self._current_wp()
                if nwp:
                    print(f"[ESCAPE] T1 — skipping stuck WP → WP {self._wp_idx}/"
                          f"{len(self._waypoints)} N={nwp[0]:.1f} E={nwp[1]:.1f}")

        elif tier == 2:
            # On-demand RRT* replan to current target
            print("[ESCAPE] T2 — invoking on-demand RRT* replan")
            ok = False
            if wp is not None:
                ok = await self._replan_via_rrt(wp)
            if not ok:
                print("[ESCAPE] T2 — RRT* failed, fallback to skip + escape")
                await self._escape_stuck()
                if self._wp_idx < len(self._waypoints) - 1:
                    self._wp_idx += 1
            p = self._pose()
            if p is not None:
                await self.drone.send_position_setpoint(
                    p["north"], p["east"], p["down"], p["yaw_deg"]
                )
            await asyncio.sleep(1.5)

        else:
            # Tier 3+: skip multiple WPs to bypass row entirely
            old = self._wp_idx
            self._wp_idx = min(self._wp_idx + 3, len(self._waypoints) - 1)
            nwp = self._current_wp()
            jumped = self._wp_idx - old
            print(f"[ESCAPE] T3+ — skipping {jumped} WPs to bypass row")
            if nwp:
                print(f"  New target: WP {self._wp_idx}/{len(self._waypoints)} "
                      f"N={nwp[0]:.1f} E={nwp[1]:.1f}")
            self._reset_stuck()
            self._consecutive_stuck = 0   # bypassed row — clear escalation
            p = self._pose()
            if p is not None:
                await self.drone.send_position_setpoint(
                    p["north"], p["east"], p["down"], p["yaw_deg"]
                )
            await asyncio.sleep(1.0)

        print("[FSM] ESCAPE → EXPLORE")
        self._state = MissionState.EXPLORE

    # ------------------------------------------------------------------
    # Crash recovery — re-arm, re-takeoff, resume mission (Phase 4)
    # ------------------------------------------------------------------
    async def _attempt_restart(self):
        self._restart_count += 1
        print(f"[RECOVERY] Restart attempt {self._restart_count}/{MAX_RESTARTS}")

        if self._restart_count > MAX_RESTARTS:
            print("[RECOVERY] Max restarts reached — ending mission")
            self._state = MissionState.DONE
            return

        # ── Stage 1: stop OFFBOARD so PX4 switches to HOLD/LAND ─────────
        print("[RECOVERY] Stage 1 — stopping OFFBOARD")
        try:
            await self.drone.drone.offboard.stop()
        except Exception as e:
            print(f"[RECOVERY]   offboard.stop() failed (ignored): {e}")
        await asyncio.sleep(1.0)

        # ── Stage 2: land if airborne ────────────────────────────────────
        p = self._pose()
        alt_m = (-p["down"]) if (p is not None) else 0.0
        if alt_m > 0.3:
            print(f"[RECOVERY] Stage 2 — airborne ({alt_m:.1f}m), commanding land")
            try:
                await self.drone.drone.action.land()
            except Exception as e:
                print(f"[RECOVERY]   land() failed (ignored): {e}")
            # Wait up to 20 s for altitude to drop below 0.2 m
            for _ in range(40):
                await asyncio.sleep(0.5)
                p = self._pose()
                alt_m = (-p["down"]) if (p is not None) else 0.0
                if alt_m < 0.2:
                    break
            print(f"[RECOVERY]   landed (alt={alt_m:.2f}m)")
        else:
            print("[RECOVERY] Stage 2 — already on ground, skipping land")

        # ── Stage 3: wait for disarm / PX4 settle ────────────────────────
        print("[RECOVERY] Stage 3 — waiting for disarm / settle (3 s)")
        await asyncio.sleep(3.0)

        # ── Stage 4: re-arm and take off ─────────────────────────────────
        print("[RECOVERY] Stage 4 — re-arm and takeoff")
        try:
            await self.drone.rearm_and_takeoff()
        except Exception as e:
            print(f"[RECOVERY] Re-takeoff failed: {e} — ending mission")
            self._state = MissionState.DONE
            return

        # Wait for telemetry to re-populate after re-takeoff
        for _ in range(50):
            if self.state.latest_position is not None:
                break
            await asyncio.sleep(0.1)

        # Reset stuck detection from new position; stay in current phase/waypoint
        self._reset_stuck()
        self._flip_count = 0
        self._was_in_offboard = False   # will flip back True once telemetry confirms OFFBOARD
        print(f"[RECOVERY] Airborne again — resuming phase={self._phase} WP={self._wp_idx}")

    # ------------------------------------------------------------------
    # Dispatch-table control loop
    # ------------------------------------------------------------------
    async def _control_loop(self):
        dt = 1.0 / CONTROL_HZ

        _dispatch = {
            MissionState.EXPLORE: self._tick_explore,
            MissionState.ESCAPE:  self._tick_escape,
        }

        print(f"[FSM] Control loop started — state: {self._state.value}")

        while self._state not in (MissionState.DONE, MissionState.LAND):
            t0 = time.monotonic()

            # Track when we first enter OFFBOARD — gates crash detection
            if self.state.is_in_offboard:
                self._was_in_offboard = True

            # Crash detection: was in OFFBOARD, now out + disarmed = crashed and grounded.
            # Only fires during active mission states (not during escape/recovery spin-up).
            if (self._was_in_offboard
                    and not self.state.is_in_offboard
                    and not self.state.is_armed
                    and self._state in (MissionState.EXPLORE,
                                        MissionState.SCAN,
                                        MissionState.ESCAPE)):
                await self._attempt_restart()
                continue

            # Flip detection: wait for attitude to settle; on clearance check if drone crashed
            if self.state.is_flipped:
                roll  = self.state.latest_roll  or 0.0
                pitch = self.state.latest_pitch or 0.0
                if not self._was_flipped:
                    self._flip_start_time = time.monotonic()
                    print(f"[RECOVERY] Flip detected (roll={roll:.1f}° pitch={pitch:.1f}°) — waiting for settle")
                self._was_flipped = True
                # Hard recovery if still flipped after 5 s — drone cannot right itself
                if time.monotonic() - self._flip_start_time > 5.0:
                    print("[RECOVERY] Flip timeout (>5s upside-down) — forcing hard recovery")
                    self._was_flipped = False
                    self._flip_count += 1
                    if self._restart_count < MAX_RESTARTS:
                        await self._attempt_restart()
                    else:
                        print("[RECOVERY] Max restarts reached — ending mission")
                        self._state = MissionState.DONE
                    continue
                # Don't send position setpoints to a potentially crashed drone — just wait
                await asyncio.sleep(0.5)
                continue

            # Flip just cleared — two-tier recovery
            if self._was_flipped:
                self._was_flipped = False
                self._flip_count += 1
                p = self._pose()
                alt_m = (-p["down"]) if (p is not None) else 0.0
                armed = self.state.is_armed
                offboard = self.state.is_in_offboard

                # Hard recovery: disarmed / OFFBOARD lost / crashed to ground /
                # or repeated flips at same WP (stuck in tight space even if airborne)
                hard = (
                    not armed
                    or not offboard
                    or alt_m < 0.5
                    or self._flip_count >= 2
                )

                if hard:
                    print(
                        f"[RECOVERY] Hard recovery "
                        f"(alt={alt_m:.1f}m armed={armed} offboard={offboard} "
                        f"flips@wp={self._flip_count}) — land + re-arm"
                    )
                    self._flip_count = 0
                    if self._restart_count < MAX_RESTARTS:
                        await self._attempt_restart()
                    else:
                        print("[RECOVERY] Max restarts reached — ending mission")
                        self._state = MissionState.DONE
                    continue
                else:
                    # Soft recovery: hold position 3 s to let drone stabilise.
                    # Use WP target altitude — p["down"] may be wrong after
                    # a flip-induced EKF home reference jump (seen as -459m / 15m boomerang).
                    wp_now = self._current_wp()
                    hold_d = wp_now[2] if wp_now else -ALT_YELLOW
                    print(
                        f"[RECOVERY] Soft recovery flip #{self._flip_count} "
                        f"(alt={alt_m:.1f}m hold_d={hold_d:.2f}) — holding 3s then resuming"
                    )
                    await self.drone.send_position_setpoint(
                        p["north"], p["east"], hold_d, p["yaw_deg"]
                    )
                    await asyncio.sleep(3.0)

            # OFFBOARD persistence watchdog (Phase B):
            # If PX4 drops OFFBOARD mid-mission while armed and airborne, attempt
            # automatic re-entry (no full re-arm). Up to 3 attempts.
            # If grounded/disarmed, fall through to crash detection above.
            if (not self.state.is_in_offboard
                    and self._state in (MissionState.EXPLORE, MissionState.ESCAPE)):
                _now = time.monotonic()
                if self._offboard_lost_since == 0.0:
                    self._offboard_lost_since = _now
                lost_dur = _now - self._offboard_lost_since

                p = self._pose()
                airborne = p is not None and p["down"] < -0.3

                # Sustained loss + still airborne + armed → try re-entry
                if (lost_dur > 2.0
                        and airborne
                        and self.state.is_armed
                        and self._offboard_recovery_count < 3):
                    self._offboard_recovery_count += 1
                    print(
                        f"[OFFBOARD-WD] Lost {lost_dur:.1f}s while airborne "
                        f"(alt={-p['down']:.2f}m) — re-entry attempt "
                        f"{self._offboard_recovery_count}/3"
                    )
                    try:
                        await self.drone._start_offboard_with_confirm(confirm_timeout=3.0)
                        print("[OFFBOARD-WD] Re-entered ✓")
                        self._offboard_lost_since = 0.0
                    except Exception as e:
                        print(f"[OFFBOARD-WD] Re-entry failed: {e}")
                    continue

                if _now - self._offboard_warn_time > 5.0:
                    alt = -p["down"] if p else 0.0
                    print(
                        f"[WARN] Not in OFFBOARD (armed={self.state.is_armed} "
                        f"alt={alt:.2f}m lost={lost_dur:.1f}s) — pausing navigation"
                    )
                    self._offboard_warn_time = _now
                await asyncio.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            else:
                # Healthy OFFBOARD — reset watchdog state
                if self._offboard_lost_since != 0.0:
                    self._offboard_lost_since = 0.0
                    self._offboard_recovery_count = 0

            handler = _dispatch.get(self._state)
            if handler is not None:
                await handler()
            else:
                print(f"[FSM] No handler for state {self._state.value} — idling one tick")

            # Always yield to event loop so gz-transport callbacks drain (fixes 6.3)
            await asyncio.sleep(max(0.0, dt - (time.monotonic() - t0)))

        print(f"[FSM] Control loop exited — final state: {self._state.value}")

    # ------------------------------------------------------------------
    # Helpers — monitor task teardown
    # ------------------------------------------------------------------
    async def _teardown_monitor(self, monitor):
        if monitor is None:
            return
        self.stop_evt.set()
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

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
        await asyncio.sleep(2)

        # NEW: Start position monitor BEFORE arm/takeoff so Drone has live
        # altitude/mode access during ascent. Avoids blind sleep(20) trap.
        monitor = asyncio.create_task(
            position_monitor_task(self.drone, self.state, self.stop_evt)
        )

        print("[INIT] Waiting for initial telemetry...")
        for _ in range(50):  # 5 s timeout
            if self.state.latest_position is not None and self.state.latest_yaw is not None:
                break
            await asyncio.sleep(0.1)

        if self.state.latest_position is None or self.state.latest_yaw is None:
            print("[FSM] Initial telemetry did not populate — aborting")
            self._state = MissionState.LAND
            await self._teardown_monitor(monitor)
            return
        print(
            f"[INIT] Initial telemetry OK — pos N={self.state.latest_position.north_m:.2f} "
            f"E={self.state.latest_position.east_m:.2f} D={self.state.latest_position.down_m:.2f}"
        )

        self._state = MissionState.TAKEOFF
        print(f"[FSM] {self._state.value}")
        print("[INIT] Arming and taking off (pure OFFBOARD)...")
        try:
            await self.drone.arm_and_takeoff(target_alt=ALT_YELLOW)
            self.state.is_armed = True
        except Exception as e:
            print(f"[FSM] Takeoff failed: {e}")
            print("[FSM] TAKEOFF → LAND (aborting safely)")
            self._state = MissionState.LAND
            await self._teardown_monitor(monitor)
            try:
                await self.drone.land()
            except Exception:
                pass
            return

        # Telemetry is already populated from monitor — no second wait needed.
        if self.state.latest_position is None or self.state.latest_yaw is None:
            print("[FSM] Telemetry lost after takeoff")
            self._state = MissionState.LAND
            await self._teardown_monitor(monitor)
            if self.state.is_armed:
                try:
                    await self.drone.land()
                finally:
                    self.state.is_armed = False
            return

        p = self._pose()
        if p is None:
            print("[FSM] Pose unavailable after takeoff")
            self._state = MissionState.LAND
            await self._teardown_monitor(monitor)
            if self.state.is_armed:
                try:
                    await self.drone.land()
                finally:
                    self.state.is_armed = False
            return

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
            await self._teardown_monitor(monitor)
            await self.drone.land()
            return

        self._state = MissionState.CENTERING
        print(f"[FSM] {self._state.value}")
        await self._safe_centering()

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
            await self._teardown_monitor(monitor)

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
        print("[INFO]   Tip: python3 qualifier_main.py barrels.pt  to enable YOLO\n")

    mission = QualifierMission(model_path)
    try:
        await mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] Keyboard interrupt")
        if mission.state.is_armed:
            try:
                await mission.drone.land()
            finally:
                mission.state.is_armed = False
    finally:
        try:
            mission.detector.close()
        except Exception:
            pass
        try:
            mission.depth_rx.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
