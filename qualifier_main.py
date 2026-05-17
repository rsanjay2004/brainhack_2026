# qualifier_main.py
#
# ─────────────────────────────────────────────────────────────────────────────
#  DRONE FUNCTIONALITY CHECKLIST  (DO NOT REMOVE ANY OF THESE BEHAVIOURS)
# ─────────────────────────────────────────────────────────────────────────────
#  Perception:
#    • Depth-camera obstacle detection — AvoidancePlanner reports L/C/R clearance
#    • Live 2D obstacle map (GlobalMapper) — accumulates obstacles over flight
#    • YOLO/HSV barrel detector — every tick during EXPLORE (Phase 3)
#    • 360° startup scan — seeds initial obstacle map
#    • Phase-transition scan — re-scan environment at new altitude
#
#  Navigation:
#    • Velocity-controlled flight (not position setpoints) with hard cap VEL_MAX
#    • Carrot lookahead (LOOKAHEAD_DIST) — short bursts instead of distant pull
#    • Geometric wall repulsion (continuous, FOV-independent)
#    • Memory-map repulsion (GlobalMapper.get_repulsion_vector)
#    • Path-history repulsion — drone avoids returning to recently visited cells
#    • Origin repulsion — heavily penalizes returning to spawn
#    • Yaw biases toward closest side obstacle (early lateral awareness)
#    • Boustrophedon (lawnmower) sweep WPs per zone
#    • RRT* path planning — startup validation + background per leg
#    • Nearest-unvisited WP skip — never backtrack across arena on a skip
#
#  Safety / collision avoidance:
#    • Three-tier clearance bands: SAFE_DIST → CRIT_DIST → emergency (<0.6m)
#    • Emergency LATERAL SLIDE toward most-open sector (no retreating)
#    • LOOK_AROUND state — 360° depth scan when AVOID persists (>2s) so drone
#      sees beyond its 60° FOV before committing to a direction
#    • Altitude bump — climbs 0.5m when stuck in AVOID-CRIT >10s
#    • EKF altitude/XY-jump guards — abort hover on implausible pose
#    • Forbidden-zone WP skipping
#
#  Recovery:
#    • Tiered ESCAPE state — reactive direction, on-demand RRT*, row skip
#    • Crash recovery: land → disarm → re-arm → takeoff → resume
#    • OFFBOARD watchdog — auto re-enters OFFBOARD on drop
#
#  Logging:
#    • Verbose decision-making logs with cardinal directions (N/NE/E/...)
#    • Rate-limited [NAV], [AVOID], [LOOK], [ESCAPE], [FSM] traces
# ─────────────────────────────────────────────────────────────────────────────

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
from RRTStarPlanner import RRTStarPlanner

# Configuration constants, arena geometry, and MissionState — see mission_config.py.
# Pure navigation helpers (cardinal labels, segment math) — see nav_helpers.py.
from mission_config import *  # noqa: F401,F403
from nav_helpers import yaw_to_cardinal, sector_cardinals, nearest_on_seg


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
        self._wp_approach_start = time.monotonic()
        self._wp_dest_checked: bool = False        # per-WP live obstacle check flag
        self._stuck_ref_n = 0.0
        self._stuck_ref_e = 0.0
        self._last_min_clearance = SAFE_DIST  # updated each velocity tick (depth-camera)
        self._last_map_clearance = SAFE_DIST  # updated each velocity tick (global map, 360°)
        self._avoidance_since: float | None = None  # time entered CRIT avoidance zone

        self._yellow_streak = 0
        self._red_streak = 0

        self._grid = None
        self._last_cell = (-1, -1)

        # Detection deduplication: track which cells have confirmed barrels per class
        self._counted_cells = {"YELLOW": set(), "RED": set()}
        self._last_detection_frame_id = -1

        # Map update throttle counter
        self._map_tick = 0

        # Path history for backtrack repulsion
        self._path_history: list = []
        self._path_hist_tick = 0

        # Time of last escape (for post-escape look-ahead reduction)
        self._last_escape_time = 0.0

        # Last commanded velocity — shown in NAV status log for speed visibility
        self._last_vn = 0.0
        self._last_ve = 0.0

        # EKF jump detector — tracks previous tick position
        self._last_ekf_n: float | None = None
        self._last_ekf_e: float | None = None

        # AVOID log throttle — print at most 1/s (every 20 ticks at 20 Hz)
        self._avoid_log_tick = 0

        # BOUNDARY log throttle — print at most once per 2 s
        self._boundary_log_time = 0.0

        # Altitude override: climbs when stuck in AVOID-CRIT for >10s
        self._crit_alt_start: float | None = None
        self._alt_bonus_m = 0.0
        self._emerg_log_t = 0.0

        # Consecutive AVOID frames — triggers LOOK_AROUND state when >TRIGGER
        self._avoid_streak = 0

        # NAV status log throttle — print at most once per 5 s during EXPLORE
        self._nav_log_time = 0.0

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

        # Cached first-leg path from Stage 2 validation — avoids redundant RRT* call
        self._cached_first_leg_path = None

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

    async def _rotate_waypoints_to_most_open(self, waypoints):
        """
        Two-stage starting-WP selection:
          Stage 1 — heuristic shortlist: rank by 1/(1+obs_count+0.05*path_len)
                    with a 12m distance cap (fallback uncapped if no nearby WPs).
          Stage 2 — RRT* validation: run RRT* sequentially for the top 5
                    heuristic candidates; stop at first valid path found.
                    Falls back to heuristic winner if all RRT* fail.

        Why two stages: 360-scan map is sparse (~100 points). Heuristic score
        alone says nothing about *reachability* — a "low obs_count" path may
        still cross between pillars the scan didn't catch. RRT* exposes
        unreachable WPs before the drone wastes time crashing into them.
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

        # Stage 1 — heuristic shortlist (12m cap, uncapped fallback)
        # Tuples: (idx, score, path_len, min_path_dist)
        # min_path_dist = closest any obstacle comes to the direct path line.
        # Hard filter: prefer paths with >PATH_CLEAR_M clearance from all obstacles;
        # fall back to best available if nothing clears the threshold.
        PATH_CLEAR_M = 2.0
        candidates = []
        for dist_cap in (12.0, float("inf")):
            raw = []
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
                    min_path_dist = float(np.min(dists_to_path))
                else:
                    obs_count = 0
                    min_path_dist = float("inf")
                score = 1.0 / (1.0 + obs_count + 0.05 * path_len)
                raw.append((idx, score, path_len, min_path_dist))
            if raw:
                # Prefer paths whose direct line stays ≥ PATH_CLEAR_M from all obstacles
                clear = [c for c in raw if c[3] >= PATH_CLEAR_M]
                if clear:
                    candidates = clear
                else:
                    # All paths pass near an obstacle — use best available clearance
                    best_dist = max(c[3] for c in raw)
                    candidates = [c for c in raw if c[3] >= best_dist * 0.9]
                    print(f"[NAV] No path with >={PATH_CLEAR_M}m obstacle clearance "
                          f"(best={best_dist:.1f}m, {len(candidates)} candidate(s))")
                break

        if not candidates:
            # All WPs filtered out — fall back to nearest non-wall, then to order[0]
            for idx in order:
                if not self._is_near_wall(waypoints[idx][0], waypoints[idx][1], extra=0.0):
                    best_idx = idx
                    break
            else:
                best_idx = order[0]
            wp = waypoints[best_idx]
            print(f"[NAV] All-wall fallback start: WP {best_idx} N={wp[0]:.1f} E={wp[1]:.1f}")
            return waypoints[best_idx:] + waypoints[:best_idx]

        # Stage 2 — validate top 5 heuristic candidates with RRT* in parallel
        candidates.sort(key=lambda c: -c[1])  # highest score first
        top_k = candidates[:5]

        bounds = np.array([[self._n_min, self._n_max], [self._e_min, self._e_max]])
        if obs_pts.shape[0] == 0:
            obs_for_rrt = np.array([[self._n_max + 100, self._e_max + 100]])
        else:
            obs_for_rrt = obs_pts

        async def _plan_one(idx, wp):
            try:
                path = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._rrt.plan,
                        [cur_n, cur_e],
                        [wp[0], wp[1]],
                        obs_for_rrt,
                        bounds,
                    ),
                    timeout=6.0,
                )
                return (idx, path)
            except (Exception, asyncio.TimeoutError) as e:
                print(f"[NAV] RRT* validation WP {idx} skipped: {e}")
                return (idx, None)

        print(f"[NAV] Validating top {len(top_k)} WP candidates with RRT* "
              f"(map={obs_pts.shape[0]}pts iter={RRT_MAX_ITER})...")
        # Sequential — concurrent threads fight for Python GIL and all timeout.
        # Stop at first valid path found (Stage 1 score already ranked candidates).
        results = []
        for c in top_k:
            r = await _plan_one(c[0], waypoints[c[0]])
            results.append(r)
            if r[1] is not None:
                for rc in top_k[len(results):]:
                    results.append((rc[0], None))
                break

        # Pre-compute obstacle positions for destination clearance check
        dest_obs = self.mapper.get_global_points()
        MIN_WP_DEST_CLEARANCE = 2.5  # WP destination must be this far from any obstacle

        valid = []
        for idx, path in results:
            if path is None or len(path) < 2:
                continue
            total = 0.0
            for i in range(len(path) - 1):
                total += math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            # Reject WP if destination is inside an obstacle cluster —
            # reactive layer will deadlock against obstacles before drone arrives
            if dest_obs.shape[0] > 0:
                wp_n, wp_e = waypoints[idx][0], waypoints[idx][1]
                dists = np.linalg.norm(dest_obs - np.array([[wp_n, wp_e]]), axis=1)
                if np.min(dists) < MIN_WP_DEST_CLEARANCE:
                    print(f"[NAV] WP {idx} rejected — destination too close to obstacle "
                          f"(min_dist={np.min(dists):.1f}m < {MIN_WP_DEST_CLEARANCE}m)")
                    continue
            valid.append((idx, total))

        if valid:
            valid.sort(key=lambda v: v[1])
            best_idx, best_len = valid[0]
            wp = waypoints[best_idx]
            # Cache the winning path so _plan_first_leg can reuse it without
            # running another RRT* that would compete with zombie threads from
            # the timed-out concurrent _plan_one calls above.
            self._cached_first_leg_path = next(
                (path for ridx, path in results if ridx == best_idx and path is not None),
                None,
            )
            print(
                f"[NAV] RRT*-validated start: WP {best_idx} "
                f"N={wp[0]:.1f} E={wp[1]:.1f} path={best_len:.1f}m "
                f"({len(valid)}/{len(top_k)} candidates feasible)"
            )
        else:
            # All RRT* failed — no cached path to reuse
            self._cached_first_leg_path = None
            # Still apply dest clearance to heuristic list
            best_idx = None
            for c in candidates:
                idx = c[0]
                wp_n, wp_e = waypoints[idx][0], waypoints[idx][1]
                if dest_obs.shape[0] > 0:
                    dists = np.linalg.norm(dest_obs - np.array([[wp_n, wp_e]]), axis=1)
                    if np.min(dists) < MIN_WP_DEST_CLEARANCE:
                        print(f"[NAV] Heuristic WP {idx} rejected — dest too close to obstacle "
                              f"(min={np.min(dists):.1f}m)")
                        continue
                best_idx = idx
                break
            if best_idx is None:
                best_idx = candidates[0][0]  # last resort — all fail clearance
            wp = waypoints[best_idx]
            print(
                f"[NAV] All {len(top_k)} RRT* validations failed — "
                f"heuristic start: WP {best_idx} N={wp[0]:.1f} E={wp[1]:.1f}"
            )

        return waypoints[best_idx:] + waypoints[:best_idx]

    async def _plan_first_leg(self):
        """RRT* from current pose to current WP 0. Inserts intermediate
        sub-WPs at index 0 so the drone navigates around known obstacles
        for the first leg instead of going straight.
        First tries to reuse the Stage 2 validation path to avoid redundant
        RRT* and zombie-thread CPU contention."""
        if not self._waypoints:
            return False
        p = self._pose()
        if p is None:
            return False

        target_wp = self._waypoints[0]
        alt_d = target_wp[2]

        # Fast path: Stage 2 already computed a route — reuse it.
        cached = getattr(self, '_cached_first_leg_path', None)
        if cached is not None and len(cached) > 2:
            self._cached_first_leg_path = None
            intermediate = [(float(pt[0]), float(pt[1]), alt_d) for pt in cached[1:-1]]
            if intermediate:
                self._waypoints = intermediate + self._waypoints
                print(f"[NAV] T={self._elapsed():.0f}s  First-leg: reusing Stage 2 path "
                      f"({len(intermediate)} sub-WPs)")
                return True

        self._cached_first_leg_path = None

        # Slow path: run fresh RRT* (zombie threads may still be running, so use 12s timeout)
        obs_pts = self.mapper.get_global_points()
        if obs_pts.shape[0] == 0:
            obs_pts = np.array([[self._n_max + 100, self._e_max + 100]])
        bounds = np.array([[self._n_min, self._n_max], [self._e_min, self._e_max]])

        try:
            path = await asyncio.wait_for(
                asyncio.to_thread(
                    self._rrt.plan,
                    [p["north"], p["east"]],
                    [target_wp[0], target_wp[1]],
                    obs_pts,
                    bounds,
                ),
                timeout=12.0,
            )
        except (Exception, asyncio.TimeoutError) as e:
            print(f"[NAV] T={self._elapsed():.0f}s  First-leg RRT* skipped: {type(e).__name__}")
            return False

        if path is None or len(path) <= 2:
            print(f"[NAV] T={self._elapsed():.0f}s  First-leg RRT* trivial/failed — direct route")
            return False

        intermediate = [(float(pt[0]), float(pt[1]), alt_d) for pt in path[1:-1]]
        if not intermediate:
            return False

        self._waypoints = intermediate + self._waypoints
        print(f"[NAV] T={self._elapsed():.0f}s  First-leg fresh RRT*: "
              f"{len(intermediate)} sub-WPs inserted")
        return True

    # ------------------------------------------------------------------
    # Phase-transition scan — climb to target altitude then full 360° scan.
    # Call before _start_phase("RED") so Zone 2/3 obstacles are mapped
    # before the drone flies into unknown territory.
    # ------------------------------------------------------------------
    async def _phase_transition_scan(self, target_alt_m, timeout=20.0):
        p = self._pose()
        if p is None:
            return
        target_d = -target_alt_m
        print(f"[TRANS] Climbing to {target_alt_m:.1f}m for pre-phase scan")

        # Climb to target altitude while holding XY
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self._pose()
            if p is None:
                await asyncio.sleep(0.1)
                continue
            if abs(p["down"] - target_d) < 0.4:
                break
            await self.drone.send_position_setpoint(p["north"], p["east"], target_d, p["yaw_deg"])
            await asyncio.sleep(0.1)

        # Clear Phase 1 floor-level obstacles — at 4.5m the drone flies above most of them.
        # Keeping the Phase 1 map fills the 4.5m-altitude area with phantom obstacles,
        # making the drone appear trapped even in open space.
        print("[TRANS] Clearing Phase 1 obstacle map before altitude scan")
        self.mapper.clear()

        # 360° scan at new altitude — rebuilds map from 4.5m perspective
        await self._startup_scan()
        pts = self.mapper.get_global_points()
        print(f"[TRANS] Pre-phase scan done — {pts.shape[0]} obstacle pts in map")

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
    def _filter_obstacle_wps(self, waypoints, min_clearance=3.5):
        """Remove sweep WPs that land inside or too close to known obstacles."""
        obs = self.mapper.get_global_points()
        if obs.shape[0] == 0:
            return waypoints
        kept = []
        for wp in waypoints:
            dists = np.linalg.norm(obs - np.array([[wp[0], wp[1]]]), axis=1)
            if np.min(dists) >= min_clearance:
                kept.append(wp)
        removed = len(waypoints) - len(kept)
        if removed > 0:
            print(f"[NAV] Filtered {removed}/{len(waypoints)} sweep WPs inside obstacle clusters")
        return kept if kept else waypoints  # fallback: keep all if all removed

    def _build_zone_sweep(self, altitude, row_spacing, zones=None):
        down = -altitude
        if zones is None:
            zones = [
                (ZONE1_N, ZONE1_E),
                (ZONE2_N, ZONE2_E),
                (ZONE3_N, ZONE3_E),
            ]
        wps = []
        # WPs sit at WALL_MARGIN + WP_WALL_BUFFER from walls — extra buffer keeps
        # the drone from skimming wall margins (the prior margin-flush layout put
        # WPs literally at the boundary, leading to wall-hug behaviour).
        buf = WALL_MARGIN + WP_WALL_BUFFER
        for (n_lo, n_hi), (e_lo, e_hi) in zones:
            n0 = self._origin_n + n_lo + buf
            n1 = self._origin_n + n_hi - buf
            e0 = self._origin_e + e_lo + buf
            e1 = self._origin_e + e_hi - buf
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
    def _nearest_unvisited_wp(self, cur_n, cur_e):
        """Return index of nearest unvisited WP by Euclidean distance (search from wp_idx+1).
        Prevents backtracking across the arena when next-in-list WP is on the opposite side."""
        best_i = None
        best_dist = float('inf')
        for i in range(self._wp_idx + 1, len(self._waypoints)):
            wp = self._waypoints[i]
            ci, cj = self._ned_to_cell(wp[0], wp[1])
            if self._valid_cell(ci, cj) and self._grid[ci][cj].visited_count == 0:
                d = math.hypot(wp[0] - cur_n, wp[1] - cur_e)
                if d < best_dist:
                    best_dist = d
                    best_i = i
        return best_i

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
        self._wp_approach_start = time.monotonic()
        self._wp_dest_checked = False
        self._reset_stuck()
        if self._consecutive_stuck > 0:
            self._consecutive_stuck = 0   # arriving at any WP = made progress
        self._flip_count = 0
        wp = self._current_wp()
        if wp:
            p_adv = self._pose()
            pos_str = (f"from=({p_adv['north']:.1f},{p_adv['east']:.1f})"
                       if p_adv else "from=?")
            print(f"[NAV] T={self._elapsed():.0f}s  ADVANCE→WP {self._wp_idx}/{len(self._waypoints)}  "
                  f"{pos_str}  →N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m  "
                  f"Tleft={self._time_left():.0f}s  {self.tracker.summary()}  "
                  f"{self._coverage_summary()}")
        # Background plan for the leg AFTER the one we just started
        self._start_rrt_plan(self._wp_idx, self._wp_idx + 1)

    async def _start_phase(self, phase):
        self._phase = phase
        if phase == "YELLOW":
            # Yellow phase: Zone 1 only (ground-level yellow barrels).
            # Zone 2/3 are upper zones — swept at 4.5m during Red phase.
            # Sweeping them here at 1.8m wastes ~5 min on unreachable WPs.
            wps = self._build_zone_sweep(ALT_YELLOW, ROW_SPACING_LOW,
                                         zones=[(ZONE1_N, ZONE1_E)])
            wps = self._filter_obstacle_wps(wps)
            self._waypoints = await self._rotate_waypoints_to_most_open(wps)
            print(f"\n[PHASE 1] Yellow sweep  alt={ALT_YELLOW}m  Zone 1 only  "
                  f"{len(self._waypoints)} waypoints")
        else:
            # Red phase: transition strip in northern Zone 1 → Zone 2+3.
            # The transition strip (N=8→16, full E width) guides the drone from Zone 1
            # center toward the Zone 2/3 passages before attempting distant Zone 2+3 WPs.
            # Without this, Phase 2 WPs (N=28.5) are unreachable from (N=4, E=0) directly.
            ZONE1_NORTH_STRIP_N = (8.0, 16.0)   # northern slice of Zone 1
            ZONE1_NORTH_STRIP_E = (-12.0, 16.0)
            trans_wps = self._build_zone_sweep(ALT_RED, ROW_SPACING_HIGH,
                                               zones=[(ZONE1_NORTH_STRIP_N, ZONE1_NORTH_STRIP_E)])
            zone_wps = self._build_zone_sweep(ALT_RED, ROW_SPACING_HIGH,
                                              zones=[(ZONE2_N, ZONE2_E), (ZONE3_N, ZONE3_E)])
            wps = trans_wps + zone_wps
            wps = self._filter_obstacle_wps(wps)
            self._waypoints = await self._rotate_waypoints_to_most_open(wps)
            print(f"\n[PHASE 2] Red sweep  alt={ALT_RED}m  trans+Zone2+3  "
                  f"{len(self._waypoints)} waypoints ({len(trans_wps)} transition)")
        self._wp_idx        = 0
        self._wp_approach_start = time.monotonic()
        self._yellow_streak = 0
        self._red_streak    = 0
        self._reset_stuck()
        # Plan first leg (current pose → selected WP) — inserts sub-WPs at index 0
        await self._plan_first_leg()
        wp = self._current_wp()
        if wp:
            print(f"[NAV] WP 1/{len(self._waypoints)}  "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} Alt={-wp[2]:.1f}m")
        # Pre-plan next leg in background
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
        self._avoidance_since = None

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
        # When obstacles are within CRIT_DIST, drone may be navigating around them
        # slowly — suppress stuck for up to 15s. After 15s still in CRIT avoidance
        # with no escape, treat as genuinely cornered and allow stuck to fire.
        if getattr(self, '_last_min_clearance', SAFE_DIST) < CRIT_DIST:
            if self._avoidance_since is None:
                self._avoidance_since = time.monotonic()
            if time.monotonic() - self._avoidance_since < 4.0:  # was 8.0
                self._stuck_timer = time.monotonic()
                return False
            # 8s elapsed in CRIT zone → fall through to normal stuck check
        else:
            self._avoidance_since = None

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
    _nearest_on_seg = staticmethod(nearest_on_seg)

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
        # Use fewer iterations for background plans — reduces GIL contention with
        # gz.transport callbacks that causes "User callback queue slow" spam.
        # Direct route is the fallback anyway, so quality loss is acceptable.
        _flight_rrt = RRTStarPlanner(
            safety_margin=RRT_SAFETY_M,
            step_size=RRT_STEP_M,
            max_iter=RRT_FLIGHT_ITER,
        )

        def _plan_sync():
            # KDTree needs ≥1 point; use a dummy far away when map is empty
            if obs_snap.shape[0] == 0:
                obs_pts = np.array([[n_max + 100, e_max + 100]])
            else:
                obs_pts = obs_snap
            bounds = np.array([[n_min, n_max], [e_min, e_max]])
            return _flight_rrt.plan(
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
            path = await asyncio.wait_for(
                asyncio.to_thread(
                    self._rrt.plan,
                    [p["north"], p["east"]],
                    [target_wp[0], target_wp[1]],
                    obs_pts,
                    bounds,
                ),
                timeout=6.0,
            )
        except (Exception, asyncio.TimeoutError) as e:
            print(f"[REPLAN] RRT* skipped: {e}")
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
    # Open-space direction finder — used when all depth sectors are critical
    # ------------------------------------------------------------------
    # Thin wrappers — actual implementation lives in nav_helpers.py.
    _yaw_to_cardinal  = staticmethod(yaw_to_cardinal)
    _sector_cardinals = staticmethod(sector_cardinals)

    def _find_open_escape_direction(self, cur_n, cur_e, goal_n, goal_e):
        """
        Cast 16 rays from current position. For each ray, compute the minimum
        lateral clearance from obstacles in GlobalMapper along the first 6m.
        Pick direction with highest clearance + small goal-alignment bonus.

        This replaces memory-repulsion in the all_critical case so the drone
        actively seeks open space rather than bouncing off obstacles.
        Returns (dn, de) unit vector.
        """
        obs = self.mapper.get_global_points()
        goal_mag = math.hypot(goal_n, goal_e)
        gn = goal_n / goal_mag if goal_mag > 1e-3 else 1.0
        ge = goal_e / goal_mag if goal_mag > 1e-3 else 0.0

        if obs.shape[0] == 0:
            return gn, ge

        # Only obstacles within 8m matter
        d_all = np.hypot(obs[:, 0] - cur_n, obs[:, 1] - cur_e)
        nearby = obs[d_all < 8.0]
        if nearby.shape[0] == 0:
            return gn, ge

        n_rays = 16
        best_score = -float('inf')
        best_dn, best_de = gn, ge

        for i in range(n_rays):
            angle = 2.0 * math.pi * i / n_rays
            dn = math.cos(angle)
            de = math.sin(angle)

            # Project nearby obstacles onto this ray
            proj = (nearby[:, 0] - cur_n) * dn + (nearby[:, 1] - cur_e) * de
            diff_n = (nearby[:, 0] - cur_n) - proj * dn
            diff_e = (nearby[:, 1] - cur_e) - proj * de
            lat = np.hypot(diff_n, diff_e)

            # Obstacles ahead on this ray within 6m
            ahead = (proj > 0.05) & (proj < 6.0)
            min_lat = float(np.min(lat[ahead])) if np.any(ahead) else 6.0

            # Goal-alignment bonus (-1 to +1)
            align = dn * gn + de * ge

            # Clearance dominates; goal bonus breaks ties
            score = min_lat + 0.35 * align

            if score > best_score:
                best_score = score
                best_dn, best_de = dn, de

        return best_dn, best_de

    # ------------------------------------------------------------------
    # Velocity setpoint — goal + avoidance + memory + wall blend → (vn, ve, vd, yaw)
    # ------------------------------------------------------------------
    def _compute_velocity_setpoint(self, pose, target_n, target_e, target_d, depth):
        cur_n, cur_e = pose["north"], pose["east"]
        cur_yaw = pose["yaw_deg"]

        # Emergency: outside outer arena walls — push toward zone 1 centre at VEL_MIN.
        # Does NOT include forbidden zone check — forbidden zone is handled by wall repulsion
        # from the cutout wall segments in ARENA_WALL_SEGS_REL. Including _in_forbidden here
        # (with margin=WALL_MARGIN=2.5m) expands the forbidden zone SOUTH to N=13.5, which
        # blocks the only passages to Zone 2/3 and traps the drone in Zone 1 permanently.
        _outside_outer_walls = (cur_n < self._n_min or cur_n > self._n_max or
                                 cur_e < self._e_min or cur_e > self._e_max)
        if _outside_outer_walls:
            safe_n = self._origin_n + (ZONE1_N[0] + ZONE1_N[1]) / 2.0
            safe_e = self._origin_e + (ZONE1_E[0] + ZONE1_E[1]) / 2.0

            # Guard: if EKF reports position impossibly far from arena (corruption),
            # chasing the "safe" center from a wrong position oscillates wildly.
            # Just hover and let EKF jump detector handle it.
            dist_from_safe = math.hypot(cur_n - safe_n, cur_e - safe_e)
            _now = time.monotonic()
            if dist_from_safe > 60.0:
                if _now - self._boundary_log_time >= 5.0:
                    print(f"[BOUNDARY] T={self._elapsed():.0f}s  EKF position "
                          f"({cur_n:.1f},{cur_e:.1f}) implausibly far — hovering (EKF corrupt?)")
                    self._boundary_log_time = _now
                vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))
                return 0.0, 0.0, vd, cur_yaw

            dn = safe_n - cur_n
            de = safe_e - cur_e
            mag = math.hypot(dn, de)
            if mag > 1e-3:
                dn /= mag
                de /= mag
            yaw = math.degrees(math.atan2(de, dn))
            vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))
            if _now - self._boundary_log_time >= 5.0:
                print(f"[BOUNDARY] T={self._elapsed():.0f}s  "
                      f"({cur_n:.1f},{cur_e:.1f}) near wall — recovering to ({safe_n:.1f},{safe_e:.1f})")
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
                if self._avoid_log_tick % 45 == 1:  # ~3s between prints
                    cl = info["clearance"]
                    fwd_c, left_c, right_c = self._sector_cardinals(cur_yaw)
                    facing = self._yaw_to_cardinal(cur_yaw)
                    goal_c = self._yaw_to_cardinal(
                        math.degrees(math.atan2(goal_e, goal_n)))
                    if cl['center'] < 0.6 or min(cl['left'], cl['right']) < 0.5:
                        action = "→ EMERG slide (handled below)"
                    elif cl['left'] < cl['right']:
                        action = f"→ steer RIGHT (avoid {left_c} wall {cl['left']:.1f}m)"
                    else:
                        action = f"→ steer LEFT (avoid {right_c} wall {cl['right']:.1f}m)"
                    print(f"[AVOID] T={self._elapsed():.0f}s  facing={facing}  goal={goal_c}  "
                          f"pos=({cur_n:.1f},{cur_e:.1f})  "
                          f"{left_c}={cl['left']:.1f}m  "
                          f"{fwd_c}={cl['center']:.1f}m  "
                          f"{right_c}={cl['right']:.1f}m  "
                          f"{action}")

        # Emergency: obstacle < 0.6 m ahead OR < 0.5 m to either side.
        # Slide TOWARD the most open sector rather than backing away — avoids
        # retreating to positions the drone already came from.
        if center_clearance < 0.6 or min(left_clearance, right_clearance) < 0.5:
            yaw_rad = math.radians(cur_yaw)
            fwd_n,  fwd_e  = math.cos(yaw_rad), math.sin(yaw_rad)
            left_n, left_e = math.cos(yaw_rad - math.pi / 2), math.sin(yaw_rad - math.pi / 2)
            right_n, right_e = math.cos(yaw_rad + math.pi / 2), math.sin(yaw_rad + math.pi / 2)
            sectors = [
                (center_clearance, fwd_n,  fwd_e,  "FWD"),
                (left_clearance,   left_n, left_e,  "LEFT"),
                (right_clearance,  right_n, right_e, "RIGHT"),
            ]
            best_clr,  best_n,  best_e,  best_lbl  = max(sectors, key=lambda s: s[0])
            worst_clr, worst_n, worst_e, worst_lbl = min(sectors, key=lambda s: s[0])
            vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))
            if best_clr > SAFE_DIST:
                # Open sector available — slide into it immediately
                slide_n, slide_e = best_n, best_e
                action_str = f"SLIDE→{best_lbl} ({best_clr:.1f}m open)"
            else:
                # All sectors tight — back away from worst
                slide_n, slide_e = -worst_n, -worst_e
                action_str = f"BACK from {worst_lbl} ({worst_clr:.1f}m)"
            _now_em = time.monotonic()
            if _now_em - self._emerg_log_t >= 2.0:
                fwd_c, left_c, right_c = self._sector_cardinals(cur_yaw)
                print(f"[AVOID-EMERG] T={self._elapsed():.0f}s  "
                      f"facing={self._yaw_to_cardinal(cur_yaw)}  "
                      f"{left_c}={left_clearance:.1f}m  "
                      f"{fwd_c}={center_clearance:.1f}m  "
                      f"{right_c}={right_clearance:.1f}m  "
                      f"→ {action_str}")
                self._emerg_log_t = _now_em
            return VEL_MIN * slide_n, VEL_MIN * slide_e, vd, cur_yaw

        # Memory-map repulsion from GlobalMapper
        mem_n, mem_e = self.mapper.get_repulsion_vector(cur_n, cur_e, MAP_INFLUENCE_M)
        if not (math.isfinite(mem_n) and math.isfinite(mem_e)):
            mem_n, mem_e = 0.0, 0.0

        # Geometric wall repulsion — always active, no camera FOV dependency
        wall_n, wall_e = self._compute_wall_repulsion(cur_n, cur_e)

        # Path history repulsion — penalize returning to recently visited positions.
        # Pushes drone away from cells it flew through in the last ~16s, preventing
        # circles and backtracking. Heavy weight (W_PATH_HIST=1.5) per user request.
        hist_n, hist_e = 0.0, 0.0
        if self._path_history:
            for (hn, he) in self._path_history:
                hd = math.hypot(cur_n - hn, cur_e - he)
                if 0.3 < hd < PATH_HIST_INFL:
                    w = 1.0 - hd / PATH_HIST_INFL
                    hist_n += w * (cur_n - hn) / hd
                    hist_e += w * (cur_e - he) / hd
            hmag = math.hypot(hist_n, hist_e)
            if hmag > 1e-6:
                hist_n /= hmag
                hist_e /= hmag

        # Origin (spawn) repulsion — heavy penalty for returning to start.
        # Drone keeps wandering back to spawn; this kills that pattern.
        orig_n, orig_e = 0.0, 0.0
        od = math.hypot(cur_n - self._origin_n, cur_e - self._origin_e)
        if 0.3 < od < ORIGIN_INFL_M:
            w_orig = 1.0 - od / ORIGIN_INFL_M
            orig_n = w_orig * (cur_n - self._origin_n) / od
            orig_e = w_orig * (cur_e - self._origin_e) / od

        # All three sectors critical — use memory repulsion to escape obstacle cluster
        all_critical = (left_clearance < CRIT_DIST and center_clearance < CRIT_DIST
                        and right_clearance < CRIT_DIST)

        # Use minimum clearance across all sectors for speed and avoidance weight.
        # Corner clips happen because center is clear but a side is close — using
        # center-only let the drone fly full speed into a wall corner.
        min_clearance = min(center_clearance, left_clearance, right_clearance)
        self._last_min_clearance = min_clearance

        # Map-based proximity — 360° awareness regardless of camera FOV.
        # When the drone rounds a corner, the camera hasn't seen the new pillar face
        # yet, but the map already has those points from the startup scan.
        obs_pts_vel = self.mapper.get_global_points()
        if obs_pts_vel.shape[0] > 0:
            _dists = np.hypot(obs_pts_vel[:, 0] - cur_n, obs_pts_vel[:, 1] - cur_e)
            map_clearance = float(np.min(_dists))
        else:
            map_clearance = SAFE_DIST
        self._last_map_clearance = map_clearance

        # Dynamic memory-repulsion weight: ramps from W_MEM_AVOID up to 1.0 as
        # map-known obstacles close in (helps avoid unseen pillar faces on corners).
        # Capped at 1.0 — was 2.0, which overpowered goal and caused wandering.
        if map_clearance < SAFE_DIST:
            _t_map = max(0.0, 1.0 - map_clearance / SAFE_DIST)
            w_mem_dyn = W_MEM_AVOID + _t_map * (1.0 - W_MEM_AVOID)
        else:
            w_mem_dyn = W_MEM_AVOID

        # Speed and w_avoid use camera clearance ONLY (not map).
        # Map clearance is too dense to use for speed — it permanently reports
        # <SAFE_DIST and locks drone into VEL_MIN crawl the entire mission.
        w_avoid = 0.1 if min_clearance >= SAFE_DIST else W_AVOID

        # Blend: goal + avoidance + memory + wall
        if all_critical:
            # All depth sectors blocked — seek most open direction in map
            open_n, open_e = self._find_open_escape_direction(
                cur_n, cur_e, goal_n, goal_e
            )
            if self._avoid_log_tick % 45 == 1:
                open_card = self._yaw_to_cardinal(math.degrees(math.atan2(open_e, open_n)))
                goal_card = self._yaw_to_cardinal(math.degrees(math.atan2(goal_e, goal_n)))
                fwd_c, left_c, right_c = self._sector_cardinals(cur_yaw)
                print(f"[AVOID] T={self._elapsed():.0f}s  ALL SECTORS BLOCKED  "
                      f"facing={self._yaw_to_cardinal(cur_yaw)}  "
                      f"pos=({cur_n:.1f},{cur_e:.1f})  "
                      f"{left_c}={left_clearance:.1f}m  "
                      f"{fwd_c}={center_clearance:.1f}m  "
                      f"{right_c}={right_clearance:.1f}m  "
                      f"→ seek open={open_card}  goal={goal_card}")
            blend_n = (open_n + W_WALL * wall_n + W_PATH_HIST * hist_n
                       + W_ORIGIN_REPEL * orig_n)
            blend_e = (open_e + W_WALL * wall_e + W_PATH_HIST * hist_e
                       + W_ORIGIN_REPEL * orig_e)
        else:
            # Binary goal weight: full drive toward WP unless camera is blocked.
            # SOFT_DIST gradual ramp caused goal_w=0.5 in normal corridors
            # (cam_clr~1.4m), which stalled the drone immediately. SOFT_DIST
            # is kept for yaw tilt only (gives depth camera earlier side-wall view).
            goal_w = 0.4 if blocked else 1.0
            blend_n = (goal_w * goal_n + w_avoid * avoid_n + w_mem_dyn * mem_n
                       + W_WALL * wall_n + W_PATH_HIST * hist_n
                       + W_ORIGIN_REPEL * orig_n)
            blend_e = (goal_w * goal_e + w_avoid * avoid_e + w_mem_dyn * mem_e
                       + W_WALL * wall_e + W_PATH_HIST * hist_e
                       + W_ORIGIN_REPEL * orig_e)

        mag = math.hypot(blend_n, blend_e)
        if mag > 1e-3:
            blend_n /= mag
            blend_e /= mag
        else:
            blend_n, blend_e = goal_n, goal_e

        # Guard: abort to hover if blend is non-finite
        if not (math.isfinite(blend_n) and math.isfinite(blend_e)):
            return 0.0, 0.0, 0.0, cur_yaw

        # --- Speed scaling: center sector clearance ---
        # min(L,C,R) crawls the drone when flying along a corridor wall (L=0.6m kills speed
        # even with C=3.0m). Use center clearance — only slow down if the path ahead is blocked.
        # Apply a cap when BOTH sides are critical (< CRIT_DIST) to prevent overshooting turns.
        speed_clr = center_clearance
        if left_clearance < CRIT_DIST and right_clearance < CRIT_DIST:
            speed_clr = min(speed_clr, SAFE_DIST * 0.8)   # both walls close — cap at ~80% SAFE
        if speed_clr >= SAFE_DIST:
            speed = VEL_MAX
        elif speed_clr > CRIT_DIST:
            t = (speed_clr - CRIT_DIST) / (SAFE_DIST - CRIT_DIST)
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
        # Store for NAV status log
        self._last_vn = vn
        self._last_ve = ve

        # --- Altitude P-controller ---
        # NED: down < 0 when above ground; positive vd moves toward ground
        vd = max(-ALT_VEL_MAX, min(ALT_VEL_MAX, ALT_KP * (target_d - pose["down"])))

        # --- Yaw control: motion direction + bias toward closest side obstacle ---
        # Pure motion-direction yaw means depth camera always looks at the goal,
        # so side obstacles enter FOV only after the drone is already close.
        # Biasing yaw toward the tighter side makes the depth camera track that
        # obstacle, giving better local awareness for wall-hug scenarios.
        desired_yaw = math.degrees(math.atan2(blend_e, blend_n))
        if not math.isfinite(desired_yaw):
            desired_yaw = cur_yaw

        side_tilt_max = 35.0
        # Yaw tilts toward tighter side starting at SOFT_DIST — gives depth camera
        # earlier visibility of side obstacles (was SAFE_DIST, which was too late).
        if (left_clearance < right_clearance
                and left_clearance < center_clearance
                and left_clearance < SOFT_DIST):
            bias = side_tilt_max * (1.0 - left_clearance / SOFT_DIST)
            desired_yaw -= bias   # yaw left in NED clockwise convention
        elif (right_clearance < left_clearance
                and right_clearance < center_clearance
                and right_clearance < SOFT_DIST):
            bias = side_tilt_max * (1.0 - right_clearance / SOFT_DIST)
            desired_yaw += bias   # yaw right

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
                print(f"\n[PHASE 1 DONE] T={self._elapsed():.0f}s  {self.tracker.summary()}  "
                      f"{self._coverage_summary()}")
                if self._time_left() > 120:
                    # Climb to RED altitude + scan before entering Zone 2/3 blind.
                    # Phase 1 map has only Zone 1 data — without this scan the drone
                    # flies into Zone 2/3 with zero obstacle awareness and crashes.
                    await self._phase_transition_scan(ALT_RED)
                    await self._start_phase("RED")
                elif self._time_left() > 60:
                    print("[MISSION] Not enough time for full red sweep — skipping pre-scan.")
                    await self._start_phase("RED")
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

        # EKF jump guard: if position teleports >5m in one control tick (impossible
        # at VEL_MAX=1.0 m/s, which gives 0.07m/tick), EKF is corrupted — hover.
        _last_n = getattr(self, '_last_ekf_n', None)
        _last_e = getattr(self, '_last_ekf_e', None)
        if _last_n is not None:
            _jump = math.hypot(cur_n - _last_n, cur_e - _last_e)
            if _jump > 5.0:
                print(f"[EKF] T={self._elapsed():.0f}s  Position jump {_jump:.1f}m detected "
                      f"({_last_n:.1f},{_last_e:.1f})→({cur_n:.1f},{cur_e:.1f}) — "
                      f"hovering 3s for EKF to settle")
                for _ in range(45):
                    p_hov = self._pose()
                    if p_hov and self.state.is_in_offboard:
                        await self.drone.send_velocity(0.0, 0.0, 0.0, p_hov["yaw_deg"])
                    await asyncio.sleep(0.1)
                self._reset_stuck()
                self._last_ekf_n = None   # clear so next tick gets a fresh baseline
                return
        self._last_ekf_n = cur_n
        self._last_ekf_e = cur_e

        # EKF altitude corruption guard: if reported altitude is implausible
        # (>target+8m or below ground), altitude controller commands max vz,
        # causing a crash dive. Zero vz and wait for EKF to settle instead.
        _alt_m = -pose["down"]
        _target_alt = ALT_RED if self._phase == "RED" else ALT_YELLOW
        if _alt_m > _target_alt + 8.0 or _alt_m < -0.5:
            # Irrecoverable: altitude so extreme EKF will never self-correct.
            # Hovering just wastes the remaining mission time. End immediately.
            if _alt_m < -50.0 or _alt_m > 200.0:
                print(f"[EKF] FATAL T={self._elapsed():.0f}s  "
                      f"Altitude {_alt_m:.1f}m — irrecoverable EKF corruption. "
                      f"Ending mission immediately.")
                self._state = MissionState.DONE
                return
            _now_a = time.monotonic()
            if _now_a - getattr(self, '_ekf_alt_log_t', 0.0) >= 3.0:
                print(f"[EKF] T={self._elapsed():.0f}s  Altitude {_alt_m:.1f}m implausible "
                      f"(target={_target_alt}m) — hovering, waiting for EKF settle")
                self._ekf_alt_log_t = _now_a
            for _ in range(30):
                p_hov = self._pose()
                if p_hov and self.state.is_in_offboard:
                    await self.drone.send_velocity(0.0, 0.0, 0.0, p_hov["yaw_deg"])
                await asyncio.sleep(0.1)
            self._reset_stuck()
            self._last_ekf_n = None
            return

        self._update_grid(pose)

        if self._arrived(wp):
            self._try_apply_rrt_plan()
            self._advance_wp()
            return

        approach_elapsed = time.monotonic() - self._wp_approach_start

        # One-shot live obstacle check per WP — skips WPs inside pillar clusters
        # discovered AFTER startup scan (filter_obstacle_wps only ran with 101 pts)
        if not self._wp_dest_checked:
            self._wp_dest_checked = True
            obs_live = self.mapper.get_global_points()
            if obs_live.shape[0] >= 5:
                dists_live = np.linalg.norm(obs_live - np.array([[wp[0], wp[1]]]), axis=1)
                if np.min(dists_live) < 3.5:
                    print(f"[NAV] WP {self._wp_idx} dest blocked by live map "
                          f"(min={np.min(dists_live):.1f}m) — skipping")
                    self._advance_wp()
                    return

        if approach_elapsed > WP_APPROACH_TIMEOUT:
            p_to = self._pose()
            clr = getattr(self, '_last_min_clearance', 9.9)
            obs_to = self.mapper.get_global_points()
            obs_near_wp = (int(np.min(np.linalg.norm(obs_to - np.array([[wp[0], wp[1]]]), axis=1)))
                           if obs_to.shape[0] > 0 else 999)
            horiz_to = math.hypot(p_to["north"] - wp[0], p_to["east"] - wp[1]) if p_to else -1
            print(f"[NAV] WP {self._wp_idx} approach timeout ({approach_elapsed:.0f}s) — "
                  f"dist={horiz_to:.1f}m clr={clr:.1f}m obs_at_dest={obs_near_wp}m — skipping")
            # Jump to nearest unvisited WP (by Euclidean distance) rather than next
            # in list — prevents backtracking to the opposite side of the arena when
            # the sequential boustrophedon wraps around after a skip.
            jumped = False
            if self._grid is not None and p_to is not None:
                skip_to = self._nearest_unvisited_wp(p_to["north"], p_to["east"])
                if skip_to is not None and skip_to > self._wp_idx:
                    wp_check = self._waypoints[skip_to]
                    skipped = skip_to - self._wp_idx - 1
                    if skipped > 0:
                        print(f"[NAV] Nearest unvisited → WP {skip_to}/{len(self._waypoints)} "
                              f"N={wp_check[0]:.1f} E={wp_check[1]:.1f} "
                              f"(skipped {skipped} farther WPs)")
                    self._wp_idx = skip_to
                    self._wp_approach_start = time.monotonic()
                    self._wp_dest_checked = False
                    self._reset_stuck()
                    jumped = True
            if not jumped:
                self._advance_wp()
            return

        if self._check_stuck():
            p_stk = self._pose()
            pos_stk = f"({p_stk['north']:.1f},{p_stk['east']:.1f})" if p_stk else "?"
            print(f"[FSM] T={self._elapsed():.0f}s  Stuck at {pos_stk} "
                  f"WP {self._wp_idx}→N={wp[0]:.1f} E={wp[1]:.1f} "
                  f"clr={getattr(self,'_last_min_clearance',9.9):.1f}m → ESCAPE")
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
        if DETECTION_ENABLED:
            self._run_detection()

        # Rate-limited NAV status log (every 10s) — verbose decision-making detail
        _nav_now = time.monotonic()
        if _nav_now - self._nav_log_time >= 10.0:
            self._nav_log_time = _nav_now
            horiz = math.hypot(pose["north"] - wp[0], pose["east"] - wp[1])
            obs_near = self.mapper.get_global_points()
            obs_count_5m = (
                int(np.sum(np.linalg.norm(
                    obs_near - np.array([[cur_n, cur_e]]), axis=1) < 5.0))
                if obs_near.shape[0] > 0 else 0
            )
            spd = math.hypot(getattr(self, '_last_vn', 0.0), getattr(self, '_last_ve', 0.0))
            hdg_deg = math.degrees(math.atan2(
                getattr(self, '_last_ve', 0.0), getattr(self, '_last_vn', 0.0)))
            cam_clr = getattr(self, '_last_min_clearance', 9.9)
            map_clr = getattr(self, '_last_map_clearance', 9.9)
            eff_clr = min(cam_clr, map_clr)
            if eff_clr < CRIT_DIST:
                status = "AVOID-CRIT"
            elif eff_clr < SAFE_DIST:
                status = "AVOID-SLOW"
            else:
                status = "CLEAR"
            # Cardinal labels for depth sectors and movement heading
            facing = self._yaw_to_cardinal(pose["yaw_deg"])
            fwd_c, left_c, right_c = self._sector_cardinals(pose["yaw_deg"])
            hdg_card = self._yaw_to_cardinal(hdg_deg)
            goal_card = self._yaw_to_cardinal(math.degrees(math.atan2(
                wp[1] - cur_e, wp[0] - cur_n)))
            # Why drone may not be going straight toward WP
            if status == "AVOID-CRIT":
                decision = f"BLOCKED — all sectors <{CRIT_DIST}m, using escape direction"
            elif status == "AVOID-SLOW":
                decision = f"SLOWING — cam {cam_clr:.1f}m <{SAFE_DIST}m, blending avoidance"
            else:
                decision = f"FREE — driving toward WP ({goal_card}) at {spd:.2f}m/s"
            print(f"\n[NAV] T={self._elapsed():.0f}s  [{status}]  phase={self._phase}  "
                  f"pos=({cur_n:.1f},{cur_e:.1f}) alt={-pose['down']:.1f}m  "
                  f"facing={facing} moving={hdg_card}\n"
                  f"      WP {self._wp_idx}/{len(self._waypoints)} "
                  f"→ N={wp[0]:.1f} E={wp[1]:.1f} ({goal_card})  dist={horiz:.1f}m  "
                  f"t_wp={approach_elapsed:.0f}s\n"
                  f"      cam: {left_c}={cam_clr:.1f}m(L)  {fwd_c}={cam_clr:.1f}m(C)  "
                  f"{right_c}={cam_clr:.1f}m(R)  map_clr={map_clr:.1f}m  "
                  f"alt_bonus=+{self._alt_bonus_m:.1f}m\n"
                  f"      Decision: {decision}  obs5m={obs_count_5m}  "
                  f"map={obs_near.shape[0]}pts")

        # Record path history every PATH_HIST_INTERVAL ticks for backtrack repulsion
        self._path_hist_tick += 1
        if self._path_hist_tick % PATH_HIST_INTERVAL == 0:
            self._path_history.append((cur_n, cur_e))
            if len(self._path_history) > PATH_HIST_LEN:
                self._path_history.pop(0)

        # Altitude override: when stuck in AVOID-CRIT >10s, climb 0.5m steps
        # up to ALT_MAX_EXPLORE. Lets drone find routes above obstacles.
        eff_clr_now = min(getattr(self, '_last_min_clearance', 9.9),
                          getattr(self, '_last_map_clearance', 9.9))
        phase_base_alt = ALT_RED if self._phase == "RED" else ALT_YELLOW
        if eff_clr_now < CRIT_DIST:
            if self._crit_alt_start is None:
                self._crit_alt_start = time.monotonic()
            elif time.monotonic() - self._crit_alt_start > 10.0:
                new_bonus = min(ALT_MAX_EXPLORE - phase_base_alt,
                                self._alt_bonus_m + 0.5)
                if new_bonus > self._alt_bonus_m + 0.01:
                    self._alt_bonus_m = new_bonus
                    print(f"[ALT] T={self._elapsed():.0f}s  "
                          f"AVOID-CRIT >10s — climbing to "
                          f"{phase_base_alt + self._alt_bonus_m:.1f}m "
                          f"(+{self._alt_bonus_m:.1f}m bonus)")
                self._crit_alt_start = time.monotonic()
        else:
            self._crit_alt_start = None
            if self._alt_bonus_m > 0.01:
                self._alt_bonus_m = max(0.0, self._alt_bonus_m - 0.05)

        # Track consecutive blocked/avoid frames. After AVOID_STREAK_TRIGGER
        # ticks (~2s) of being blocked, drone enters LOOK_AROUND to do a 360°
        # depth scan — narrow camera FOV can't see lateral paths otherwise.
        if eff_clr_now < SAFE_DIST:
            self._avoid_streak += 1
            if self._avoid_streak > AVOID_STREAK_TRIGGER:
                print(f"[FSM] T={self._elapsed():.0f}s  AVOID streak "
                      f"{self._avoid_streak} >{AVOID_STREAK_TRIGGER} "
                      f"→ LOOK_AROUND")
                self._avoid_streak = 0
                self._state = MissionState.LOOK_AROUND
                return
        else:
            self._avoid_streak = max(0, self._avoid_streak - 1)

        # Carrot point: target a point LOOKAHEAD_DIST ahead along path to WP.
        # Prevents lunging at distant WPs; makes movement smoother and more
        # reactive to local obstacles.
        carrot_n, carrot_e, carrot_d = self._carrot_point(pose, wp)
        carrot_d -= self._alt_bonus_m   # NED: subtract = fly higher

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

            if DETECTION_ENABLED:
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
    # State tick: LOOK_AROUND
    # ------------------------------------------------------------------
    # Triggered when AVOID persists >2s. Depth camera FOV is ~60° — so when
    # facing a pillar, all three L/C/R sectors see it and drone thinks every
    # direction is blocked. Rotating in place to scan all 8 cardinal directions
    # reveals open lateral paths that a single-frame view misses.
    async def _tick_look_around(self):
        p = self._pose()
        if p is None:
            self._state = MissionState.EXPLORE
            return
        hold_n, hold_e, hold_d = p["north"], p["east"], p["down"]
        base_yaw = p["yaw_deg"]
        print(f"[LOOK] T={self._elapsed():.0f}s  360° depth scan from "
              f"({hold_n:.1f},{hold_e:.1f}) — narrow FOV → must look around")

        best_yaw, best_clr = base_yaw, 0.0
        scan_results = []
        for delta in range(0, 360, 45):
            yaw = (base_yaw + delta) % 360.0
            try:
                await self.drone.rotate_to_yaw(yaw)
            except Exception:
                pass
            # Hold position + settle
            for _ in range(6):
                pp = self._pose()
                if pp and self.state.is_in_offboard:
                    await self.drone.send_position_setpoint(hold_n, hold_e, hold_d, yaw)
                await asyncio.sleep(0.06)

            # Sample depth several times, take max (most optimistic clearance)
            clrs = []
            for _ in range(3):
                pp = self._pose()
                depth = self._sanitize_depth(self.depth_rx.get_frame())
                if depth is not None and pp is not None:
                    try:
                        _, _, _, info = self.planner.compute_position_ned(
                            depth, pp, step_size=1.0)
                        clrs.append(info["clearance"]["center"])
                    except Exception:
                        pass
                await asyncio.sleep(0.05)
            clr = max(clrs) if clrs else 0.0
            card = self._yaw_to_cardinal(yaw)
            scan_results.append((yaw, card, clr))
            print(f"[LOOK]   yaw={yaw:5.0f}° ({card:>2})  cam_clr={clr:.1f}m")
            if clr > best_clr:
                best_clr = clr
                best_yaw = yaw

        best_card = self._yaw_to_cardinal(best_yaw)
        print(f"[LOOK] BEST: {best_card} ({best_yaw:.0f}°)  cam_clr={best_clr:.1f}m")

        if best_clr < 1.5:
            # No open direction — escalate to ESCAPE
            print("[LOOK] No clear direction — falling back to ESCAPE")
            self._state = MissionState.ESCAPE
            return

        # Commit a short burst in the best direction. Reject directions
        # heading toward origin if drone is already >4m from spawn.
        burst_dist = min(best_clr - 0.5, LOOK_BURST_M)
        target_n = hold_n + burst_dist * math.cos(math.radians(best_yaw))
        target_e = hold_e + burst_dist * math.sin(math.radians(best_yaw))
        d_from_origin_now = math.hypot(hold_n - self._origin_n, hold_e - self._origin_e)
        d_from_origin_new = math.hypot(target_n - self._origin_n, target_e - self._origin_e)
        if d_from_origin_now > 4.0 and d_from_origin_new < d_from_origin_now - 0.5:
            # Burst heads back toward origin — pick next-best direction instead
            scan_results.sort(key=lambda s: -s[2])
            for yaw_c, card_c, clr_c in scan_results[1:]:
                if clr_c < 1.5:
                    break
                tn = hold_n + min(clr_c - 0.5, LOOK_BURST_M) * math.cos(math.radians(yaw_c))
                te = hold_e + min(clr_c - 0.5, LOOK_BURST_M) * math.sin(math.radians(yaw_c))
                d_new = math.hypot(tn - self._origin_n, te - self._origin_e)
                if d_new >= d_from_origin_now - 0.5:
                    print(f"[LOOK] BEST→origin; switching to {card_c} ({yaw_c:.0f}°) "
                          f"cam_clr={clr_c:.1f}m")
                    best_yaw, best_card, best_clr = yaw_c, card_c, clr_c
                    burst_dist = min(best_clr - 0.5, LOOK_BURST_M)
                    target_n, target_e = tn, te
                    break

        print(f"[LOOK] Commit burst {burst_dist:.1f}m {best_card} "
              f"→ ({target_n:.1f},{target_e:.1f})")
        try:
            await self.drone.rotate_to_yaw(best_yaw)
        except Exception:
            pass
        # Send position setpoint and wait for arrival (or 5s timeout)
        t_burst_start = time.monotonic()
        while time.monotonic() - t_burst_start < 5.0:
            pp = self._pose()
            if pp is None:
                break
            d_remain = math.hypot(pp["north"] - target_n, pp["east"] - target_e)
            if d_remain < 0.6:
                break
            if self.state.is_in_offboard:
                await self.drone.send_position_setpoint(target_n, target_e, hold_d, best_yaw)
            await asyncio.sleep(0.1)
        self._last_escape_time = time.monotonic()

        # After burst — skip to nearest unvisited WP so drone resumes coverage
        p2 = self._pose()
        if p2 is not None and self._grid is not None:
            skip_to = self._nearest_unvisited_wp(p2["north"], p2["east"])
            if skip_to is not None:
                self._wp_idx = skip_to
                nwp = self._current_wp()
                if nwp:
                    print(f"[LOOK] → nearest unvisited WP {self._wp_idx} "
                          f"N={nwp[0]:.1f} E={nwp[1]:.1f}")

        self._avoid_streak = 0
        self._reset_stuck()
        print("[FSM] LOOK_AROUND → EXPLORE")
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
        p_esc = self._pose()
        pos_esc = f"({p_esc['north']:.1f},{p_esc['east']:.1f})" if p_esc else "?"
        print(f"[ESCAPE] T={self._elapsed():.0f}s  Tier {tier} at {pos_esc} "
              f"WP={self._wp_idx} clr={getattr(self,'_last_min_clearance',9.9):.1f}m")

        if tier == 1:
            # Reactive escape — random non-wall direction
            await self._escape_stuck()
            p = self._pose()
            if p is not None:
                await self.drone.send_position_setpoint(
                    p["north"], p["east"], p["down"], p["yaw_deg"]
                )
            await asyncio.sleep(2.0)
            # Skip to nearest unvisited WP — avoids retrying same blocked direction
            p = self._pose()
            skip_to = (self._nearest_unvisited_wp(p["north"], p["east"])
                       if (p is not None and self._grid is not None) else None)
            if skip_to is not None:
                self._wp_idx = skip_to
                nwp = self._current_wp()
                if nwp:
                    print(f"[ESCAPE] T1 — skipping stuck WP → nearest unvisited WP {self._wp_idx}/"
                          f"{len(self._waypoints)} N={nwp[0]:.1f} E={nwp[1]:.1f}")
            elif self._wp_idx < len(self._waypoints) - 1:
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
        # Only call offboard.stop() when actually in OFFBOARD — calling it while
        # already out of OFFBOARD causes PX4 to log "not-existing command 176" spam.
        if self.state.is_in_offboard:
            print("[RECOVERY] Stage 1 — stopping OFFBOARD")
            try:
                await self.drone.drone.offboard.stop()
            except Exception as e:
                print(f"[RECOVERY]   offboard.stop() failed (ignored): {e}")
            await asyncio.sleep(1.0)
        else:
            print("[RECOVERY] Stage 1 — not in OFFBOARD, skipping stop")
            await asyncio.sleep(0.3)

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

        # ── Stage 3: wait for disarm / PX4 settle + EKF altitude to stabilize ─
        print("[RECOVERY] Stage 3 — waiting for disarm / settle (3 s)")
        await asyncio.sleep(3.0)
        # Wait for EKF altitude to report near ground level.
        # A corrupted EKF (e.g. alt=-1.16m = underground) will cause OFFBOARD
        # to be rejected by PX4 even after successful re-arm.
        for _ in range(90):   # up to 45 s — SITL EKF needs longer after hard crash
            p = self._pose()
            if p is not None and -0.5 <= p["down"] <= 1.5:
                break
            await asyncio.sleep(0.5)
        else:
            print("[RECOVERY] EKF altitude unstable after 45s — proceeding anyway (may abort)")

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

        # Clear stale obstacle map — EKF position shifted during crash so old obstacle
        # points are in the wrong reference frame and produce bad repulsion vectors.
        print("[RECOVERY] Clearing stale obstacle map")
        self.mapper.clear()

        # Reset stuck / flip state from new position
        self._reset_stuck()
        self._flip_count = 0
        self._was_in_offboard = False   # will flip back True once telemetry confirms OFFBOARD

        # EKF XY stability check — verify position is not still jumping before we scan.
        # If two consecutive readings differ by >3m the EKF is still corrupted; wait up to 20s.
        _prev_rn: float | None = None
        _prev_re: float | None = None
        for _ in range(40):
            p = self._pose()
            if p is not None:
                rn, re = p["north"], p["east"]
                if _prev_rn is not None:
                    jump = math.hypot(rn - _prev_rn, re - _prev_re)
                    if jump < 3.0:
                        break
                    print(f"[RECOVERY] EKF XY still unstable (jump={jump:.1f}m) — waiting…")
                _prev_rn, _prev_re = rn, re
            await asyncio.sleep(0.5)
        else:
            print("[RECOVERY] EKF XY did not stabilise in 20s — proceeding anyway")

        # Rebuild obstacle awareness with a fresh 360° scan from new position,
        # then regenerate waypoints so drone starts from a valid local context
        # instead of resuming a potentially unreachable Zone 2/3 WP.
        print(f"[RECOVERY] Rebuilding map + restarting phase={self._phase} from new position")
        await self._startup_scan()
        await self._start_phase(self._phase)
        print(f"[RECOVERY] Phase restarted — WP 0/{len(self._waypoints)}")

    # ------------------------------------------------------------------
    # Dispatch-table control loop
    # ------------------------------------------------------------------
    async def _control_loop(self):
        dt = 1.0 / CONTROL_HZ

        _dispatch = {
            MissionState.EXPLORE:     self._tick_explore,
            MissionState.ESCAPE:      self._tick_escape,
            MissionState.LOOK_AROUND: self._tick_look_around,
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
                                        MissionState.ESCAPE,
                                        MissionState.LOOK_AROUND)):
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
                # Send velocity=0 — minimal input, no dependence on EKF altitude accuracy.
                # Position setpoints with corrupted EKF altitude (e.g. EKF says 12m when
                # drone is at 1.8m) cause violent descent commands and worsen the crash.
                p_flip = self._pose()
                if p_flip is not None and self.state.is_in_offboard:
                    try:
                        await self.drone.send_velocity(0.0, 0.0, 0.0, p_flip["yaw_deg"])
                    except Exception:
                        await asyncio.sleep(0.1)
                else:
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
                    # Soft recovery: hold current EKF position for 3s via velocity=0.
                    # Using WP target altitude (e.g. -1.8m) when EKF reports 12m causes
                    # the position controller to command a violent 10m descent → second crash.
                    # velocity=0 means "stop moving" regardless of EKF accuracy.
                    print(
                        f"[RECOVERY] Soft recovery flip #{self._flip_count} "
                        f"(alt={alt_m:.1f}m) — velocity=0 hold 3s then resuming"
                    )
                    for _ in range(30):   # 3 s at ~10 Hz
                        p_now = self._pose()
                        yaw_now = p_now["yaw_deg"] if p_now else 0.0
                        try:
                            await self.drone.send_velocity(0.0, 0.0, 0.0, yaw_now)
                        except Exception:
                            pass
                        await asyncio.sleep(0.1)

            # OFFBOARD persistence watchdog (Phase B):
            # If PX4 drops OFFBOARD mid-mission while armed and airborne, attempt
            # automatic re-entry (no full re-arm). Up to 3 attempts.
            # If grounded/disarmed, fall through to crash detection above.
            if (not self.state.is_in_offboard
                    and self._state in (MissionState.EXPLORE, MissionState.ESCAPE,
                                        MissionState.LOOK_AROUND)):
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

        await self._start_phase("YELLOW")
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
    import os
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not model_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in ("yolov10n.pt", "yolov8n.pt", "barrels.pt"):
            p = os.path.join(script_dir, candidate)
            if os.path.isfile(p):
                model_path = p
                break
    if model_path:
        print(f"[CONFIG] YOLO model: {model_path}")
    else:
        print("[CONFIG] No YOLO model found — using HSV colour detection only")

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
