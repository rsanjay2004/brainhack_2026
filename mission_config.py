# mission_config.py
#
# All configuration constants, arena geometry, and the MissionState enum used
# by qualifier_main.py. Split out so qualifier_main.py stays focused on the
# state-machine logic rather than tunable parameters.
#
# When the official arena map drops, swap the geometry block below; nothing
# else in this file should need to change.

from enum import Enum

import numpy as np

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
W_WALL       = 1.0   # continuous geometric wall repulsion weight
WALL_INF_M   = 5.0   # wall influence radius — starts repulsion earlier
WP_WALL_BUFFER = 1.0 # extra buffer beyond WALL_MARGIN when generating sweep WPs
# ─────────────────────────────────────────────────────────────────────────────

# RRT* path planning
RRT_MAX_ITER    = 2000  # iterations for startup WP validation
RRT_FLIGHT_ITER = 500   # iterations for background plans during flight
RRT_SAFETY_M = 1.0   # obstacle clearance margin (m)
RRT_STEP_M   = 1.5   # tree extension step size (m)

# Crash recovery
MAX_RESTARTS = 2      # max re-attempts within the 10-minute window

# Altitudes
ALT_YELLOW    = 1.8    # low — ground-level yellow barrels visible in lower frame
ALT_RED       = 4.5    # high — elevated red barrels come into camera FOV
ALT_MAX_EXPLORE = 7.0  # ceiling for altitude boost when stuck in AVOID-CRIT

# Row spacing — wider than camera sightline since detection runs during flight
ROW_SPACING_LOW  = 5.0   # at 1.8m alt camera sees ~5m ahead
ROW_SPACING_HIGH = 7.0   # at 4.5m alt camera sees further

CONTROL_HZ     = 15.0
ARRIVAL_RADIUS = 1.0    # horizontal arrival threshold (m)
ARRIVAL_ALT    = 0.5    # vertical arrival threshold (m)
MISSION_LIMIT  = 600.0  # s — 10 min hard cap

# Virtual target blending
W_AVOID     = 0.5   # depth-camera avoidance weight
W_MEM_AVOID = 0.3   # memory map avoidance weight

# Avoidance
SOFT_DIST = 2.5   # pre-avoidance: start gently reducing goal weight at this distance
SAFE_DIST = 1.5   # avoidance activates within this range
CRIT_DIST = 0.8   # emergency backup/crawl

# Map memory
MAP_RETENTION_M = 60.0  # full arena diagonal ~52m
MAP_INFLUENCE_M = 4.5
MAP_Z_MAX = 10.0

# Velocity setpoint limits
VEL_MAX      = 1.5   # m/s — open space cruise speed
VEL_MIN      = 0.3   # m/s — near obstacles / boundary recovery
YAW_RATE_MAX = 15.0  # deg per control tick
ALT_KP       = 2.0   # P-gain for altitude velocity controller
ALT_VEL_MAX  = 0.5   # m/s — max vertical correction speed

# Stuck detection
STUCK_TIMEOUT_S = 4.0
STUCK_DIST_M = 0.4
WP_APPROACH_TIMEOUT = 45.0
STUCK_ESCAPE_M = 2.5

# Detection gate — set True to enable barrel detection for scoring
DETECTION_ENABLED = True

# Detection confirmation
DETECT_CONFIRM = 4

# Carrot-point lookahead
LOOKAHEAD_DIST = 1.5   # m — shorter bursts so drone re-evaluates more often

MERGE_DIST = 3.0

# Map update throttle
MAP_THROTTLE = 4

# Path history — repel drone from recently visited positions
PATH_HIST_INTERVAL = 3    # record position every N ticks
PATH_HIST_LEN      = 80   # keep last 80 points (~16s of flight)
PATH_HIST_INFL     = 8.0  # repulsion influence radius (m)
W_PATH_HIST        = 1.5  # repulsion weight — HEAVILY penalize backtracking

# Origin repulsion — strong penalty for returning to spawn point
ORIGIN_INFL_M      = 6.0  # influence radius around spawn (m)
W_ORIGIN_REPEL     = 2.0  # weight — even stronger than path history

# LOOK_AROUND trigger — when narrow camera FOV keeps drone blocked, do 360° scan
AVOID_STREAK_TRIGGER = 30   # ticks of consecutive blocked frames (~2s at 15Hz)
LOOK_BURST_M         = 3.0  # commit this far in best direction after scan

# Visited grid
CELL_SIZE     = 2.0          # m — grid cell resolution
GRID_N_ORIGIN = ZONE1_N[0]   # -4.0  m — south edge relative to spawn NED
GRID_E_ORIGIN = ZONE1_E[0]   # -12.0 m — west  edge relative to spawn NED
GRID_N_CELLS  = 18
GRID_E_CELLS  = 20

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
    LOOK_AROUND = "LOOK_AROUND"
    DONE = "DONE"
    LAND = "LAND"
