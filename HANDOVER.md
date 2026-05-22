# Brainhack 2026 RoboVerse Qualifier — Handover

Autonomous drone exploration + barrel counting. PX4 SITL + Gazebo, MAVSDK Python,
`gz` transport for sensors. Main file: `qualifier_main_v39.py`.

---

## 1. Architecture (as it stands)

Per control tick, `drive_path_step()` runs a priority stack:

1. `drive_backtrack_step` — follow breadcrumbs out of a dead end
2. depth-frame guard
3. emergency front-block — `start_dead_end_escape` / `handle_front_blocked`
4. **corner-clip guard** (new) — edge away from a close side wall, no rotation
5. zone-loop escape
6. GATEWAY side-opening — *fallback only, gated off when a global path exists*
7. EXPRESS corridor — *fallback only, gated off when a global path exists*
8. warm replan
9. committed CENTERLINE — *fallback only, gated off when a global path exists*
10. no-path recovery scan
11. **global frontier-path follower** — the authoritative decision-maker

Mapping: log-odds occupancy grid (persistent), frontier-based A* planner.
Barrel detection: YOLO on RGB frames, `detection_to_global_ned` → world NED,
deduplicated into `barrel_tracks.csv`.

---

## 2. Painpoints (root causes, in priority order)

### P1 — Multi-layer decision thrash *(biggest single problem)*
EXPRESS, CENTERLINE, TOPO, GATEWAY, DEAD_END each independently pick a heading
every tick. They override one another: the drone turns toward layer A's heading,
layer B re-picks, the drone turns back — it oscillates in place ("hovers at
corners") instead of moving. Fixed by making the **global planner authoritative**:
reactive layers are skipped whenever a committed global path exists. They remain
only as fallback when there is no path.

### P2 — Drone wedges in tight spots
At pillar corners the drone gets nose-to-obstacle. Recovery handlers used to
re-scan every tick and never commit a rotation (spin-lock), or set a frontier
path and never follow it (set-and-abandon freeze). Fixed with decisive
`yaw_to_fast` rotations and a path-followability check before skipping backtrack.

### P3 — Costmap inflation trade-off
Grid resolution 0.40 m forces a near-binary choice:
- inflation ≥ 0.8 m (≥2 cells) → seals doorways < ~1.6 m → drone trapped in a room
- inflation 0.4 m (1 cell) → doorways open, but planned paths hug walls → corner clips

Currently at 0.40 m (doorways open). The corner-clip guard (P6) compensates
reactively. A finer grid (0.25–0.30 m) would give real middle-ground inflation but
costs 2–5× more cells per `inflated_occupied()` call — risky on the slow VM.

### P4 — VM simulation instability
2-core, no-GPU VM runs Gazebo + 1080p camera below real-time. Symptoms:
`User callback queue slow`, mavsdk gRPC `Connection reset by peer` at takeoff,
heartbeat timeouts. Mitigations: `pkill -f mavsdk_server` before each run,
restart PX4 if a run dies mid-air, lower camera res / headless if GUI not needed.

### P5 — Frontier planner starvation
`avoid_recent_path` hard-excluded frontiers reachable only via recent breadcrumbs
→ `no non-backtracking frontier` near already-crossed areas → drone fell back to
the oscillating reactive layers. Fixed: `allow_recent_path_fallback` default on
(penalised, not banned).

### P6 — Corner clipping (caused a crash)
The drone is not a point. Rotating while one side is ~0.6 m from a wall sweeps a
prop into it. Fixed with the **corner-clip guard**: if one side is below
`side_clip_margin_m` (0.65 m) and the other side is open, edge straight away with
no yaw change before any heading decision runs.

### P7 — Accumulated complexity
~18 fix iterations each added state/args. Every extra factor is a failure mode.
Two whole features were reverted after introducing freezes
(`stuck-escape`, `frontier-goal-commitment`). Dead code that never fired in any
run was removed (backtrack branch-abandon).

---

## 3. Lessons learnt

- **One authoritative decision-maker beats many reactive layers.** The frontier
  planner already targets unexplored space using the persistent map. Reactive
  heading-pickers layered on top fight it. Keep them strictly as fallback.
- **Commitment/hysteresis stops oscillation — but must yield when blocked.**
  `frontier-goal-commitment` froze the drone by holding a goal whose path was
  blocked. Any commitment needs an escape valve.
- **Do not add persistent state casually.** Each new variable is another
  interaction and another failure mode. Prefer reading existing state.
- **Test one fix at a time.** Compound changes mask which one helped or broke.
- **A "fix" that claims success without producing motion is a freeze.** Several
  handlers set a path / printed "routing there" then returned — the drone never
  moved. Always verify a recovery action actually translates the drone.
- **Sim performance is part of correctness.** A slow VM amplifies lockstep
  stalls and connection drops; some "bugs" were environment, not code.
- **The drone has a body.** Point-mass assumptions clip corners. Safety reflexes
  must account for prop sweep, especially during rotation.

---

## 4. Plans to proceed

**Immediate (verify current state)**
- Re-run; confirm corner-clip guard fires (`[CLIP_GUARD]`) and the drone no
  longer clips; confirm planner-authoritative removes corner hover.
- Confirm doorways still open (drone leaves the start room).

**Short term**
- If stable, physically delete the now fallback-only layers that proved to be
  noise: EXPRESS corridor, TOPO gateway grid-eval, GATEWAY side-opening. They are
  ~600 lines and a dozen args of dead weight once the planner is authoritative.
  Do this as one isolated, separately-tested pass.
- Tune `min-frontier-dist-m` / `frontier-overshoot-m` for fewer stop-scan cycles.

**Medium term**
- Finer grid (0.25–0.30 m) for proper inflation control — only if VM headroom
  allows; benchmark `inflated_occupied()` cost first.
- Replace the reactive corridor layers with a pure planner + path-follower +
  reactive obstacle filter. Simplest stable architecture.

---

## 5. Barrel detection — proposed improvements

Current: YOLO runs on saved RGB frames; colour-interest (yellow/red pixel count)
pre-filters which frames to save; `detection_to_global_ned` projects bbox + depth
to world NED; tracks deduplicated in `barrel_tracks.csv`.

Proposed fixes / improvements:

1. **Live inference, not just frame-saving.** Run YOLO inline whenever
   colour-interest triggers (already a cheap gate), so barrels are counted during
   the run, not in an offline pass. Cap inference rate (e.g. ≤2 Hz) to protect
   the slow VM.

2. **Depth-fused range, with the height fallback kept.** Use the depth crop at
   the bbox centre for range; fall back to apparent-height estimate only when the
   depth crop is invalid (already partly done). Reject detections whose depth and
   height ranges disagree by > ~1 m — likely a false positive.

3. **Stronger world-space deduplication.** Cluster detections by world NED with a
   merge radius (~0.6–0.8 m). A barrel seen from multiple poses must collapse to
   one track. Require N ≥ 2–3 confirming frames before a track counts, to drop
   one-frame false positives.

4. **Confidence + class-name gating.** Some ONNX exports scramble class names;
   the code already remaps red/yellow. Add a minimum confidence threshold and log
   raw class id + remapped name so mislabels are auditable.

5. **Do not detect mid-rotation.** Motion blur + pose skew during yaw spins
   corrupts the bbox→world projection. Gate inference to near-stationary,
   low-yaw-rate frames.

6. **Deliberate barrel sweep.** When colour-interest is high but the drone is
   moving fast, briefly slow and capture a settled frame before projecting —
   accuracy of the count matters more than a fraction of a second.

---

## 6. Key tunables (current defaults)

| Arg | Default | Purpose |
|---|---|---|
| `--map-safety-radius-m` | 0.40 | A* inflation; 0.40 keeps doorways open |
| `--side-clip-margin-m` | 0.65 | corner-clip guard trigger |
| `--collision-radius-m` | 0.34 | reactive depth safety corridor |
| `--frontier-overshoot-m` | 1.5 | push path past frontier into unknown |
| `--allow-recent-path-fallback` | on | planner returns a route instead of none |
| `--vertical-escape-enabled` | off | no up sensor — blind climb crashes |
| `--duration-s` | 600 | 10-minute flight cap |

Version-control snapshots of working builds are kept in `../version_control/`.
