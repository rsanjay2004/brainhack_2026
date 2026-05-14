import math
import numpy as np
from top_down import depth_to_xy_map


# Hard cap on accumulated obstacle points — prevents unbounded memory growth
MAX_POINTS = 5000

# Spatial deduplication resolution — one point kept per VOXEL_SIZE × VOXEL_SIZE cell
VOXEL_SIZE = 0.5  # meters


class GlobalMapper:
    """
    Incremental top-down occupancy grid mapper using NED (North-East-Down) pose.

    Coordinate Conventions:
    - Pose: {'north': float, 'east': float, 'yaw': float}
    - Yaw: radians, clockwise from North (standard NED heading)
    - Camera: Forward-facing, level mount assumed (X_cam=right, Z_cam=forward)
    - Grid: row=North, col=East (matches standard map orientation)
    """
    def __init__(self, K,
                 cam_height=1.0, obs_h_min=0.1, obs_h_max=1.5,
                 z_min=0.2, z_max=15.0,
                 yaw_in_degrees=False, yaw_clockwise=True, yaw_smoothing=0.7):
        self.K = K
        self.cam_height = cam_height
        self.obs_h_min = obs_h_min
        self.obs_h_max = obs_h_max
        self.z_min = z_min
        self.z_max = z_max

        # Yaw handling
        self.yaw_in_degrees = yaw_in_degrees
        self.yaw_clockwise = yaw_clockwise
        self.yaw_smoothing = yaw_smoothing
        self.last_yaw_rad = 0.0
        self.first_frame = True

        # Global point storage: (N, 2) array [north, east] in meters
        self.global_points = np.empty((0, 2), dtype=np.float32)

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _sanitize_depth(self, depth_img):
        d = np.array(depth_img, dtype=np.float32)
        bad = ~np.isfinite(d) | (d <= 0)
        d[bad] = self.z_max
        return d

    def _voxel_filter(self, points):
        """Keep one point per VOXEL_SIZE grid cell — reduces dense clusters."""
        if points.shape[0] == 0:
            return points
        keys = np.floor(points / VOXEL_SIZE).astype(np.int32)
        structured = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
        )
        _, idx = np.unique(structured, return_index=True)
        return points[idx]

    def _local_to_ned_global(self, local_xy, north, east, yaw_rad):
        """Transform local (X_cam=right, Z_cam=forward) to NED global (north, east)."""
        X_cam = local_xy[:, 0]
        Z_cam = local_xy[:, 1]
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        north_global = north + Z_cam * c - X_cam * s
        east_global  = east  + Z_cam * s + X_cam * c
        return np.column_stack([north_global, east_global])

    # -------------------------------------------------
    # Core API
    # -------------------------------------------------

    def update_frame(self, depth_img, pose):
        north   = pose['north']
        east    = pose['east']
        raw_yaw = pose['yaw']

        # 1. Yaw conversion & smoothing
        if self.yaw_in_degrees:
            raw_yaw = np.deg2rad(raw_yaw)
        if not self.yaw_clockwise:
            raw_yaw = -raw_yaw

        if self.first_frame:
            self.last_yaw_rad = raw_yaw
            self.first_frame = False
        else:
            raw_yaw = (self.yaw_smoothing * raw_yaw
                       + (1 - self.yaw_smoothing) * self.last_yaw_rad)
        self.last_yaw_rad = raw_yaw
        yaw = raw_yaw

        # 2. Sanitize depth before projection
        depth_clean = self._sanitize_depth(depth_img)

        # 3. Extract local obstacle coordinates
        xy_obstacles = depth_to_xy_map(
            depth_clean, self.K,
            cam_height=self.cam_height,
            obs_h_min=self.obs_h_min,
            obs_h_max=self.obs_h_max,
            z_min=self.z_min,
            z_max=self.z_max,
        )

        if xy_obstacles.shape[0] == 0:
            return False

        # 4. Transform to global NED
        global_pts = self._local_to_ned_global(xy_obstacles, north, east, yaw)

        # 5. Drop any non-finite points produced by bad pose or projection
        valid = np.isfinite(global_pts).all(axis=1)
        global_pts = global_pts[valid]
        if global_pts.shape[0] == 0:
            return False

        # 6. Accumulate
        self.global_points = np.vstack([self.global_points, global_pts])

        # 7. Spatial deduplication — one point per VOXEL_SIZE cell
        self.global_points = self._voxel_filter(self.global_points)

        # 8. Hard cap — keep most recent MAX_POINTS if exceeded
        if self.global_points.shape[0] > MAX_POINTS:
            self.global_points = self.global_points[-MAX_POINTS:]

        return True

    def prune(self, north, east, retention_radius=15.0):
        """Remove obstacle points further than retention_radius from current drone position."""
        if self.global_points.shape[0] == 0:
            return
        dists = np.hypot(self.global_points[:, 0] - north,
                         self.global_points[:, 1] - east)
        self.global_points = self.global_points[dists <= retention_radius]

    def get_repulsion_vector(self, north, east, influence_radius=4.5):
        """
        Return a unit repulsion vector (dn, de) pushing the drone away from
        remembered nearby obstacles. Returns (0.0, 0.0) if no obstacles are
        within influence_radius. Closer obstacles contribute more weight (1/d²).
        """
        if self.global_points.shape[0] == 0:
            return 0.0, 0.0

        # Drop any corrupted points before computing repulsion
        valid = np.isfinite(self.global_points).all(axis=1)
        pts = self.global_points[valid]
        if pts.shape[0] == 0:
            return 0.0, 0.0

        dists = np.hypot(pts[:, 0] - north, pts[:, 1] - east)
        mask = dists < influence_radius
        if not np.any(mask):
            return 0.0, 0.0

        nearby = pts[mask]
        d = dists[mask]
        diff_n = north - nearby[:, 0]
        diff_e = east  - nearby[:, 1]
        weights = 1.0 / (d ** 2 + 1e-3)
        rep_n = np.sum(weights * diff_n / (d + 1e-6))
        rep_e = np.sum(weights * diff_e / (d + 1e-6))

        if not (math.isfinite(rep_n) and math.isfinite(rep_e)):
            return 0.0, 0.0

        mag = math.hypot(rep_n, rep_e)
        if mag < 1e-6:
            return 0.0, 0.0
        return rep_n / mag, rep_e / mag

    def get_global_points(self):
        """Returns copy of accumulated (north, east) points in meters."""
        return self.global_points.copy()

    def save_points(self, filename="global_obstacles.npy"):
        np.save(filename, self.global_points)
        print(f"Saved {len(self.global_points)} points to {filename}")


# ================= Sample usage EXAMPLE =================
async def run():
    import asyncio
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from depth_receiver import DepthReceiver
    from drone_control import Drone
    from get_position_with_task import SharedState, position_monitor_task

    K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])
    receiver = DepthReceiver("/depth_camera")
    time.sleep(5)

    mapper = GlobalMapper(
        K, cam_height=1.0, obs_h_min=0.1, obs_h_max=1.5,
        yaw_in_degrees=True, yaw_smoothing=1.0, z_min=0.3, z_max=5.0
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    stop_event = asyncio.Event()
    drone = Drone()
    await drone.connect()
    await drone.arm_and_takeoff()

    state = SharedState()
    monitor_task = asyncio.create_task(
        position_monitor_task(drone, state, stop_event)
    )
    await asyncio.sleep(3)

    for i in range(3):
        pose = {
            'yaw':   state.latest_yaw,
            'north': state.latest_position.north_m,
            'east':  state.latest_position.east_m,
            'down':  state.latest_position.down_m,
        }
        depth_img = receiver.get_frame()
        if depth_img is None:
            print("No depth data received yet.")
            continue
        mapper.update_frame(depth_img, pose)
        print(f"N: {pose['north']} E:{pose['east']} Yaw:{pose['yaw']}")
        await drone.send_position_setpoint(
            north=pose['north'] + 3, east=pose['east'],
            down=pose['down'], yaw_deg=0
        )
        await asyncio.sleep(5)

    pts = mapper.get_global_points()
    ax.clear()
    if len(pts) > 0:
        dists = np.linalg.norm(pts, axis=1)
        ax.scatter(pts[:, 1], pts[:, 0], c=dists, s=4, cmap='viridis', edgecolors='none')

    pose = {
        'yaw':   state.latest_yaw,
        'north': state.latest_position.north_m,
        'east':  state.latest_position.east_m,
        'down':  state.latest_position.down_m,
    }
    ax.plot(pose['east'], pose['north'], 'r*', markersize=12, label='Drone')
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()

    await drone.land()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
