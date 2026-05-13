import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from top_down import depth_to_xy_map
from depth_receiver import DepthReceiver
import time
from drone_control import Drone
import asyncio
from get_position_with_task import SharedState, position_monitor_task, run


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
        
    def _local_to_ned_global(self, local_xy, north, east, yaw_rad):
        """Transform local (X_cam=right, Z_cam=forward) to NED global (north, east)"""
        X_cam = local_xy[:, 0]
        Z_cam = local_xy[:, 1]
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        north_global = north + Z_cam * c - X_cam * s
        east_global  = east  + Z_cam * s + X_cam * c
        return np.column_stack([north_global, east_global])
    
    def update_frame(self, depth_img, pose):
        north = pose['north']
        east = pose['east']
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
            raw_yaw = self.yaw_smoothing * raw_yaw + (1 - self.yaw_smoothing) * self.last_yaw_rad
        self.last_yaw_rad = raw_yaw
        yaw = raw_yaw
        
        # 2. Extract local obstacle coordinates
        xy_obstacles = depth_to_xy_map(
            depth_img, self.K,
            cam_height=self.cam_height,
            obs_h_min=self.obs_h_min,
            obs_h_max=self.obs_h_max,
            z_min=self.z_min,
            z_max=self.z_max
        )
        
        if xy_obstacles.shape[0] == 0:
            return False
            
        # 3. Transform to global NED frame & accumulate
        global_pts = self._local_to_ned_global(xy_obstacles, north, east, yaw)
        self.global_points = np.vstack([self.global_points, global_pts])
        return True
    
    def get_global_points(self):
        """Returns copy of accumulated (north, east) points in meters"""
        return self.global_points.copy()
    
    def save_points(self, filename="global_obstacles.npy"):
        np.save(filename, self.global_points)
        print(f"✅ Saved {len(self.global_points)} points to {filename}")


# ================= Sample usage EXAMPLE =================
async def run():
    K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])
    receiver = DepthReceiver("/depth_camera")
    time.sleep(5)

    mapper = GlobalMapper(
        K, cam_height=1.0, obs_h_min=0.1, obs_h_max=1.5,
        yaw_in_degrees=True, yaw_smoothing=1.0,z_min=0.3,z_max=5.0
    )
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = plt.cm.colors.ListedColormap(['#808080', '#FFFFFF', '#000000'])
    
    stop_event = asyncio.Event()
    # 1. SETUP THE DRONE 
    drone = Drone()
    await drone.connect()
    await drone.arm_and_takeoff()

    # 2. Setup shared state & cancellation
    state = SharedState()    
    # Start background position monitor task
    monitor_task = asyncio.create_task(position_monitor_task(drone, state, stop_event))
    await asyncio.sleep(3)

    for i in range(3):
        pose = {}
        pose['yaw'] = state.latest_yaw
        pose['north'] = state.latest_position.north_m
        pose['east'] = state.latest_position.east_m
        pose['down'] = state.latest_position.down_m
        depth_img = receiver.get_frame()
        if depth_img is None:
            print("No depth data received yet.")
            continue
        mapper.update_frame(depth_img, pose)
        print(F"N: {pose['north']} E:{pose['east']} Yaw:{pose['yaw']}")
        await drone.send_position_setpoint(north=pose['north']+3, east=pose['east'], down=pose['down'], yaw_deg=0)
        await asyncio.sleep(5)

    await drone.send_position_setpoint(north=pose['north'], east=pose['east'], down=pose['down'], yaw_deg=90)
    await asyncio.sleep(5)
    pose = {}
    pose['yaw'] = state.latest_yaw
    pose['north'] = state.latest_position.north_m
    pose['east'] = state.latest_position.east_m
    pose['down'] = state.latest_position.down_m
    depth_img = receiver.get_frame()
    if depth_img is None:
        print("No depth data received yet.")
    else:
        mapper.update_frame(depth_img, pose)
        print(F"N: {pose['north']} E:{pose['east']} Yaw:{pose['yaw']}")

    await drone.send_position_setpoint(north=pose['north'], east=pose['east']+3, down=pose['down'], yaw_deg=90)
    await asyncio.sleep(5)
    pose = {}
    pose['yaw'] = state.latest_yaw
    pose['north'] = state.latest_position.north_m
    pose['east'] = state.latest_position.east_m
    pose['down'] = state.latest_position.down_m
    depth_img = receiver.get_frame()
    if depth_img is None:
        print("No depth data received yet.")
    else:
        mapper.update_frame(depth_img, pose)
        print(F"N: {pose['north']} E:{pose['east']} Yaw:{pose['yaw']}")


    pts = mapper.get_global_points()
    ax.clear()
    if len(pts) > 0:
        # Color by distance from origin for depth perception
        dists = np.linalg.norm(pts, axis=1)
        ax.scatter(pts[:, 1], pts[:, 0], c=dists, s=4, cmap='viridis', edgecolors='none')
    
    ax.plot(pose['east'], pose['north'], 'r*', markersize=12, label='Drone')
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    plt.pause(0.05)
    
    plt.show()

    await drone.land()

if __name__ == "__main__":
    asyncio.run(run())