import numpy as np
import matplotlib.pyplot as plt
from depth_receiver import DepthReceiver
import time
from matplotlib.patches import Circle

def depth_to_xy_map(
    depth_img, K,
    cam_height=1.0, obs_h_min=0.05, obs_h_max=1.5,
    z_min=0.2, z_max=15.0,
):
    """
    Convert float32 depth image to metric X-Y top-down obstacle map.
    
    Returns:
        points: Nx3 array (X, Y, Z) in camera optical frame [meters]
        xy_obstacles: Mx2 array of actual (X, Y) obstacle coordinates [meters]
    """
        
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # 1. Keep valid depths by distance range (remove invalid readings)
    z = depth_img.astype(np.float32)
    valid_depth = (z > z_min) & (z < z_max)
    
    # 2. Use camera intrinsics to back-project onto camera optical frame (X-right, Y-down, Z-forward) to determine actual distance
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u[valid_depth] - cx) * z[valid_depth] / fx
    y_cam = (v[valid_depth] - cy) * z[valid_depth] / fy
    z_fwd = z[valid_depth]  # forward distance in camera frame
    
    # 3. Filter by vertical height (remove ground & ceiling)
    # In optical frame: Y points DOWN. Ground is at y = cam_height.
    y_min_cam = cam_height - obs_h_max
    y_max_cam = cam_height - obs_h_min
    valid_h = (y_cam >= y_min_cam) & (y_cam <= y_max_cam)
    
    # Keep only valid obstacle points
    x_cam, y_cam, z_fwd = x_cam[valid_h], y_cam[valid_h], z_fwd[valid_h]
#    points = np.stack((x_cam, y_cam, z_fwd), axis=-1)
    
    # 5. Extract actual X-Y obstacle coordinates (lateral X, forward Z)
    # For top-down navigation, we care about (X, Z) in camera frame
    xy_obstacles = np.stack((x_cam, z_fwd), axis=-1)  # Mx2 array in meters
       
    return xy_obstacles


# ================= EXAMPLE USAGE =================
if __name__ == "__main__":
    # Your exact intrinsic matrix
    K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])
                  
    # Simulated Gazebo depth (replace with your actual depth image)
    receiver = DepthReceiver("/depth_camera")
    time.sleep(5)

    depth_img = receiver.get_frame()  # Should return HxW float32 array in meters

    if depth_img is None:
        print("No depth data received yet.")
        exit(0)
    
    # Run pipeline
    xy_obstacles = depth_to_xy_map(
        depth_img, K,
        cam_height=1.0, obs_h_min=0.1, obs_h_max=1.2
    )
    
    print(f"✅ Obstacle X-Y coords: {xy_obstacles.shape[0]} detections")
    print(f"✅ First 5 obstacle positions (X lateral, Z forward) [meters]:")
    print(xy_obstacles[:5])
    
    # ================= METRIC VISUALIZATION =================
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
       
    # 2. Scatter plot of actual X-Y obstacle coordinates
    ax2 = axes
    if xy_obstacles.shape[0] > 0:
        # Color by distance for visual clarity
        dists = np.linalg.norm(xy_obstacles, axis=1)
        sc = ax2.scatter(xy_obstacles[:, 0], xy_obstacles[:, 1], 
                        c=dists, s=15, cmap='viridis_r', edgecolors='white', linewidth=0.3)
        plt.colorbar(sc, ax=ax2, label="Distance from camera [m]")
    ax2.set_xlabel("Lateral X [m] (right +)")
    ax2.set_ylabel("Forward Z [m]")
    ax2.set_title("Detected Obstacle Coordinates\n(actual X-Y positions in meters)")
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # ================= OPTIONAL: Export coordinates for planning =================
    # xy_obstacles is ready to use for path planning, collision checking, etc.
    # Example: find closest obstacle
    if xy_obstacles.shape[0] > 0:
        dists = np.linalg.norm(xy_obstacles, axis=1)
        closest_idx = np.argmin(dists)
        print(f"\n🎯 Closest obstacle at X={xy_obstacles[closest_idx,0]:.2f}m, "
              f"Z={xy_obstacles[closest_idx,1]:.2f}m (distance: {dists[closest_idx]:.2f}m)")
