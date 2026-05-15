import numpy as np


def depth_to_xy_map(
    depth_img, K,
    cam_height=1.0, obs_h_min=0.05, obs_h_max=1.5,
    z_min=0.2, z_max=15.0,
    roll_rad=0.0, pitch_rad=0.0,
    subsample=4,
):
    """
    Convert float32 depth image to top-down obstacle coordinates in body frame.

    Attitude-compensated: uses live drone altitude and roll/pitch to correctly
    classify obstacle heights regardless of current altitude or drone attitude.
    Subsamples for performance (default 4x reduces pixels ~16x).

    Coordinate conventions:
      Camera frame: X_c = right, Y_c = down, Z_c = forward
      Body frame:   X_b = North (forward), Y_b = East (right), Z_b = Down
      roll_rad > 0  = right wing down (rotation about body X/North)
      pitch_rad > 0 = nose up (rotation about body Y/East)

    Returns:
        xy_obstacles: Mx2 float32 array of (east_body, north_body) [meters].
                      Column 0 = lateral-right in body frame (GlobalMapper X_cam).
                      Column 1 = forward in body frame (GlobalMapper Z_cam).
                      GlobalMapper._local_to_ned_global applies yaw rotation to
                      convert these to global NED — no change needed there.
    """
    # --- Subsample for performance ---
    depth_img = depth_img[::subsample, ::subsample]
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Adjusted intrinsics for subsampled pixel coordinates.
    # Original: x_c = (u_orig - cx) / fx * z, u_orig = u_sub * subsample
    # Equivalent: x_c = (u_sub - cx/subsample) / (fx/subsample) * z
    fx = K[0, 0] / subsample
    fy = K[1, 1] / subsample
    cx = K[0, 2] / subsample
    cy = K[1, 2] / subsample

    z = depth_img.astype(np.float32)
    valid_depth = (z > z_min) & (z < z_max)

    x_c = (u[valid_depth] - cx) * z[valid_depth] / fx   # right in camera frame
    y_c = (v[valid_depth] - cy) * z[valid_depth] / fy   # down in camera frame
    z_c = z[valid_depth]                                  # forward in camera frame

    if x_c.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # --- Attitude-compensated world Down component ---
    # Camera → body: x_b = z_c, y_b = x_c, z_b = y_c (forward-facing level mount)
    # Body → world NED (roll φ, pitch θ, yaw handled separately by GlobalMapper):
    #   world_down = sin(θ)*z_c - sin(φ)*cos(θ)*x_c + cos(φ)*cos(θ)*y_c
    sp, cp = float(np.sin(pitch_rad)), float(np.cos(pitch_rad))
    sr, cr = float(np.sin(roll_rad)),  float(np.cos(roll_rad))

    world_down = sp * z_c - sr * cp * x_c + cr * cp * y_c

    # Height above ground in world frame
    obs_height = cam_height - world_down
    valid_h = (obs_height >= obs_h_min) & (obs_height <= obs_h_max)

    x_c = x_c[valid_h]
    y_c = y_c[valid_h]
    z_c = z_c[valid_h]

    if x_c.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # --- Attitude-compensated body-frame horizontal components ---
    # (pre-yaw, so GlobalMapper._local_to_ned_global can apply yaw correctly)
    #   east_body  = cos(φ)*x_c + sin(φ)*y_c
    #   north_body = cos(θ)*z_c + sin(φ)*sin(θ)*x_c - cos(φ)*sin(θ)*y_c
    east_body  = cr * x_c + sr * y_c
    north_body = cp * z_c + sr * sp * x_c - cr * sp * y_c

    # Column 0 = east_body (lateral), column 1 = north_body (forward)
    # Matches GlobalMapper._local_to_ned_global's (X_cam, Z_cam) convention
    return np.stack((east_body, north_body), axis=-1).astype(np.float32)


# ================= EXAMPLE USAGE =================
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from depth_receiver import DepthReceiver

    K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])

    receiver = DepthReceiver("/depth_camera")
    time.sleep(5)

    depth_img = receiver.get_frame()

    if depth_img is None:
        print("No depth data received yet.")
        exit(0)

    xy_obstacles = depth_to_xy_map(
        depth_img, K,
        cam_height=1.8, obs_h_min=0.1, obs_h_max=1.2,
        roll_rad=0.0, pitch_rad=0.0, subsample=4,
    )

    print(f"Obstacle body-frame coords: {xy_obstacles.shape[0]} detections")
    print(f"First 5 (east_body, north_body) [m]:")
    print(xy_obstacles[:5])

    fig, ax = plt.subplots(figsize=(8, 8))
    if xy_obstacles.shape[0] > 0:
        dists = np.linalg.norm(xy_obstacles, axis=1)
        sc = ax.scatter(xy_obstacles[:, 0], xy_obstacles[:, 1],
                        c=dists, s=15, cmap='viridis_r',
                        edgecolors='white', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="Distance from camera [m]")
    ax.set_xlabel("East body [m] (right +)")
    ax.set_ylabel("North body [m] (forward +)")
    ax.set_title("Obstacle coords in body frame (pre-yaw)")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
