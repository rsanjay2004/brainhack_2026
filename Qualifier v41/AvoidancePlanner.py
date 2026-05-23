import numpy as np
import math


class AvoidancePlanner:
    def __init__(self,
                 K,
                 width,
                 height,
                 max_speed=1.0,
                 safe_distance=2.5,
                 critical_distance=0.8,
                 num_bins=36,
                 smoothing_alpha=0.6):

        # --- Camera intrinsics ---
        self.fx = K[0, 0]
        self.cx = K[0, 2]

        self.width = width
        self.height = height

        # --- Planning params ---
        self.max_speed = max_speed
        self.safe_distance = safe_distance
        self.critical_distance = critical_distance
        self.num_bins = num_bins

        # --- Smoothing ---
        self.alpha = smoothing_alpha
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_north = None
        self.prev_east = None
        self.prev_down = None

    # -------------------------------------------------
    # Pixel → angle (intrinsics-based)
    # -------------------------------------------------
    def pixel_to_angle(self, u):
        return math.atan((u - self.cx) / self.fx)

    # -------------------------------------------------
    # Depth → polar histogram (true angular)
    # -------------------------------------------------
    def compute_histogram(self, depth_map):
        h, w = depth_map.shape

        histogram = np.zeros(self.num_bins)
        angles = np.zeros(self.num_bins)
        distances = np.zeros(self.num_bins)

        for i in range(self.num_bins):
            x_start = int(i * w / self.num_bins)
            x_end = int((i + 1) * w / self.num_bins)

            region = depth_map[:, x_start:x_end]

            # Robust distance (closest obstacles dominate)
            d = np.nanpercentile(region, 20)
            distances[i] = d

            # Cost function
            if d <= self.critical_distance:
                cost = 1.0
            else:
                cost = np.clip(1.0 / (d + 1e-3), 0, 1)

            histogram[i] = cost

            # True angle
            u_center = (x_start + x_end) / 2.0
            angles[i] = self.pixel_to_angle(u_center)

        return histogram, angles, distances

    # -------------------------------------------------
    # Compute clearance metrics
    # -------------------------------------------------
    def compute_clearance(self, depth_map):
        w = depth_map.shape[1]

        left = np.nanpercentile(depth_map[:, :w//3], 20)
        center = np.nanpercentile(depth_map[:, w//3:2*w//3], 20)
        right = np.nanpercentile(depth_map[:, 2*w//3:], 20)

        return left, center, right

    # -------------------------------------------------
    # Detect blockage condition
    # -------------------------------------------------
    def detect_blocked(self, left, center, right):
        return (
            center < self.critical_distance and
            left < self.safe_distance and
            right < self.safe_distance
        )

    # -------------------------------------------------
    # Detect corridor / open space
    # -------------------------------------------------
    def detect_environment(self, left, center, right):
        if center > self.safe_distance and left > self.safe_distance and right > self.safe_distance:
            return "OPEN"
        elif center > self.safe_distance:
            return "FORWARD_CLEAR"
        elif left > right:
            return "LEFT_OPEN"
        else:
            return "RIGHT_OPEN"

    # -------------------------------------------------
    # Select best direction (reactive only)
    # -------------------------------------------------
    def select_direction(self, histogram, angles):
        best_idx = np.argmin(histogram)
        return angles[best_idx], best_idx

    # -------------------------------------------------
    # Convert angle → velocity
    # -------------------------------------------------
    def angle_to_velocity(self, angle, forward_clearance):
        if forward_clearance > self.safe_distance:
            speed = self.max_speed
        elif forward_clearance > self.critical_distance:
            speed = self.max_speed * (
                (forward_clearance - self.critical_distance) /
                (self.safe_distance - self.critical_distance)
            )
        else:
            speed = 0.0

        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        return vx, vy, speed

    # -------------------------------------------------
    # Emergency avoidance (override only)
    # -------------------------------------------------
    def emergency_override(self, left, center, right):
        if center < self.critical_distance:
            if left > right:
                return 0.0, -self.max_speed   # go left
            else:
                return 0.0, self.max_speed    # go right
        return None

    # -------------------------------------------------
    # Velocity smoothing
    # -------------------------------------------------
    def smooth(self, vx, vy):
        vx_s = self.alpha * self.prev_vx + (1 - self.alpha) * vx
        vy_s = self.alpha * self.prev_vy + (1 - self.alpha) * vy

        self.prev_vx = vx_s
        self.prev_vy = vy_s

        return vx_s, vy_s

    # -------------------------------------------------
    # Position smoothing
    # -------------------------------------------------
    def smooth_position(self, north, east, down):
        if self.prev_north is None:
            # First call → no smoothing
            self.prev_north = north
            self.prev_east = east
            self.prev_down = down
            return north, east, down

        north_s = self.alpha * self.prev_north + (1 - self.alpha) * north
        east_s  = self.alpha * self.prev_east  + (1 - self.alpha) * east
        down_s  = self.alpha * self.prev_down  + (1 - self.alpha) * down

        self.prev_north = north_s
        self.prev_east  = east_s
        self.prev_down  = down_s

        return north_s, east_s, down_s

    # -------------------------------------------------
    # MAIN API
    # -------------------------------------------------
    def compute_position_ned(self, depth_map, pose, step_size=1.5):
        """
        Convert avoidance result into NED position setpoint
        step_size = how far to move per decision (meters)
        """

        # --- Step 1: Histogram ---
        histogram, angles, distances = self.compute_histogram(depth_map)

        # --- Step 2: Clearance ---
        left, center, right = self.compute_clearance(depth_map)

        # --- Step 3: Environment ---
        env_type = self.detect_environment(left, center, right)

        # --- Step 4: Block detection ---
        blocked = self.detect_blocked(left, center, right)

        # --- Step 5: Select direction ---
        angle, best_idx = self.select_direction(histogram, angles)

        # --- Step 6: Emergency override ---
        emergency = self.emergency_override(left, center, right)
        if emergency is not None:
            vx_body, vy_body = emergency
            angle = math.atan2(vy_body, vx_body)
        else:
            vx_body = math.cos(angle)
            vy_body = math.sin(angle)

        # -------------------------------------------------
        # 🔁 BODY → NED TRANSFORM
        # -------------------------------------------------
        yaw = pose["yaw"]

        # Rotation: body → NED
        north_dir = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
        east_dir  = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)

        # Normalize direction
        norm = math.sqrt(north_dir**2 + east_dir**2) + 1e-6
        north_dir /= norm
        east_dir  /= norm

        # -------------------------------------------------
        # 📍 GENERATE POSITION SETPOINT
        # -------------------------------------------------
        north = pose["north"] + step_size * north_dir
        east  = pose["east"]  + step_size * east_dir
        down  = pose["down"]  # keep altitude constant

        north, east, down = self.smooth_position(north, east, down)

        # -------------------------------------------------
        # OUTPUT INFO
        # -------------------------------------------------
        info = {
            "blocked": blocked,
            "environment": env_type,
            "clearance": {
                "left": float(left),
                "center": float(center),
                "right": float(right),
            },
            "selected_direction": {
                "angle_rad": float(angle),
                "bin_index": int(best_idx),
                "distance": float(distances[best_idx]),
            },
            "target_ned": {
                "north": float(north),
                "east": float(east),
                "down": float(down)
            }
        }

        return north, east, down, info

    def compute_velocity(self, depth_map):
        # --- Step 1: Histogram ---
        histogram, angles, distances = self.compute_histogram(depth_map)

        # --- Step 2: Clearance ---
        left, center, right = self.compute_clearance(depth_map)

        # --- Step 3: Environment understanding ---
        env_type = self.detect_environment(left, center, right)

        # --- Step 4: Block detection ---
        blocked = self.detect_blocked(left, center, right)

        # --- Step 5: Reactive direction ---
        angle, best_idx = self.select_direction(histogram, angles)

        vx, vy, speed = self.angle_to_velocity(angle, center)

        # --- Step 6: Emergency override ---
        emergency = self.emergency_override(left, center, right)
        if emergency is not None:
            vx, vy = emergency

        # --- Step 7: Smooth motion ---
        vx, vy = self.smooth(vx, vy)

        # -------------------------------------------------
        # OUTPUT FOR HIGH-LEVEL PLANNER
        # -------------------------------------------------
        info = {
            "blocked": blocked,
            "environment": env_type,
            "clearance": {
                "left": float(left),
                "center": float(center),
                "right": float(right),
            },
            "selected_direction": {
                "angle_rad": float(angle),
                "bin_index": int(best_idx),
                "distance": float(distances[best_idx]),
            },
            "histogram": histogram.tolist(),
            "forward_speed": float(speed)
        }

        return vx, vy, info