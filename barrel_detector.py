"""
Barrel detector for RoboVerse 2026 Qualifier.

Uses HSV color filtering as the primary detection method (reliable in sim).
Falls back to a trained YOLO model if one is supplied.
"""

import math
import os
import threading

import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# ---------------------------------------------------------------------------
# Gazebo camera topic
# ---------------------------------------------------------------------------
CAMERA_TOPIC = (
    "/world/roboverse/model/x500_vision_0"
    "/link/camera_link/sensor/IMX214/image"
)

# ---------------------------------------------------------------------------
# HSV colour ranges
# Yellow barrel (bright yellow / orange-yellow)
YELLOW_LOWER = np.array([18, 100, 100])
YELLOW_UPPER = np.array([38, 255, 255])

# Red barrel (hue wraps at 0/180 → two ranges)
RED_LOWER1 = np.array([0,  130, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([165, 130, 100])
RED_UPPER2 = np.array([180, 255, 255])

# Minimum contour area (pixels²) to count as a detection
MIN_BLOB_AREA = 400


class BarrelDetector:
    """
    Thread-safe barrel detector.

    Subscribes to the Gazebo camera topic in the background.
    Call detect() from any coroutine or thread to get the latest result.

    If a trained YOLO model path is provided and the file exists, YOLO
    inference is used instead of colour filtering.  The model is expected
    to have class names containing 'yellow' and 'red' (or class IDs 0 and 1).
    """

    def __init__(self, model_path: str = ""):
        self._lock = threading.Lock()
        self._latest_frame = None

        # Optional YOLO model (pass path to the qualifier model from Discord)
        self._yolo = None
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO  # noqa: PLC0415
                self._yolo = YOLO(model_path)
                print(f"[Detector] YOLO model loaded: {model_path}")
            except Exception as exc:
                print(f"[Detector] YOLO load failed ({exc}); using colour detection.")

        # Gazebo transport subscription
        self._node = Node()
        if self._node.subscribe(Image, CAMERA_TOPIC, self._on_image):
            print(f"[Detector] Subscribed to camera: {CAMERA_TOPIC}")
        else:
            print(f"[Detector] WARNING: Camera subscription failed: {CAMERA_TOPIC}")

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest_frame = frame_bgr

    # ------------------------------------------------------------------
    def detect(self) -> dict:
        """
        Return {'yellow': bool, 'red': bool}.
        Thread-safe; returns False for both if no frame received yet.
        """
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            return {"yellow": False, "red": False}

        if self._yolo is not None:
            return self._detect_yolo(frame)
        return self._detect_colour(frame)

    # ------------------------------------------------------------------
    def _detect_colour(self, frame_bgr: np.ndarray) -> dict:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        # --- Yellow ---
        y_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        y_mask = cv2.morphologyEx(y_mask, cv2.MORPH_OPEN, kernel)
        y_cnts, _ = cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow = any(cv2.contourArea(c) >= MIN_BLOB_AREA for c in y_cnts)

        # --- Red (two hue ranges) ---
        r_mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
            cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
        )
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN, kernel)
        r_cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red = any(cv2.contourArea(c) >= MIN_BLOB_AREA for c in r_cnts)

        return {"yellow": yellow, "red": red}

    # ------------------------------------------------------------------
    def _detect_yolo(self, frame_bgr: np.ndarray) -> dict:
        results = self._yolo(frame_bgr, verbose=False, conf=0.4)
        yellow = False
        red = False
        for result in results:
            for box in result.boxes or []:
                cls_id = int(box.cls[0].cpu().item())
                name = self._yolo.names.get(cls_id, "").lower()
                if "yellow" in name or cls_id == 0:
                    yellow = True
                elif "red" in name or cls_id == 1:
                    red = True
        return {"yellow": yellow, "red": red}


# ---------------------------------------------------------------------------

class DetectionTracker:
    """
    Tracks unique barrel detections, deduplicating by spatial proximity.

    A new detection is considered unique only if no previous detection of
    the same colour exists within `merge_distance` metres.
    """

    def __init__(self, merge_distance: float = 3.0):
        self._merge = merge_distance
        self._lock = threading.Lock()
        self._yellow: list[list[float]] = []  # [[north, east], ...]
        self._red: list[list[float]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _is_new(positions: list, north: float, east: float, dist: float) -> bool:
        return all(math.hypot(n - north, e - east) >= dist for n, e in positions)

    # ------------------------------------------------------------------
    def try_add_yellow(self, north: float, east: float) -> bool:
        with self._lock:
            if self._is_new(self._yellow, north, east, self._merge):
                self._yellow.append([north, east])
                return True
        return False

    def try_add_red(self, north: float, east: float) -> bool:
        with self._lock:
            if self._is_new(self._red, north, east, self._merge):
                self._red.append([north, east])
                return True
        return False

    # ------------------------------------------------------------------
    @property
    def yellow_count(self) -> int:
        with self._lock:
            return len(self._yellow)

    @property
    def red_count(self) -> int:
        with self._lock:
            return len(self._red)

    def score(self) -> int:
        return self.yellow_count * 50 + self.red_count * 100

    def summary(self) -> str:
        return (
            f"Y={self.yellow_count}×50={self.yellow_count*50}pts  "
            f"R={self.red_count}×100={self.red_count*100}pts  "
            f"Total={self.score()}pts"
        )
