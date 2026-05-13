import math
import os
import threading

import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

CAMERA_TOPIC = (
    "/world/roboverse/model/x500_vision_0"
    "/link/camera_link/sensor/IMX214/image"
)

# HSV ranges for yellow barrels
YELLOW_LOWER = np.array([18, 100, 100])
YELLOW_UPPER = np.array([38, 255, 255])

# HSV ranges for red barrels (red wraps around hue=0, so two ranges needed)
RED_LOWER1 = np.array([0,  130, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([165, 130, 100])
RED_UPPER2 = np.array([180, 255, 255])

MIN_BLOB_AREA = 400  # px² — minimum contour size to count as a detection


class BarrelDetector:

    def __init__(self, model_path: str = ""):
        self._lock         = threading.Lock()
        self._latest_frame = None
        self._yolo         = None

        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(model_path)
                print(f"[Detector] YOLO model loaded: {model_path}")
            except Exception as exc:
                print(f"[Detector] YOLO load failed ({exc}), using colour detection")

        self._node = Node()
        if self._node.subscribe(Image, CAMERA_TOPIC, self._on_image):
            print(f"[Detector] Camera subscribed")
        else:
            print(f"[Detector] WARNING: camera subscription failed")

    def _on_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        with self._lock:
            self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def detect(self):
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            return {"yellow": False, "red": False}
        return self._detect_yolo(frame) if self._yolo else self._detect_colour(frame)

    def _detect_colour(self, frame_bgr):
        hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        y_mask = cv2.morphologyEx(cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER), cv2.MORPH_OPEN, kernel)
        y_cnts, _ = cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow = any(cv2.contourArea(c) >= MIN_BLOB_AREA for c in y_cnts)

        r_mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
                                cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN, kernel)
        r_cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red = any(cv2.contourArea(c) >= MIN_BLOB_AREA for c in r_cnts)

        return {"yellow": yellow, "red": red}

    def _detect_yolo(self, frame_bgr):
        yellow, red = False, False
        for result in self._yolo(frame_bgr, verbose=False, conf=0.4):
            for box in result.boxes or []:
                cls_id = int(box.cls[0].cpu().item())
                name   = self._yolo.names.get(cls_id, "").lower()
                if "yellow" in name or cls_id == 0:
                    yellow = True
                elif "red" in name or cls_id == 1:
                    red = True
        return {"yellow": yellow, "red": red}


class DetectionTracker:

    def __init__(self, merge_distance: float = 3.0):
        self._merge  = merge_distance
        self._lock   = threading.Lock()
        self._yellow = []
        self._red    = []

    @staticmethod
    def _is_new(positions, north, east, dist):
        return all(math.hypot(n - north, e - east) >= dist for n, e in positions)

    def try_add_yellow(self, north, east):
        with self._lock:
            if self._is_new(self._yellow, north, east, self._merge):
                self._yellow.append([north, east])
                return True
        return False

    def try_add_red(self, north, east):
        with self._lock:
            if self._is_new(self._red, north, east, self._merge):
                self._red.append([north, east])
                return True
        return False

    @property
    def yellow_count(self):
        with self._lock:
            return len(self._yellow)

    @property
    def red_count(self):
        with self._lock:
            return len(self._red)

    def score(self):
        return self.yellow_count * 50 + self.red_count * 100

    def summary(self):
        return (f"Y={self.yellow_count}x50={self.yellow_count*50}  "
                f"R={self.red_count}x100={self.red_count*100}  "
                f"Total={self.score()}")
