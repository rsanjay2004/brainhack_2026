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

# Yellow barrel — wider H range and lower S/V thresholds to handle
# Gazebo's lighting which can desaturate colours compared to real life
YELLOW_LOWER = np.array([15,  60,  60])
YELLOW_UPPER = np.array([45, 255, 255])

# Red barrel — hue wraps at 0/180 in OpenCV so two ranges are needed
RED_LOWER1 = np.array([0,   60,  60])
RED_UPPER1 = np.array([15, 255, 255])
RED_LOWER2 = np.array([155,  60,  60])
RED_UPPER2 = np.array([180, 255, 255])

# Minimum contour area to count as a detection (px²)
# Separate thresholds: red barrels are elevated and may appear smaller
MIN_AREA_YELLOW = 300
MIN_AREA_RED    = 200


class BarrelDetector:

    def __init__(self, model_path=""):
        self._lock         = threading.Lock()
        self._latest_frame = None
        self._yolo         = None

        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(model_path)
                print(f"[Detector] YOLO model: {model_path}")
            except Exception as exc:
                print(f"[Detector] YOLO failed ({exc}), using colour detection")

        self._node = Node()
        if self._node.subscribe(Image, CAMERA_TOPIC, self._on_image):
            print("[Detector] Camera subscribed")
        else:
            print("[Detector] WARNING: camera subscription failed — check topic name")

    def _on_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        with self._lock:
            self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def detect(self, phase="YELLOW"):
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            return {"yellow": False, "red": False}
        return self._detect_yolo(frame) if self._yolo else self._detect_colour(frame, phase)

    def _detect_colour(self, frame_bgr, phase="YELLOW"):
        kernel = np.ones((5, 5), np.uint8)
        h, w   = frame_bgr.shape[:2]

        # Phase-aware ROI selection:
        # YELLOW phase — fly at 1.8 m; ground barrels (~0.5 m tall) project
        #   onto the bottom ~35% of the 480 px frame at 2-5 m range.
        #   Check bottom 40% first, then full frame as fallback.
        # RED phase — fly at 4.5 m; elevated barrels (~2-3 m tall) project
        #   onto the lower-middle portion (rows 150-420).
        #   Check that band first, then full frame as fallback.
        if phase == "YELLOW":
            regions = [frame_bgr[int(h * 0.60):, :], frame_bgr]
        else:
            regions = [frame_bgr[int(h * 0.30):int(h * 0.88), :], frame_bgr]

        yellow_found = False
        red_found    = False

        for region in regions:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

            # Yellow
            y_mask = cv2.morphologyEx(
                cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER),
                cv2.MORPH_OPEN, kernel
            )
            cnts, _ = cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if any(cv2.contourArea(c) >= MIN_AREA_YELLOW for c in cnts):
                yellow_found = True

            # Red (two hue ranges merged)
            r_mask = cv2.bitwise_or(
                cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
                cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
            )
            r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if any(cv2.contourArea(c) >= MIN_AREA_RED for c in cnts):
                red_found = True

        return {"yellow": yellow_found, "red": red_found}

    def _detect_yolo(self, frame_bgr):
        yellow, red = False, False
        for result in self._yolo(frame_bgr, verbose=False, conf=0.35):
            for box in result.boxes or []:
                cls_id = int(box.cls[0].cpu().item())
                name   = self._yolo.names.get(cls_id, "").lower()
                if "yellow" in name or cls_id == 0:
                    yellow = True
                elif "red" in name or cls_id == 1:
                    red = True
        return {"yellow": yellow, "red": red}


class DetectionTracker:

    def __init__(self, merge_distance=3.0):
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
        return (f"Y={self.yellow_count}x50={self.yellow_count * 50}  "
                f"R={self.red_count}x100={self.red_count * 100}  "
                f"Total={self.score()}")
