# barrel_detector.py

import time
import math
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

CAMERA_TOPIC = (
    "/world/roboverse/model/x500_vision_0"
    "/link/camera_link/sensor/IMX214/image"
)

@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray

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

    def __init__(self, model_path="", topic=CAMERA_TOPIC):
        self._node = Node()
        self._lock = threading.Lock()
        self._result_lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = 0
        self._latest_frame_ts: float = 0.0

        self._last_color_result: Dict[str, Any] = {
            "yellow": False,
            "red": False,
            "frame_id": -1,
            "timestamp": 0.0,
            "source": "none",
            "detections": [],
        }

        self._last_yolo_result: Dict[str, Any] = {
            "yellow": False,
            "red": False,
            "frame_id": -1,
            "timestamp": 0.0,
            "source": "none",
            "detections": [],
        }

        self._last_returned_frame_id: int = -1

        self._model = None
        if model_path and YOLO is not None:
            self._model = YOLO(model_path)
        elif model_path and YOLO is None:
            print("[DETECTOR] YOLO requested but ultralytics is not available; using color-only detection")

        self._yolo_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=1)
        self._worker_stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self._model is not None:
            self._worker_thread = threading.Thread(
                target=self._yolo_worker,
                name="barrel-detector-yolo",
                daemon=True,
            )
            self._worker_thread.start()

        self._node.subscribe(Image, topic, self._on_image)

    def _on_image(self, msg: Image) -> None:
        if not self._lock.acquire(blocking=False):
            return  # drop frame — prevents gz callback queue backup
        packet = None
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._latest_frame = frame_bgr
            self._latest_frame_id += 1
            self._latest_frame_ts = time.time()
            packet = FramePacket(
                frame_id=self._latest_frame_id,
                timestamp=self._latest_frame_ts,
                frame_bgr=frame_bgr.copy(),
            )
        finally:
            self._lock.release()

        if packet is None:
            return

        if self._model is not None:
            try:
                while True:
                    self._yolo_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._yolo_queue.put_nowait(packet)
            except queue.Full:
                pass

    def detect(self, phase: str = "YELLOW") -> Dict[str, Any]:
        packet = self._get_latest_packet()
        if packet is None:
            return {
                "yellow": False,
                "red": False,
                "frame_id": -1,
                "timestamp": 0.0,
                "source": "none",
                "detections": [],
            }

        if packet.frame_id == self._last_returned_frame_id:
            return self._merge_results(
                color_result=self._last_color_result,
                yolo_result=self._last_yolo_result,
            )

        self._last_returned_frame_id = packet.frame_id
        self._last_color_result = self._detect_color(
            packet.frame_bgr,
            packet.frame_id,
            packet.timestamp,
            phase,
        )

        with self._result_lock:
            yolo_result = dict(self._last_yolo_result)

        return self._merge_results(
            color_result=self._last_color_result,
            yolo_result=yolo_result,
        )

    def _merge_results(self, color_result: Dict[str, Any], yolo_result: Dict[str, Any]) -> Dict[str, Any]:
        merged_detections = list(color_result.get("detections", [])) + list(yolo_result.get("detections", []))
        latest_frame_id = max(color_result.get("frame_id", -1), yolo_result.get("frame_id", -1))
        latest_timestamp = max(color_result.get("timestamp", 0.0), yolo_result.get("timestamp", 0.0))

        return {
            "yellow": bool(color_result.get("yellow", False) or yolo_result.get("yellow", False)),
            "red": bool(color_result.get("red", False) or yolo_result.get("red", False)),
            "frame_id": latest_frame_id,
            "timestamp": latest_timestamp,
            "source": (
                "hybrid"
                if color_result.get("source") != "none" and yolo_result.get("source") != "none"
                else color_result.get("source") if color_result.get("source") != "none"
                else yolo_result.get("source")
            ),
            "detections": merged_detections,
        }

    def _detect_color(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        timestamp: float,
        phase: str,
    ) -> Dict[str, Any]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        if phase.upper() == "YELLOW":
            roi = hsv[int(h * 0.35): int(h * 0.95), :]
        else:
            roi = hsv[int(h * 0.10): int(h * 0.80), :]

        # Phase-gate: only check the relevant colour — prevents cross-firing
        if phase.upper() == "YELLOW":
            yellow_mask = cv2.inRange(roi, YELLOW_LOWER, YELLOW_UPPER)
            yellow = self._mask_has_object(yellow_mask, MIN_AREA_YELLOW)
            red = False
        else:
            red_mask = cv2.bitwise_or(
                cv2.inRange(roi, RED_LOWER1, RED_UPPER1),
                cv2.inRange(roi, RED_LOWER2, RED_UPPER2),
            )
            red = self._mask_has_object(red_mask, MIN_AREA_RED)
            yellow = False

        detections: List[Dict[str, Any]] = []
        if yellow:
            detections.append({"class_name": "yellow", "confidence": None, "bbox": None, "mode": "color"})
        if red:
            detections.append({"class_name": "red", "confidence": None, "bbox": None, "mode": "color"})

        return {
            "yellow": yellow,
            "red": red,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source": "color" if (yellow or red) else "none",
            "detections": detections,
        }

    def _mask_has_object(self, mask: np.ndarray, min_area: int) -> bool:
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            aspect = h / max(w, 1)
            if 0.8 <= aspect <= 3.5:
                return True
        return False
    
    def _yolo_worker(self) -> None:
        while not self._worker_stop.is_set():
            try:
                packet = self._yolo_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                result = self._detect_yolo(packet.frame_bgr, packet.frame_id, packet.timestamp)
            except Exception as e:
                print(f"[DETECTOR] YOLO worker error: {e}")
                result = {
                    "yellow": False,
                    "red": False,
                    "frame_id": packet.frame_id,
                    "timestamp": packet.timestamp,
                    "source": "none",
                    "detections": [],
                }

            with self._result_lock:
                self._last_yolo_result = result

    def _detect_yolo(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> Dict[str, Any]:
        results = self._model(frame_bgr, verbose=False)

        yellow = False
        red = False
        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {})
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = str(names.get(cls_id, cls_id)).lower()

                if conf < 0.35:
                    continue

                xyxy = box.xyxy[0].tolist()
                det = {
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [float(v) for v in xyxy],
                    "mode": "yolo",
                }
                detections.append(det)

                if "yellow" in class_name:
                    yellow = True
                if "red" in class_name:
                    red = True

        return {
            "yellow": yellow,
            "red": red,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source": "yolo" if detections else "none",
            "detections": detections,
        }

    def close(self) -> None:
        self._worker_stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)

    def _get_latest_packet(self) -> Optional[FramePacket]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return FramePacket(
                frame_id=self._latest_frame_id,
                timestamp=self._latest_frame_ts,
                frame_bgr=self._latest_frame.copy(),
            )


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