#!/usr/bin/env python3
"""
qualifier_main_v44.py

Frontier-based autonomous exploration for the drone barrel-counting qualifier.

v25 keeps the accurate v5/v23-style depth-image occupancy mapper, keeps the faster
"quick corridor centreline" exploration layer.  It limits startup/recovery scans,
uses early-exit heading selection, and starts moving after a small number of
settled depth snapshots instead of spending minutes scanning in place.

Why this version is different from the earlier one:
- v41 keeps the good v35-v41 counting/mapping, but adds a progress-commitment layer so the drone stops re-deciding every few centimetres. It only rescans when truly blocked, at a dead end, or after a meaningful committed move.
- It treats doorway/corridor openings as graph edges, prefers unvisited gateways that lead to unknown space, and only backtracks after confirmed dead ends.
- It keeps the working YOLO red/yellow barrel counter and fast corridor centreline control.
- v32 keeps v29/v30 express corridor and replaces unsafe rectangle regularisation with corner-safe line regularisation: when the drone sees a long straight corridor with two side boundaries, it aligns to the corridor centreline and commits forward at higher speed instead of repeatedly replanning/scanning.
- It does NOT keep turning right every time the side depth looks open.
- It builds a lightweight occupancy grid from the depth camera.
- It selects frontier cells: known-free cells next to unknown space.
- It plans through known-free cells with A* and drives toward the selected frontier.
- It saves RGB pictures sparsely, with priority for frames that look red/yellow.
- It avoids slow full 360 photo sweeps unless explicitly requested.
- It uses settled, discrete yaw snapshots for mapping scans so false walls are less likely.

Put this file in the same folder as the organiser files:
    drone_control.py, depth_receiver.py, get_position_with_task.py

Typical dataset-collection run:
    python3 qualifier_main_v38.py --mode collect --duration-s 900 --takeoff-altitude-m 3.0 --photo-interval-s 5 --sparse-photo-every-m 3

More aggressive mapping:
    python3 qualifier_main_v38.py --mode collect --duration-s 1200 --cruise-speed-m-s 0.8 --replan-interval-s 1.5

After YOLO training:
    python3 qualifier_main_v38.py --mode count --model-zip my_model.zip --duration-s 300

Or, if you already unzipped the model:
    python3 qualifier_main_v38.py --mode count --model train/weights/best.pt --duration-s 300
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import heapq
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

from depth_receiver import DepthReceiver
from drone_control import Drone
from get_position_with_task import SharedState, position_monitor_task

try:
    from AvoidancePlanner import AvoidancePlanner
except Exception as _avoid_import_exc:
    AvoidancePlanner = None  # type: ignore[assignment]
    print(f"[WARN] Could not import AvoidancePlanner.py: {_avoid_import_exc}")


DEFAULT_RGB_TOPIC = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_DEPTH_TOPIC = "/depth_camera"

K_DEFAULT = np.array(
    [
        [433.0, 0.0, 320.0],
        [0.0, 433.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# -----------------------------------------------------------------------------
# Basic math / pose helpers
# -----------------------------------------------------------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def wrap_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def yaw_error_deg(target: float, current: float) -> float:
    return wrap_deg(target - current)


def yaw_from_vector_deg(dn: float, de: float) -> float:
    # PX4 NED yaw convention: 0=N, +90=E.
    return wrap_deg(math.degrees(math.atan2(de, dn)))


def body_to_ned(forward: float, right: float, yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    north = forward * math.cos(yaw) - right * math.sin(yaw)
    east = forward * math.sin(yaw) + right * math.cos(yaw)
    return north, east


def ned_to_body(north: float, east: float, yaw_deg: float) -> Tuple[float, float]:
    """Convert NED velocity to body-frame [forward, right]."""
    yaw = math.radians(yaw_deg)
    forward = north * math.cos(yaw) + east * math.sin(yaw)
    right = -north * math.sin(yaw) + east * math.cos(yaw)
    return forward, right


def safe_percentile(region: np.ndarray, percentile: float, fallback: float) -> float:
    valid = region[np.isfinite(region) & (region > 0.15) & (region < 40.0)]
    if valid.size < 20:
        return float(fallback)
    return float(np.percentile(valid, percentile))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# -----------------------------------------------------------------------------
# RGB subscriber
# -----------------------------------------------------------------------------
class RGBReceiver:
    def __init__(self, topic: str):
        self.topic = topic
        self.node = Node()
        self.frame_bgr: Optional[np.ndarray] = None
        self.last_stamp_ms = 0
        ok = self.node.subscribe(Image, topic, self._callback)
        if ok:
            print(f"[RGB] Subscribed to {topic}")
        else:
            print(f"[RGB] ERROR: failed to subscribe to {topic}")

    def _callback(self, msg: Any) -> None:
        try:
            h = int(msg.height)
            w = int(msg.width)
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            pixels = h * w
            if pixels <= 0 or raw.size < pixels:
                return
            channels = raw.size // pixels
            channels = 3 if channels not in (1, 3, 4) else channels
            arr = raw[: pixels * channels].reshape((h, w, channels))
            if channels == 1:
                bgr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif channels == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.frame_bgr = bgr.copy()
            self.last_stamp_ms = now_ms()
        except Exception as exc:
            print(f"[RGB] callback error: {type(exc).__name__}: {exc}")

    def get_frame(self) -> Optional[np.ndarray]:
        return None if self.frame_bgr is None else self.frame_bgr.copy()


# -----------------------------------------------------------------------------
# Occupancy grid mapper and frontier planner
# -----------------------------------------------------------------------------
Cell = Tuple[int, int]


class OccupancyGridMapper:
    """
    Lightweight dynamic occupancy grid in NED coordinates.

    Cell axes:
        i = north index
        j = east index
    Stored log values:
        <= -1 known free
        >= occ_threshold occupied
        missing key unknown
    """

    def __init__(
        self,
        K: np.ndarray,
        resolution_m: float = 0.40,
        ray_max_m: float = 7.0,
        ray_min_m: float = 0.30,
        obstacle_margin_m: float = 0.35,
        safety_radius_m: float = 0.65,
        max_abs_log: int = 8,
        sample_cols: int = 80,
        occ_threshold: int = 4,
        occ_update: int = 2,
    ):
        self.K = K
        self.res = float(resolution_m)
        self.ray_max_m = float(ray_max_m)
        self.ray_min_m = float(ray_min_m)
        self.obstacle_margin_m = float(obstacle_margin_m)
        self.safety_radius_m = float(safety_radius_m)
        self.max_abs_log = int(max_abs_log)
        self.sample_cols = int(sample_cols)
        self.occ_threshold = max(2, int(occ_threshold))
        self.occ_update = max(1, int(occ_update))
        self.logodds: Dict[Cell, int] = {}
        self.visit_count: Dict[Cell, int] = {}
        self.last_update_ms = 0

    def world_to_cell(self, north: float, east: float) -> Cell:
        return (int(round(north / self.res)), int(round(east / self.res)))

    def cell_to_world(self, cell: Cell) -> Tuple[float, float]:
        return (cell[0] * self.res, cell[1] * self.res)

    def mark_free(self, cell: Cell, amount: int = 1) -> None:
        old = self.logodds.get(cell, 0)
        self.logodds[cell] = max(-self.max_abs_log, old - amount)

    def mark_occ(self, cell: Cell, amount: Optional[int] = None) -> None:
        old = self.logodds.get(cell, 0)
        if amount is None:
            amount = self.occ_update
        self.logodds[cell] = min(self.max_abs_log, old + int(amount))

    def is_known_free(self, cell: Cell) -> bool:
        return self.logodds.get(cell, 0) <= -1

    def is_occupied(self, cell: Cell) -> bool:
        return self.logodds.get(cell, 0) >= self.occ_threshold

    def neighbors8(self, cell: Cell) -> Iterable[Cell]:
        i, j = cell
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                yield (i + di, j + dj)

    def neighbors4(self, cell: Cell) -> Iterable[Cell]:
        i, j = cell
        yield (i + 1, j)
        yield (i - 1, j)
        yield (i, j + 1)
        yield (i, j - 1)

    def mark_visited_pose(self, pose: Dict[str, float]) -> None:
        cell = self.world_to_cell(pose["north"], pose["east"])
        self.visit_count[cell] = self.visit_count.get(cell, 0) + 1
        self.mark_free(cell, amount=2)
        self.clear_pose_footprint(pose, reason="visited_pose")

    def clear_pose_footprint(self, pose: Optional[Dict[str, float]], radius_m: Optional[float] = None, reason: str = "self") -> int:
        """Clear occupied evidence inside the drone's own footprint.

        Depth-image projection artefacts can occasionally place a black occupied
        cell directly under the vehicle. Once inflated for planning, that makes
        the planner think the drone is inside an obstacle and can cause the route
        follower to reject a clear corridor, turn around, or backtrack.

        This clears only a small bubble around the current vehicle centre, not the
        full safety radius. Real collision protection still comes from the live
        depth safety filter; the occupancy grid should never contain the drone
        itself as an obstacle.
        """
        if pose is None or not bool(getattr(self, "self_clear_enabled", True)):
            return 0
        if radius_m is None:
            radius_m = float(getattr(self, "self_clear_radius_m", 0.72))
        radius_m = max(0.0, float(radius_m))
        if radius_m <= 0.01:
            return 0
        free_log = -abs(int(getattr(self, "self_clear_free_log", 4)))
        radius_cells = max(1, int(math.ceil(radius_m / self.res)))
        centre = self.world_to_cell(float(pose["north"]), float(pose["east"]))
        changed = 0
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if (di * self.res) ** 2 + (dj * self.res) ** 2 > radius_m * radius_m:
                    continue
                cell = (centre[0] + di, centre[1] + dj)
                old = self.logodds.get(cell, 0)
                if old > free_log:
                    self.logodds[cell] = free_log
                    changed += 1
                self.visit_count[cell] = max(1, self.visit_count.get(cell, 0))
        if changed:
            self._regularized_cache_key = None
            self._regularized_cache = None
        return changed

    def update_from_depth(self, depth: Optional[np.ndarray], pose: Optional[Dict[str, float]]) -> bool:
        if depth is None or pose is None:
            return False

        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return False

        self.mark_visited_pose(pose)

        # Middle vertical band: avoids most floor/ceiling and keeps wall/object returns.
        y1 = int(0.30 * h)
        y2 = int(0.70 * h)
        band = depth[y1:y2, :]

        fx = float(self.K[0, 0])
        cx = float(self.K[0, 2])
        yaw = math.radians(float(pose["yaw_deg"]))
        c = math.cos(yaw)
        s = math.sin(yaw)
        north0 = float(pose["north"])
        east0 = float(pose["east"])

        cols = np.linspace(0, w - 1, self.sample_cols).astype(int)
        for u in cols:
            # Use a narrow column window to reduce speckle.
            x1 = max(0, u - 2)
            x2 = min(w, u + 3)
            d = safe_percentile(band[:, x1:x2], 15, self.ray_max_m)
            d = clamp(d, self.ray_min_m, self.ray_max_m)

            # Horizontal camera ray. x_right = lateral, z_fwd = forward.
            x_per_z = (float(u) - cx) / fx
            free_until = max(self.ray_min_m, min(self.ray_max_m, d - self.obstacle_margin_m))

            # Mark free along the ray.
            step = max(0.20, 0.65 * self.res)
            r = self.ray_min_m
            while r <= free_until:
                x_right = x_per_z * r
                z_fwd = r
                north = north0 + z_fwd * c - x_right * s
                east = east0 + z_fwd * s + x_right * c
                self.mark_free(self.world_to_cell(north, east), amount=1)
                r += step

            # Mark the hit as occupied if it is not just max-range/open space.
            if d < self.ray_max_m - 0.25:
                x_right = x_per_z * d
                z_fwd = d
                north = north0 + z_fwd * c - x_right * s
                east = east0 + z_fwd * s + x_right * c
                self.mark_occ(self.world_to_cell(north, east))

        # Last pass: never leave occupied evidence inside the drone's own
        # footprint after mapping this frame. This specifically prevents the
        # black-dot-on-red-marker failure mode.
        self.clear_pose_footprint(pose, reason="post_depth_update")

        if bool(getattr(self, "map_prune_diagonal_ghosts", True)):
            self._filtered_raw_occupied_cells()
        self.last_update_ms = now_ms()
        self._regularized_cache_key = None
        self._regularized_cache = None
        return True


    def _regularization_enabled(self) -> bool:
        return bool(getattr(self, "map_regularize", False))

    def _raw_occupied_cells(self, threshold: Optional[int] = None) -> set[Cell]:
        th = self.occ_threshold if threshold is None else int(threshold)
        return {c for c, v in self.logodds.items() if v >= th}

    def _filtered_raw_occupied_cells(self, threshold: Optional[int] = None) -> set[Cell]:
        """Occupied cells with diagonal ghost bridges removed.

        Depth projection errors at wall corners often appear as a thin diagonal
        chain of black cells crossing an otherwise open doorway/corridor.  The
        course geometry is mostly Manhattan/rectilinear, so cells that are
        supported mainly by diagonal neighbours but not by N/E/S/W wall runs are
        treated as weak artefacts for regularised planning.
        """
        occ = self._raw_occupied_cells(threshold=threshold)
        if not bool(getattr(self, "map_prune_diagonal_ghosts", True)) or len(occ) < 3:
            return occ

        min_axis_run = int(getattr(self, "map_diagonal_prune_min_axis_run_cells", 3))
        max_component = int(getattr(self, "map_diagonal_prune_max_component_cells", 14))
        apply_decay = bool(getattr(self, "map_diagonal_prune_apply_decay", True))
        decay = int(getattr(self, "map_diagonal_prune_decay", 2))

        def axis_run(cell: Cell) -> int:
            i, j = cell
            h = 1
            k = 1
            while (i, j + k) in occ and k <= 8:
                h += 1
                k += 1
            k = 1
            while (i, j - k) in occ and k <= 8:
                h += 1
                k += 1
            v = 1
            k = 1
            while (i + k, j) in occ and k <= 8:
                v += 1
                k += 1
            k = 1
            while (i - k, j) in occ and k <= 8:
                v += 1
                k += 1
            return max(h, v)

        def diag_count(cell: Cell) -> int:
            i, j = cell
            return sum(((i + di, j + dj) in occ) for di in (-1, 1) for dj in (-1, 1))

        def axial_count(cell: Cell) -> int:
            return sum((n in occ) for n in self.neighbors4(cell))

        weak: set[Cell] = set()
        for cell in occ:
            if diag_count(cell) > 0 and axis_run(cell) < min_axis_run and axial_count(cell) <= 1:
                weak.add(cell)
        if not weak:
            return occ

        visited: set[Cell] = set()
        remove: set[Cell] = set()
        for start in list(weak):
            if start in visited:
                continue
            stack = [start]
            comp: set[Cell] = set()
            visited.add(start)
            while stack:
                c = stack.pop()
                comp.add(c)
                for nb in self.neighbors8(c):
                    if nb in weak and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            if len(comp) <= max_component:
                remove.update(comp)

        if remove and apply_decay:
            for c in remove:
                old = self.logodds.get(c, 0)
                if old >= 0:
                    self.logodds[c] = max(-1, old - decay)
        return occ - remove

    def _make_occ_image(self, occ_cells: set[Cell]) -> Tuple[Optional[np.ndarray], int, int, int, int, int]:
        """Return a binary occupied image plus cell-index bounds.

        The image row axis is north index i, and column axis is east index j.
        This internal image is used only for line/rectangle fitting, not for the
        saved debug map orientation.
        """
        if not occ_cells:
            return None, 0, 0, 0, 0, 0
        is_ = [c[0] for c in occ_cells]
        js = [c[1] for c in occ_cells]
        pad = max(4, int(getattr(self, "map_regularize_pad_cells", 6)))
        min_i, max_i = min(is_) - pad, max(is_) + pad
        min_j, max_j = min(js) - pad, max(js) + pad
        H = max_i - min_i + 1
        W = max_j - min_j + 1
        if H <= 0 or W <= 0 or H * W > int(getattr(self, "map_regularize_max_image_cells", 400000)):
            return None, 0, 0, 0, 0, 0
        img = np.zeros((H, W), dtype=np.uint8)
        for i, j in occ_cells:
            r = i - min_i
            c = j - min_j
            if 0 <= r < H and 0 <= c < W:
                img[r, c] = 255
        return img, min_i, max_i, min_j, max_j, pad

    def regularized_occupied_cells(self) -> set[Cell]:
        """Return occupied cells after fitting walls/obstacles to simple primitives.

        This is intentionally conservative:
        - raw one-cell speckles are removed by opening / component filtering;
        - long occupied runs are snapped into straight line segments;
        - compact obstacle clusters can be represented as rectangles;
        - if regularisation finds too little evidence, it falls back to raw cells.

        The goal is not perfect CAD reconstruction.  The goal is to make A* and
        collision inflation see stable boundaries instead of jagged depth speckle.
        """
        if not self._regularization_enabled():
            return self._raw_occupied_cells()

        raw_occ = self._filtered_raw_occupied_cells()
        if not raw_occ:
            return set()

        # Cache prevents repeated Hough/CC processing during A* expansions.
        cache_key = (len(self.logodds), self.last_update_ms)
        if getattr(self, "_regularized_cache_key", None) == cache_key:
            cached = getattr(self, "_regularized_cache", None)
            if cached is not None:
                return set(cached)

        img, min_i, _max_i, min_j, _max_j, _pad = self._make_occ_image(raw_occ)
        if img is None:
            return raw_occ

        close_cells = max(1, int(round(float(getattr(self, "map_regularize_close_radius_m", 0.55)) / self.res)))
        open_cells = max(0, int(round(float(getattr(self, "map_regularize_open_radius_m", 0.10)) / self.res)))
        if close_cells > 0:
            k = 2 * close_cells + 1
            kernel = np.ones((k, k), np.uint8)
            img_clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            img_clean = img.copy()
        if open_cells > 0:
            k = 2 * open_cells + 1
            kernel = np.ones((k, k), np.uint8)
            img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_OPEN, kernel)

        out_img = np.zeros_like(img_clean)
        primitives: List[Tuple[str, float, float, float, float, float]] = []
        min_component = int(getattr(self, "map_regularize_min_component_cells", 4))
        min_line_len_cells = max(2, int(round(float(getattr(self, "map_regularize_min_line_m", 1.2)) / self.res)))
        max_line_gap_cells = max(1, int(round(float(getattr(self, "map_regularize_line_gap_m", 0.85)) / self.res)))
        line_thick_cells = max(1, int(round(float(getattr(self, "map_regularize_wall_thickness_m", 0.28)) / self.res)))
        hough_threshold = int(getattr(self, "map_regularize_hough_threshold", 5))
        snap_to_manhattan = bool(getattr(self, "map_regularize_manhattan", True))

        # 1) Straight wall/boundary extraction.
        lines = cv2.HoughLinesP(
            img_clean,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(2, hough_threshold),
            minLineLength=min_line_len_cells,
            maxLineGap=max_line_gap_cells,
        )
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in line]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < min_line_len_cells:
                    continue
                if snap_to_manhattan:
                    # The course is largely orthogonal.  Snapping line segments to
                    # N/E grid axes makes walls crisp and stops noisy zig-zagging.
                    if abs(dx) >= abs(dy):
                        y = int(round((y1 + y2) * 0.5))
                        x_lo, x_hi = sorted((x1, x2))
                        cv2.line(out_img, (x_lo, y), (x_hi, y), 255, line_thick_cells)
                        i0, j0 = min_i + y, min_j + x_lo
                        i1, j1 = min_i + y, min_j + x_hi
                        primitives.append(("line", i0 * self.res, j0 * self.res, i1 * self.res, j1 * self.res, length * self.res))
                    else:
                        x = int(round((x1 + x2) * 0.5))
                        y_lo, y_hi = sorted((y1, y2))
                        cv2.line(out_img, (x, y_lo), (x, y_hi), 255, line_thick_cells)
                        i0, j0 = min_i + y_lo, min_j + x
                        i1, j1 = min_i + y_hi, min_j + x
                        primitives.append(("line", i0 * self.res, j0 * self.res, i1 * self.res, j1 * self.res, length * self.res))
                else:
                    cv2.line(out_img, (x1, y1), (x2, y2), 255, line_thick_cells)
                    i0, j0 = min_i + y1, min_j + x1
                    i1, j1 = min_i + y2, min_j + x2
                    primitives.append(("line", i0 * self.res, j0 * self.res, i1 * self.res, j1 * self.res, length * self.res))

        # 2) Compact obstacle extraction.
        # v30 rectangle-fitted every compact component.  That made clean-looking
        # maps, but it could mistake an L-shaped wall corner for a solid box and
        # block the path around the corner.  v31 is conservative: rectangles are
        # disabled by default, and when enabled they require a compact, island-like
        # component with enough fill ratio.  Otherwise we preserve the cleaned raw
        # component instead of expanding it into a bounding rectangle.
        num_labels, labels, stats, _cent = cv2.connectedComponentsWithStats(img_clean, connectivity=8)
        enable_rectangles = bool(getattr(self, "map_regularize_rectangles", False))
        min_rect_side = max(1, int(round(float(getattr(self, "map_regularize_min_rect_side_m", 0.45)) / self.res)))
        max_rect_side = max(min_rect_side, int(round(float(getattr(self, "map_regularize_max_rect_side_m", 4.5)) / self.res)))
        rect_thick = max(1, int(round(float(getattr(self, "map_regularize_rect_thickness_m", 0.30)) / self.res)))
        min_rect_fill = float(getattr(self, "map_regularize_rect_min_fill_ratio", 0.22))
        max_rect_aspect = float(getattr(self, "map_regularize_rect_max_aspect", 4.0))
        for label_id in range(1, num_labels):
            x, y, w, h, area = [int(v) for v in stats[label_id]]
            if area < min_component:
                continue
            bbox_area = max(1, w * h)
            fill_ratio = float(area) / float(bbox_area)
            aspect = max(w / max(1, h), h / max(1, w))
            is_compact_rect_candidate = (
                enable_rectangles
                and min(w, h) >= min_rect_side
                and max(w, h) <= max_rect_side
                and fill_ratio >= min_rect_fill
                and aspect <= max_rect_aspect
            )
            if is_compact_rect_candidate:
                cv2.rectangle(out_img, (x, y), (x + w - 1, y + h - 1), 255, rect_thick)
                primitives.append(("rect", (min_i + y) * self.res, (min_j + x) * self.res, (min_i + y + h - 1) * self.res, (min_j + x + w - 1) * self.res, float(area)))
            else:
                # Preserve cleaned non-speckle cells from components that did not
                # clearly fit a safe rectangle/line. This is safer around corners
                # and door openings than drawing a solid bounding box.
                out_img[labels == label_id] = np.maximum(out_img[labels == label_id], 255)

        out_cells: set[Cell] = set()
        ys, xs = np.where(out_img > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            out_cells.add((min_i + int(y), min_j + int(x)))

        # Optionally preserve very strong raw occupied evidence so an actual box
        # observed from only one face is not thrown away by the regularizer.
        if bool(getattr(self, "map_regularize_keep_strong_raw", True)):
            strong_threshold = int(getattr(self, "map_regularize_strong_raw_threshold", max(self.occ_threshold + 2, 6)))
            out_cells.update(self._filtered_raw_occupied_cells(threshold=strong_threshold))

        min_regularized = int(getattr(self, "map_regularize_min_output_cells", 8))
        if len(out_cells) < min_regularized:
            out_cells = raw_occ

        self._regularized_cache_key = cache_key
        self._regularized_cache = set(out_cells)
        self._regularized_primitives = primitives
        return out_cells

    def inflated_occupied(self) -> set[Cell]:
        radius_cells = max(1, int(math.ceil(self.safety_radius_m / self.res)))
        blocked: set[Cell] = set()
        if bool(getattr(self, "map_regularize_use_for_planning", False)):
            occ_cells = list(self.regularized_occupied_cells())
        else:
            occ_cells = [c for c, v in self.logodds.items() if v >= self.occ_threshold]
        for ci, cj in occ_cells:
            for di in range(-radius_cells, radius_cells + 1):
                for dj in range(-radius_cells, radius_cells + 1):
                    if di * di + dj * dj <= radius_cells * radius_cells:
                        blocked.add((ci + di, cj + dj))
        return blocked

    def known_free_cells(self) -> List[Cell]:
        return [c for c, v in self.logodds.items() if v <= -1]

    def frontiers(self, current: Cell, min_dist_m: float = 1.8, max_candidates: int = 250) -> List[Cell]:
        blocked = self.inflated_occupied()
        min_dist_cells = max(1, int(math.ceil(min_dist_m / self.res)))
        free = self.known_free_cells()
        candidates: List[Tuple[float, Cell]] = []

        for cell in free:
            if cell in blocked:
                continue
            di = cell[0] - current[0]
            dj = cell[1] - current[1]
            if di * di + dj * dj < min_dist_cells * min_dist_cells:
                continue
            unknown_neighbor = any(n not in self.logodds for n in self.neighbors4(cell))
            if not unknown_neighbor:
                continue
            # Prefer frontiers that are not repeatedly visited.
            visit_penalty = 0.8 * self.visit_count.get(cell, 0)
            dist = math.hypot(di, dj)
            candidates.append((dist + visit_penalty, cell))

        candidates.sort(key=lambda x: x[0])
        return [c for _, c in candidates[:max_candidates]]

    def unknown_count_radius(self, cell: Cell, radius_cells: int) -> int:
        """Count unknown cells around a candidate frontier.

        A doorway/corridor opening generally has a large unknown region just beyond
        it, while a small noisy boundary frontier inside the current room has much
        less unknown mass.  This is used as an information-gain term so the drone
        exits rooms instead of orbiting their perimeter.
        """
        radius_cells = max(1, int(radius_cells))
        ci, cj = cell
        count = 0
        rr = radius_cells * radius_cells
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if di * di + dj * dj <= rr and (ci + di, cj + dj) not in self.logodds:
                    count += 1
        return count

    def unknown_gain_ahead_cell(self, cell: Cell, yaw_deg: float, lookahead_m: float, width_m: float) -> int:
        """Count unknown cells in a rectangular wedge beyond a frontier/heading."""
        yaw = math.radians(float(yaw_deg))
        c = math.cos(yaw)
        s = math.sin(yaw)
        n0, e0 = self.cell_to_world(cell)
        step = max(self.res, 0.40)
        lateral_step = max(self.res, 0.40)
        count = 0
        seen: Set[Cell] = set()
        d = step
        while d <= max(step, float(lookahead_m)):
            half = max(0.4, 0.5 * float(width_m) + 0.10 * d)
            lat = -half
            while lat <= half:
                n = n0 + d * c - lat * s
                e = e0 + d * s + lat * c
                cc = self.world_to_cell(n, e)
                if cc not in seen:
                    seen.add(cc)
                    if cc not in self.logodds:
                        count += 1
                lat += lateral_step
            d += step
        return count

    def path_segment_collision_free(
        self,
        start_ne: Tuple[float, float],
        end_ne: Tuple[float, float],
        inflation_extra_m: float = 0.0,
        ignore_start_m: float = 0.0,
        ignore_end_m: float = 0.0,
    ) -> bool:
        """
        Check a straight-line segment against inflated occupied grid cells.

        ignore_start_m is important in tight rooms: the current drone cell can be
        inside the inflated obstacle mask even when the real vehicle is not colliding.
        Without this, the planner can get stuck rejecting every first segment.
        """
        blocked = self.inflated_occupied()
        extra_cells = max(0, int(math.ceil(float(inflation_extra_m) / self.res)))
        if extra_cells > 0:
            expanded: set[Cell] = set(blocked)
            for ci, cj in blocked:
                for di in range(-extra_cells, extra_cells + 1):
                    for dj in range(-extra_cells, extra_cells + 1):
                        expanded.add((ci + di, cj + dj))
            blocked = expanded

        n0, e0 = start_ne
        n1, e1 = end_ne
        dist = math.hypot(n1 - n0, e1 - e0)
        if dist <= max(0.05, float(ignore_start_m)):
            return True

        steps = max(1, int(math.ceil(dist / max(0.10, self.res * 0.5))))
        for k in range(steps + 1):
            t = k / steps
            travelled = t * dist
            if travelled < float(ignore_start_m):
                continue
            if float(ignore_end_m) > 0.0 and travelled > dist - float(ignore_end_m):
                continue
            cell = self.world_to_cell(n0 + t * (n1 - n0), e0 + t * (e1 - e0))
            if cell in blocked or self.is_occupied(cell):
                return False
        return True


    def cells_near_segment(
        self,
        start_ne: Tuple[float, float],
        end_ne: Tuple[float, float],
        radius_m: float = 0.45,
        ignore_start_m: float = 0.0,
        ignore_end_m: float = 0.0,
    ) -> set[Cell]:
        """Return grid cells inside a corridor around a world-space segment."""
        n0, e0 = start_ne
        n1, e1 = end_ne
        dist = math.hypot(n1 - n0, e1 - e0)
        if dist <= 1e-6:
            return set()
        steps = max(1, int(math.ceil(dist / max(0.10, self.res * 0.5))))
        rad_cells = max(0, int(math.ceil(float(radius_m) / self.res)))
        cells: set[Cell] = set()
        for k in range(steps + 1):
            t = k / steps
            travelled = t * dist
            if travelled < float(ignore_start_m):
                continue
            if float(ignore_end_m) > 0.0 and travelled > dist - float(ignore_end_m):
                continue
            base = self.world_to_cell(n0 + t * (n1 - n0), e0 + t * (e1 - e0))
            for di in range(-rad_cells, rad_cells + 1):
                for dj in range(-rad_cells, rad_cells + 1):
                    if di * di + dj * dj <= rad_cells * rad_cells:
                        cells.add((base[0] + di, base[1] + dj))
        return cells

    def occupied_cells_near_segment(
        self,
        start_ne: Tuple[float, float],
        end_ne: Tuple[float, float],
        radius_m: float = 0.45,
        ignore_start_m: float = 0.0,
        ignore_end_m: float = 0.0,
    ) -> set[Cell]:
        cells = self.cells_near_segment(start_ne, end_ne, radius_m, ignore_start_m, ignore_end_m)
        blocked = self.inflated_occupied()
        return {c for c in cells if c in blocked or self.is_occupied(c)}

    def clear_ghost_cells_near_segment(
        self,
        start_ne: Tuple[float, float],
        end_ne: Tuple[float, float],
        radius_m: float = 0.45,
        amount: int = 5,
        ignore_start_m: float = 0.0,
        ignore_end_m: float = 0.0,
    ) -> int:
        """Actively clear cells that were falsely marked occupied along a verified-open path.

        This is only called after a settled depth check says the direction is physically
        open.  It is useful for depth-map ghost artefacts at corners: the global map may
        contain a stale black cell, but the current camera view proves the path is clear.
        """
        cells = self.cells_near_segment(start_ne, end_ne, radius_m, ignore_start_m, ignore_end_m)
        changed = 0
        for cell in cells:
            old = self.logodds.get(cell, 0)
            if old >= 0:
                self.logodds[cell] = max(-self.max_abs_log, old - int(amount))
                changed += 1
            elif cell not in self.logodds:
                self.logodds[cell] = -1
                changed += 1
        if changed:
            self._regularized_cache_key = None
            self._regularized_cache = None
        return changed

    def save_debug_map(self, output_dir: Path, pose: Optional[Dict[str, float]] = None) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.logodds:
            # Always create a visible diagnostic file. If this appears, the mission did not receive
            # usable depth frames or update_from_depth() never marked any cells.
            img = np.full((320, 520, 3), 210, dtype=np.uint8)
            cv2.putText(img, "NO OCCUPANCY DATA", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 180), 2)
            cv2.putText(img, "Check depth topic /depth_camera and pose telemetry", (35, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
            path = output_dir / "occupancy_grid.png"
            cv2.imwrite(str(path), img)
            with (output_dir / "map_diagnostics.txt").open("w") as f:
                f.write("No occupancy cells were created. Depth frames may be missing, invalid, or the program ended before mapping.\n")
            return path

        cells = list(self.logodds.keys())
        is_ = [c[0] for c in cells]
        js = [c[1] for c in cells]
        min_i, max_i = min(is_), max(is_)
        min_j, max_j = min(js), max(js)

        pad = 8
        H = max_i - min_i + 1 + 2 * pad
        W = max_j - min_j + 1 + 2 * pad
        img = np.full((H, W, 3), 210, dtype=np.uint8)  # unknown grey

        for cell, v in self.logodds.items():
            r = max_i - cell[0] + pad
            c = cell[1] - min_j + pad
            if v >= 2:
                # v31: when map regularisation is enabled, show raw occupied
                # evidence as grey and overlay accepted line/obstacle primitives
                # as black below.  This makes the debug image reflect the
                # standardised map without hiding raw evidence completely.
                if bool(getattr(self, "map_regularize_clean_display", True)) and self._regularization_enabled():
                    img[r, c] = (95, 95, 95)
                else:
                    img[r, c] = (20, 20, 20)      # occupied black-ish
            elif v <= -1:
                img[r, c] = (250, 250, 250)   # free white

        # visited cells blue-ish in BGR
        for cell, count in self.visit_count.items():
            r = max_i - cell[0] + pad
            c = cell[1] - min_j + pad
            if 0 <= r < H and 0 <= c < W:
                img[r, c] = (255, 160, 70)

        if pose is not None:
            pc = self.world_to_cell(pose["north"], pose["east"])
            r = max_i - pc[0] + pad
            c = pc[1] - min_j + pad
            cv2.circle(img, (c, r), 3, (0, 0, 255), -1)

        # Overlay regularized primitives as solid black so the debug map shows
        # the line/rectangle layer used by the planner when enabled.
        if bool(getattr(self, "map_regularize", False)):
            try:
                for cell in self.regularized_occupied_cells():
                    r = max_i - cell[0] + pad
                    c = cell[1] - min_j + pad
                    if 0 <= r < H and 0 <= c < W:
                        img[r, c] = (0, 0, 0)
                # Export fitted primitives for debugging/tuning.
                prim_path = output_dir / "map_primitives.csv"
                with prim_path.open("w", newline="") as pf:
                    pw = csv.writer(pf)
                    pw.writerow(["type", "north0", "east0", "north1", "east1", "support"])
                    for prim in getattr(self, "_regularized_primitives", []):
                        pw.writerow(list(prim))
            except Exception as exc:
                print(f"[MAP] regularized overlay failed: {exc}")

        # Redraw the vehicle marker after the regularized overlay. Otherwise a
        # regularized/raw occupied cell at the same grid location can visually
        # appear as a black dot inside the red marker even after self-clearing.
        if pose is not None:
            pc = self.world_to_cell(pose["north"], pose["east"])
            r = max_i - pc[0] + pad
            c = pc[1] - min_j + pad
            if 0 <= r < H and 0 <= c < W:
                cv2.circle(img, (c, r), 3, (0, 0, 255), -1)

        scale = 5
        img_big = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        path = output_dir / "occupancy_grid.png"
        cv2.imwrite(str(path), img_big)

        npz_path = output_dir / "occupancy_grid_raw.npz"
        arr = np.array([[i, j, v] for (i, j), v in self.logodds.items()], dtype=np.float32)
        np.savez(str(npz_path), cells_logodds=arr, resolution_m=self.res)
        return path


class FrontierPlanner:
    def __init__(self, mapper: OccupancyGridMapper):
        self.mapper = mapper

    def _heuristic(self, a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _is_traversable(self, cell: Cell, blocked: set[Cell]) -> bool:
        return self.mapper.is_known_free(cell) and cell not in blocked

    def astar(self, start: Cell, goal: Cell, max_expansions: int = 12000) -> Optional[List[Cell]]:
        blocked = self.mapper.inflated_occupied()
        if not self._is_traversable(start, blocked):
            # Current cell can be inside inflated obstacle because of noisy depth. Allow it.
            self.mapper.mark_free(start, amount=3)
        if not self._is_traversable(goal, blocked):
            return None

        open_heap: List[Tuple[float, int, Cell]] = []
        push_id = 0
        heapq.heappush(open_heap, (0.0, push_id, start))
        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        g_score: Dict[Cell, float] = {start: 0.0}
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, current = heapq.heappop(open_heap)
            expansions += 1
            if current == goal:
                path: List[Cell] = []
                c: Optional[Cell] = current
                while c is not None:
                    path.append(c)
                    c = came_from[c]
                path.reverse()
                return path

            for nb in self.mapper.neighbors8(current):
                if not self._is_traversable(nb, blocked):
                    continue

                # v31: prevent A* from cutting diagonally through a corner.
                # Without this check, a diagonal edge can pass between two
                # inflated obstacle cells, then the smoothed/strided waypoint
                # follower clips the physical wall corner.
                is_diag = (nb[0] != current[0] and nb[1] != current[1])
                if is_diag and bool(getattr(self.mapper, "no_diagonal_corner_cutting", True)):
                    side_a = (current[0], nb[1])
                    side_b = (nb[0], current[1])
                    if (not self._is_traversable(side_a, blocked)) or (not self._is_traversable(side_b, blocked)):
                        continue

                step_cost = 1.414 if is_diag else 1.0
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    push_id += 1
                    f = tentative + self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, push_id, nb))

        return None

    def plan_to_best_frontier(
        self,
        pose: Dict[str, float],
        min_frontier_dist_m: float = 1.8,
        max_frontiers_to_try: int = 60,
        rejected_cells: Optional[Set[Cell]] = None,
        recent_cells: Optional[Set[Cell]] = None,
        avoid_recent: bool = False,
        max_recent_path_hits: int = 0,
        recent_penalty_weight: float = 8.0,
        turnback_penalty_weight: float = 4.0,
    ) -> Optional[List[Tuple[float, float]]]:
        start = self.mapper.world_to_cell(pose["north"], pose["east"])
        rejected_cells = rejected_cells or set()
        recent_cells = recent_cells or set()
        candidates = self.mapper.frontiers(start, min_dist_m=min_frontier_dist_m)
        if not candidates:
            return None

        best_path: Optional[List[Cell]] = None
        best_score = float("inf")
        yaw_now = float(pose["yaw_deg"])

        for goal in candidates[:max_frontiers_to_try]:
            if goal in rejected_cells:
                continue
            path = self.astar(start, goal)
            if path is None or len(path) < 2:
                continue
            n0, e0 = self.mapper.cell_to_world(start)
            ng, eg = self.mapper.cell_to_world(goal)
            desired_yaw = yaw_from_vector_deg(ng - n0, eg - e0)
            turn_penalty = abs(yaw_error_deg(desired_yaw, yaw_now)) / 90.0
            # v35: discourage turning back into recently travelled cells during normal exploration.
            # Backtracking is still allowed by the explicit dead-end breadcrumb escape path; the
            # global frontier planner should prefer fresh branches and not reuse the blue trail.
            recent_hits = 0
            if recent_cells:
                # Ignore the first few cells around the current pose so the drone is not blocked
                # just because its own current cell is in the history buffer.
                recent_hits = sum(1 for idx, c in enumerate(path) if idx > 3 and c in recent_cells)
                if avoid_recent and recent_hits > int(max_recent_path_hits):
                    continue
            revisit_penalty = sum(self.mapper.visit_count.get(c, 0) for c in path) * 0.05
            recent_penalty = float(recent_penalty_weight) * float(recent_hits)
            turnback_penalty = float(turnback_penalty_weight) * max(0.0, abs(yaw_error_deg(desired_yaw, yaw_now)) - 95.0) / 90.0

            # v38: information-gain / gateway bias.  The old planner usually chose
            # the nearest frontier, which is fast locally but can make the drone
            # orbit the first room because every little edge of the current room is
            # a nearby frontier.  Reward frontiers that open into a large unknown
            # region, especially unknown space beyond the frontier in the direction
            # of travel.  This makes doorway/corridor exits beat perimeter loops.
            radius_m = float(getattr(self.mapper, "frontier_unknown_radius_m", 2.8))
            radius_cells = max(1, int(round(radius_m / self.mapper.res)))
            unknown_near = self.mapper.unknown_count_radius(goal, radius_cells)
            unknown_ahead = self.mapper.unknown_gain_ahead_cell(
                goal,
                desired_yaw,
                float(getattr(self.mapper, "frontier_unknown_lookahead_m", 6.0)),
                float(getattr(self.mapper, "frontier_unknown_width_m", 2.2)),
            )
            unknown_reward = (
                float(getattr(self.mapper, "frontier_unknown_weight", 0.13)) * math.sqrt(float(unknown_near))
                + float(getattr(self.mapper, "frontier_unknown_ahead_weight", 0.20)) * math.sqrt(float(unknown_ahead))
            )
            # A small progress reward helps the drone choose a farther exit instead
            # of another short perimeter frontier, while A* still enforces reachability.
            progress_reward = float(getattr(self.mapper, "frontier_progress_weight", 0.015)) * min(float(len(path)), 80.0)

            # v39: topological gateway bias.  A plain nearest-frontier planner can
            # orbit one room because local wall/perimeter frontiers keep looking cheap.
            # Here, a reachable frontier is treated as a high-value gateway only when
            # it opens into a significant unknown region ahead.  Non-gateway perimeter
            # frontiers are still allowed as fallback, but they receive a large cost.
            topo_gateway_mode = bool(getattr(self.mapper, "topological_gateway_mode", False))
            gateway_penalty = 0.0
            if topo_gateway_mode:
                min_unknown_ahead = int(getattr(self.mapper, "topo_frontier_min_unknown_ahead", 22))
                min_unknown_near = int(getattr(self.mapper, "topo_frontier_min_unknown_near", 18))
                is_gateway_frontier = (unknown_ahead >= min_unknown_ahead) or (unknown_near >= min_unknown_near)
                if is_gateway_frontier:
                    unknown_reward += float(getattr(self.mapper, "topo_frontier_gateway_bonus", 8.0))
                    # Favour gateways that require meaningful progress out of the current zone.
                    progress_reward += float(getattr(self.mapper, "topo_frontier_depth_bonus", 0.06)) * min(float(len(path)), 120.0)
                else:
                    gateway_penalty += float(getattr(self.mapper, "topo_frontier_perimeter_penalty", 24.0))

            path_cost_weight = float(getattr(self.mapper, "frontier_path_cost_weight", 1.0))
            local_loop_penalty = 0.0
            if (
                len(path) <= int(getattr(self.mapper, "frontier_local_loop_path_cells", 14))
                and unknown_ahead < int(getattr(self.mapper, "topo_frontier_min_unknown_ahead", 22))
                and unknown_near < int(getattr(self.mapper, "topo_frontier_min_unknown_near", 18))
            ):
                local_loop_penalty = float(getattr(self.mapper, "frontier_local_loop_penalty", 0.0))

            score = path_cost_weight * len(path) + turn_penalty + revisit_penalty + recent_penalty + turnback_penalty + gateway_penalty + local_loop_penalty - unknown_reward - progress_reward
            if score < best_score:
                best_score = score
                best_path = path

        if best_path is None:
            return None

        # Simplify path, but keep corner/turn cells.
        # v31: long strides are fast in corridors but dangerous around corners,
        # because they make the drone cut across the inside edge.  We therefore
        # keep extra waypoints around any A* direction change while still using
        # long strides along straight corridors.
        world_path: List[Tuple[float, float]] = []
        stride_m = float(getattr(self.mapper, "path_stride_m", 1.20))
        stride = max(1, int(round(stride_m / self.mapper.res)))
        turn_pad = max(0, int(getattr(self.mapper, "path_turn_padding_cells", 1)))
        turn_indices: Set[int] = set()
        for idx in range(1, len(best_path) - 1):
            p0 = best_path[idx - 1]
            p1 = best_path[idx]
            p2 = best_path[idx + 1]
            d1 = (int(math.copysign(1, p1[0] - p0[0])) if p1[0] != p0[0] else 0, int(math.copysign(1, p1[1] - p0[1])) if p1[1] != p0[1] else 0)
            d2 = (int(math.copysign(1, p2[0] - p1[0])) if p2[0] != p1[0] else 0, int(math.copysign(1, p2[1] - p1[1])) if p2[1] != p1[1] else 0)
            if d1 != d2:
                for k in range(idx - turn_pad, idx + turn_pad + 1):
                    if 0 < k < len(best_path):
                        turn_indices.add(k)

        last_kept_idx = 0
        for idx, cell in enumerate(best_path):
            if idx == 0:
                continue
            keep = (idx == len(best_path) - 1) or (idx in turn_indices) or ((idx - last_kept_idx) >= stride)
            if keep:
                world_path.append(self.mapper.cell_to_world(cell))
                last_kept_idx = idx
        return world_path



# -----------------------------------------------------------------------------
# Reactive collision avoidance safety layer
# -----------------------------------------------------------------------------
class DepthSafetyFilter:
    """
    Last-line safety layer in front of every velocity command.

    v19/v20 changes the safety logic from a simple left/center/right gate into a
    direction-aware collision guard.  The old logic could approve a NED velocity
    whose body-frame direction was diagonal/sideways even though the obstacle was
    in that exact movement corridor.  This version projects the depth image into
    a lightweight body-frame point set and checks a capsule/corridor in the
    *commanded direction of travel* before allowing motion.

    Important design choices:
    - Emergency default is BRAKE, not reverse. A forward-facing depth camera does
      not know what is behind the drone, so reversing near a wall is unsafe.
    - Sideways motion is cancelled if the side clearance is poor.
    - Forward speed is scaled by distance-to-obstacle and by stopping distance.
    - The organiser's AvoidancePlanner is only blended when it agrees with the
      directional safety corridor; it never overrides a hard stop.
    """

    def __init__(self, K: np.ndarray, args: argparse.Namespace):
        self.args = args
        self.enabled = not bool(getattr(args, "no_reactive_avoidance", False))
        self.K = K.astype(np.float32)
        self.fx = float(K[0, 0])
        self.cx = float(K[0, 2])
        self.last_safe_vn = 0.0
        self.last_safe_ve = 0.0
        self.emergency_streak = 0
        self.front_emergency_streak = 0
        self.last_corridor_clearance = float("inf")

        self.planner = None
        if self.enabled and AvoidancePlanner is not None:
            self.planner = AvoidancePlanner(
                K=K,
                width=640,
                height=480,
                max_speed=float(args.cruise_speed_m_s),
                safe_distance=float(args.avoid_safe_m),
                critical_distance=float(args.avoid_critical_m),
                smoothing_alpha=0.35,
            )
            print("[AVOID] Using organiser AvoidancePlanner.py as secondary reactive layer")
        elif self.enabled:
            print("[AVOID] AvoidancePlanner.py unavailable; using built-in direction-aware safety only")
        else:
            print("[AVOID] Reactive avoidance disabled by --no-reactive-avoidance")

    def _valid_depth(self, depth_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if depth_frame is None:
            return None
        arr = np.asarray(depth_frame, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _clearances(self, depth_frame: Optional[np.ndarray], fallback: float) -> Tuple[float, float, float, float]:
        arr = self._valid_depth(depth_frame)
        if arr is None:
            return 0.0, 0.0, 0.0, 0.0
        h, w = arr.shape[:2]
        # Avoid very low rows because they can see floor texture; keep enough vertical
        # extent to catch barrels, walls and box corners at 3 m altitude.
        y1 = int(float(getattr(self.args, "avoid_band_y1", 0.22)) * h)
        y2 = int(float(getattr(self.args, "avoid_band_y2", 0.72)) * h)
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        band = arr[y1:y2, :]
        percentile = float(getattr(self.args, "clearance_percentile", 10.0))
        left = safe_percentile(band[:, : w // 3], percentile, fallback)
        center = safe_percentile(band[:, w // 3: 2 * w // 3], percentile, fallback)
        right = safe_percentile(band[:, 2 * w // 3:], percentile, fallback)
        min_all = min(left, center, right)
        return left, center, right, min_all

    def _front_close_region(self, depth_frame: Optional[np.ndarray], threshold_m: float) -> Optional[np.ndarray]:
        """Return a close-pixel mask for the central front safety band.

        v25 note: a raw pixel count is not reliable enough in Gazebo.  Thin
        floor seams, depth discontinuities on barrel edges, or one stale row can
        easily produce hundreds of close pixels.  We therefore analyse connected
        components and vertical/column span before declaring a *real* front
        emergency.
        """
        arr = self._valid_depth(depth_frame)
        if arr is None:
            return None
        h, w = arr.shape[:2]
        y1 = int(float(getattr(self.args, "avoid_band_y1", 0.22)) * h)
        y2 = int(float(getattr(self.args, "avoid_hard_band_y2", getattr(self.args, "avoid_band_y2", 0.58))) * h)
        x1 = int(float(getattr(self.args, "front_close_x1", 0.38)) * w)
        x2 = int(float(getattr(self.args, "front_close_x2", 0.62)) * w)
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        region = arr[y1:y2, x1:x2]
        valid = np.isfinite(region) & (region > 0.18) & (region < float(threshold_m))
        return valid.astype(np.uint8)

    def _front_close_count(self, depth_frame: Optional[np.ndarray], threshold_m: float) -> int:
        mask = self._front_close_region(depth_frame, threshold_m)
        if mask is None:
            return 0
        return int(np.count_nonzero(mask))

    def _front_obstacle_stats(self, depth_frame: Optional[np.ndarray], threshold_m: float) -> Dict[str, int]:
        """Connected-component support for a close obstacle directly ahead.

        Returns total close pixels plus the largest connected component's area and
        span.  Real obstacles in front of the drone usually occupy a blob with
        vertical *and* horizontal extent.  False floor/depth artefacts often look
        like a thin line, small island, or texture stripe.
        """
        mask = self._front_close_region(depth_frame, threshold_m)
        if mask is None or mask.size == 0:
            return {"count": 0, "area": 0, "row_span": 0, "col_span": 0}

        total = int(np.count_nonzero(mask))
        if total <= 0:
            return {"count": 0, "area": 0, "row_span": 0, "col_span": 0}

        # Morphological open removes one-pixel floor seams; close joins true blobs.
        k = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if n <= 1:
            return {"count": total, "area": 0, "row_span": 0, "col_span": 0}

        # Ignore background at index 0.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_rel = int(np.argmax(areas))
        best = best_rel + 1
        area = int(stats[best, cv2.CC_STAT_AREA])
        col_span = int(stats[best, cv2.CC_STAT_WIDTH])
        row_span = int(stats[best, cv2.CC_STAT_HEIGHT])
        return {"count": total, "area": area, "row_span": row_span, "col_span": col_span}

    def front_obstacle_supported(self, depth_frame: Optional[np.ndarray], threshold_m: float) -> bool:
        """True only when a close front return is spatially convincing.

        This is deliberately stricter than _front_close_count(). It is used only
        for emergency braking. Slowdowns may still use a looser count.
        """
        st = self._front_obstacle_stats(depth_frame, threshold_m)
        count_min = int(getattr(self.args, "front_close_min_pixels", 220))
        area_min = int(getattr(self.args, "front_component_min_area_px", 80))
        row_min = int(getattr(self.args, "front_component_min_row_span_px", 10))
        col_min = int(getattr(self.args, "front_component_min_col_span_px", 8))
        return (
            st["count"] >= count_min
            and st["area"] >= area_min
            and st["row_span"] >= row_min
            and st["col_span"] >= col_min
        )

    def _directional_clearance(
        self,
        depth_frame: Optional[np.ndarray],
        forward: float,
        right_cmd: float,
    ) -> Tuple[float, int]:
        """Return nearest obstacle distance along the commanded body-frame direction.

        Coordinates used here:
            forward axis = depth z
            right axis   = camera x from pixel column and depth

        The method checks a finite-width corridor/capsule rather than only the
        center third of the image. This catches diagonal commands and sideways
        drift toward walls.
        """
        arr = self._valid_depth(depth_frame)
        speed = math.hypot(forward, right_cmd)
        if arr is None or speed < 1e-4:
            return float(self.args.map_ray_max_m), 0

        h, w = arr.shape[:2]
        y1 = int(float(getattr(self.args, "avoid_band_y1", 0.22)) * h)
        y2 = int(float(getattr(self.args, "avoid_band_y2", 0.72)) * h)
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        stride = max(1, int(getattr(self.args, "collision_depth_stride", 6)))

        # Downsample the depth image to keep this cheap enough for VMware.
        band = arr[y1:y2:stride, ::stride]
        if band.size == 0:
            return float(self.args.map_ray_max_m), 0

        z = band.reshape(-1).astype(np.float32)
        valid = np.isfinite(z) & (z > 0.18) & (z < float(getattr(self.args, "collision_lookahead_m", 4.0)))
        if not np.any(valid):
            return float(self.args.map_ray_max_m), 0

        cols = np.arange(0, w, stride, dtype=np.float32)
        cols = cols[: band.shape[1]]
        cols2 = np.tile(cols, band.shape[0]).reshape(-1)
        z = z[valid]
        cols2 = cols2[valid]

        # Approximate body-frame right coordinate from depth intrinsics.
        x_right = (cols2 - self.cx) * z / max(1e-6, self.fx)
        p_forward = z
        p_right = x_right

        # Unit vector in commanded travel direction.
        df = forward / speed
        dr = right_cmd / speed
        along = p_forward * df + p_right * dr
        lateral = np.abs(-p_forward * dr + p_right * df)

        # Corridor width grows slightly with distance to account for drone radius,
        # prop span, tracking error, and controller lag.
        radius = float(getattr(self.args, "collision_radius_m", 0.60))
        growth = float(getattr(self.args, "collision_corridor_growth", 0.10))
        corridor = radius + growth * np.clip(along, 0.0, float(getattr(self.args, "collision_lookahead_m", 4.0)))
        mask = (along > 0.05) & (lateral <= corridor)
        count = int(np.count_nonzero(mask))
        min_count = int(getattr(self.args, "collision_min_points", 3))
        if count < min_count:
            return float(self.args.map_ray_max_m), count
        return float(np.percentile(along[mask], 5)), count

    def filter_velocity_ned(
        self,
        desired_vn: float,
        desired_ve: float,
        pose: Dict[str, float],
        depth_frame: Optional[np.ndarray],
    ) -> Tuple[float, float, Dict[str, Any]]:
        info: Dict[str, Any] = {"active": False, "emergency": False, "reason": "clear"}
        if not self.enabled:
            return desired_vn, desired_ve, info

        yaw = float(pose.get("yaw_deg", 0.0))
        left, center, right, min_all = self._clearances(depth_frame, float(self.args.map_ray_max_m))
        info.update({"left": left, "center": center, "right": right, "min": min_all})

        # Never continue a stale/absent command when depth is absent.
        if depth_frame is None:
            info.update({"active": True, "emergency": True, "reason": "no_depth_brake"})
            self.last_safe_vn = 0.0
            self.last_safe_ve = 0.0
            return 0.0, 0.0, info

        desired_forward, desired_right = ned_to_body(desired_vn, desired_ve, yaw)
        forward = float(desired_forward)
        right_cmd = float(desired_right)
        max_speed = float(self.args.cruise_speed_m_s)

        critical = float(self.args.avoid_critical_m)
        emergency = float(self.args.emergency_stop_m)
        safe = float(self.args.avoid_safe_m)
        side_critical = float(self.args.side_critical_m)
        side_hard = float(getattr(self.args, "side_hard_m", 0.48))

        corridor_clearance, corridor_count = self._directional_clearance(depth_frame, forward, right_cmd)
        self.last_corridor_clearance = corridor_clearance
        info.update({"corridor": corridor_clearance, "corridor_count": corridor_count})

        desired_speed = math.hypot(forward, right_cmd)
        # Dynamic stopping distance: perception/controller delay + braking distance.
        reaction_s = float(getattr(self.args, "collision_reaction_s", 0.60))
        brake_accel = max(0.05, float(getattr(self.args, "collision_brake_accel_m_s2", 0.80)))
        dynamic_stop = emergency + desired_speed * reaction_s + (desired_speed * desired_speed) / (2.0 * brake_accel)

        # Hard stops: do not reverse by default. The camera cannot see behind the drone.
        front_close_count = self._front_close_count(depth_frame, emergency)
        front_slow_count = self._front_close_count(depth_frame, safe)
        front_stats = self._front_obstacle_stats(depth_frame, emergency)
        info.update({
            "front_close_count": front_close_count,
            "front_slow_count": front_slow_count,
            "front_area": front_stats.get("area", 0),
            "front_row_span": front_stats.get("row_span", 0),
            "front_col_span": front_stats.get("col_span", 0),
        })

        raw_hard_front = center < emergency and self.front_obstacle_supported(depth_frame, emergency)
        if raw_hard_front:
            self.front_emergency_streak += 1
        else:
            self.front_emergency_streak = 0

        hard_front = raw_hard_front and self.front_emergency_streak >= int(getattr(self.args, "front_hard_confirm_frames", 2))
        hard_side = (left < side_hard and right_cmd < -0.03) or (right < side_hard and right_cmd > 0.03)

        # v22: do not let a few pixels in the direction-aware corridor freeze the
        # drone forever. In this map the depth camera can see side-wall/floor-edge
        # artefacts that fall inside the mathematical corridor even when the center
        # view is clear. Treat the corridor as a hard stop only when it is both very
        # close and well supported; otherwise it becomes a slowdown.
        corridor_hard_clearance = float(getattr(self.args, "corridor_hard_brake_clearance_m", 0.75))
        corridor_hard_min_points = int(getattr(self.args, "corridor_hard_brake_min_points", 35))
        hard_corridor = (
            corridor_clearance < corridor_hard_clearance
            and corridor_count >= corridor_hard_min_points
        )

        # If the forward view is very open and both side sectors have reasonable
        # clearance, assume a marginal corridor hit is a projection/noise artefact.
        # True front obstacles are still handled by hard_front; side impacts by hard_side.
        if (
            hard_corridor
            and center > float(getattr(self.args, "corridor_clear_front_override_m", 3.0))
            and left > side_critical
            and right > side_critical
        ):
            hard_corridor = False

        if hard_front or hard_side or hard_corridor or min_all < float(getattr(self.args, "absolute_min_depth_m", 0.38)):
            self.emergency_streak += 1
            action = str(getattr(self.args, "emergency_action", "brake"))
            forward = 0.0
            right_cmd = 0.0

            # Optional sidestep is allowed only if the requested escape side is actually open.
            if action == "sidestep" and self.emergency_streak >= int(getattr(self.args, "emergency_confirm_frames", 1)):
                if left > right and left > float(getattr(self.args, "sidestep_min_clearance_m", 1.6)):
                    right_cmd = -min(0.16, max_speed * 0.35)
                elif right > left and right > float(getattr(self.args, "sidestep_min_clearance_m", 1.6)):
                    right_cmd = min(0.16, max_speed * 0.35)

            reasons = []
            if hard_front:
                reasons.append("front")
            if hard_side:
                reasons.append("side")
            if hard_corridor:
                reasons.append("corridor")
            if min_all < float(getattr(self.args, "absolute_min_depth_m", 0.38)):
                reasons.append("absolute_min")
            info.update({"active": True, "emergency": True, "reason": "hard_brake:" + "+".join(reasons)})
        else:
            self.emergency_streak = 0

            # Scale motion in the actual travel corridor. This is what prevents
            # diagonal corner-cutting at higher cruise speeds.
            corridor_safe = max(safe, dynamic_stop + float(getattr(self.args, "collision_extra_margin_m", 0.35)))
            if corridor_clearance < corridor_safe and desired_speed > 1e-4:
                scale = clamp((corridor_clearance - dynamic_stop) / max(0.05, corridor_safe - dynamic_stop), 0.0, 1.0)

                # v22: when the front sector is open and the corridor hit is likely
                # a side/floor artefact, creep instead of freezing. This gives the
                # mapper a new viewpoint and prevents infinite brake-plan loops.
                if center > float(getattr(self.args, "corridor_clear_front_override_m", 3.0)) and left > side_hard and right > side_hard:
                    # v24: if the camera centre is clearly open, a non-hard corridor
                    # hit is probably a side-wall/floor-edge projection. Creep forward
                    # instead of freezing at safe=(0, 0).
                    scale = max(scale, float(getattr(self.args, "corridor_min_scale_front_clear", 0.35)))
                elif bool(getattr(self.args, "corridor_allow_crawl", True)) and left > side_hard and right > side_hard:
                    scale = max(scale, float(getattr(self.args, "corridor_min_scale_always", 0.18)))

                forward *= scale
                right_cmd *= scale
                info.update({"active": True, "reason": "corridor_slowdown"})

            # Classic center slowdown still helps when moving roughly forward.
            if center < safe and forward > 0.0 and front_slow_count >= int(getattr(self.args, "front_slow_min_pixels", 300)):
                scale = clamp((center - emergency) / max(0.05, safe - emergency), 0.0, 1.0)
                forward = min(forward, max_speed * scale)
                info.update({"active": True, "reason": info.get("reason", "") + "+front_slowdown"})

            # Side-wall safety: cancel motion into a wall first; only add a small
            # outward bias if very close. This avoids the old oscillatory wall-hugging.
            if left < side_critical and right_cmd < 0.0:
                right_cmd = 0.0
                info.update({"active": True, "reason": info.get("reason", "") + "+cancel_left"})
            if right < side_critical and right_cmd > 0.0:
                right_cmd = 0.0
                info.update({"active": True, "reason": info.get("reason", "") + "+cancel_right"})
            if left < side_hard:
                right_cmd = max(right_cmd, min(0.12, max_speed * 0.25))
                info.update({"active": True, "reason": info.get("reason", "") + "+left_wall_bias"})
            if right < side_hard:
                right_cmd = min(right_cmd, -min(0.12, max_speed * 0.25))
                info.update({"active": True, "reason": info.get("reason", "") + "+right_wall_bias"})

            # v24: if the only issue is a non-emergency corridor slowdown while the
            # center view is open, enforce a small forward crawl. This prevents the
            # exact safe=(0,0) freeze where the drone has a viable path but never
            # gathers a new viewpoint. Hard-front/side/corridor emergencies still
            # brake above.
            if (
                bool(getattr(self.args, "force_crawl_on_corridor_slowdown", True))
                and "corridor_slowdown" in str(info.get("reason", ""))
                and center > float(getattr(self.args, "corridor_clear_front_override_m", 3.0))
                and desired_speed > 1e-4
                and math.hypot(forward, right_cmd) < float(getattr(self.args, "corridor_crawl_speed_m_s", 0.10))
            ):
                # Preserve the intended travel direction in body coordinates.
                crawl = float(getattr(self.args, "corridor_crawl_speed_m_s", 0.10))
                df = desired_forward / max(1e-6, desired_speed)
                dr = desired_right / max(1e-6, desired_speed)
                forward = crawl * df
                right_cmd = crawl * dr
                info.update({"active": True, "reason": info.get("reason", "") + "+forced_crawl"})

            # Optional organiser AvoidancePlanner blend. Only use it when close to
            # obstacles, and never let it reintroduce motion into a blocked corridor.
            if self.planner is not None and ((center < safe and front_slow_count >= int(getattr(self.args, "front_slow_min_pixels", 300))) or left < side_critical or right < side_critical):
                try:
                    avoid_forward, avoid_right, avoid_info = self.planner.compute_velocity(depth_frame)
                    blend = clamp((safe - min(center, left, right)) / max(0.05, safe - critical), 0.15, 0.55)
                    cand_f = (1.0 - blend) * forward + blend * float(avoid_forward)
                    cand_r = (1.0 - blend) * right_cmd + blend * float(avoid_right)
                    cand_clearance, cand_count = self._directional_clearance(depth_frame, cand_f, cand_r)
                    if cand_clearance >= dynamic_stop and cand_count >= 0:
                        forward, right_cmd = cand_f, cand_r
                        info["avoid_info"] = avoid_info
                        info.update({"active": True, "reason": info.get("reason", "") + "+planner_blend"})
                except Exception as exc:
                    info["avoid_error"] = f"{type(exc).__name__}: {exc}"

        # Final speed cap.
        mag = math.hypot(forward, right_cmd)
        if mag > max_speed:
            forward *= max_speed / mag
            right_cmd *= max_speed / mag

        vn, ve = body_to_ned(forward, right_cmd, yaw)

        # Stronger smoothing on emergency transitions reduces sudden lateral kicks.
        alpha = float(getattr(self.args, "avoid_smoothing_alpha", 0.40))
        if info.get("emergency"):
            alpha = min(0.15, alpha)  # brake quickly; do not coast into walls.
        vn = alpha * self.last_safe_vn + (1.0 - alpha) * vn
        ve = alpha * self.last_safe_ve + (1.0 - alpha) * ve

        # If emergency says brake, force exactly zero to avoid smoothing leakage.
        if info.get("emergency") and str(getattr(self.args, "emergency_action", "brake")) == "brake":
            vn, ve = 0.0, 0.0

        self.last_safe_vn = vn
        self.last_safe_ve = ve
        return vn, ve, info

# -----------------------------------------------------------------------------
# Image saving / YOLO counting
# -----------------------------------------------------------------------------
@dataclass
class BarrelTrack:
    group: str
    north: float
    east: float
    confidence: float
    track_id: int = 0
    hits: int = 1
    first_seen_ms: int = field(default_factory=now_ms)
    last_seen_ms: int = field(default_factory=now_ms)

    def update(self, north: float, east: float, confidence: float) -> None:
        # Keep confirmed tracks stable.  YOLO/depth projection jitter can move the
        # same physical barrel by tens of centimetres between frames; a conservative
        # low-pass update prevents that jitter from creating multiple counted tracks.
        alpha = 0.85 if self.hits >= 2 else 0.65
        self.north = alpha * self.north + (1.0 - alpha) * north
        self.east = alpha * self.east + (1.0 - alpha) * east
        self.confidence = max(self.confidence, confidence)
        self.hits += 1
        self.last_seen_ms = now_ms()


class FrameLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = output_dir / "frames.csv"
        self._csv_file = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["filename", "timestamp_ms", "north", "east", "down", "yaw_deg", "mode", "note", "interest"])
        self.saved_count = 0

    def save(self, frame_bgr: np.ndarray, pose: Dict[str, float], mode: str, note: str, interest: str = "") -> str:
        ts = now_ms()
        filename = f"frame_{self.saved_count:06d}_{ts}.jpg"
        path = self.image_dir / filename
        cv2.imwrite(str(path), frame_bgr)
        self._writer.writerow(
            [
                filename,
                ts,
                float(pose.get("north", 0.0)),
                float(pose.get("east", 0.0)),
                float(pose.get("down", 0.0)),
                float(pose.get("yaw_deg", 0.0)),
                mode,
                note,
                interest,
            ]
        )
        self._csv_file.flush()
        self.saved_count += 1
        print(f"[PHOTO] saved {filename} note={note} interest={interest}")
        return filename

    def close(self) -> None:
        self._csv_file.close()


def color_interest(frame_bgr: np.ndarray, min_pixels: int) -> Tuple[bool, str, int]:
    """Cheap pre-YOLO detector for dataset collection: red/yellow coloured regions."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Red wraps around hue 0. Yellow is roughly 20-40 in OpenCV hue units.
    red = (((h <= 10) | (h >= 170)) & (s >= 55) & (v >= 45))
    yellow = ((h >= 18) & (h <= 45) & (s >= 45) & (v >= 55))

    # Ignore top 15% and bottom 10% to reduce UI/floor noise.
    H, W = h.shape
    roi = np.zeros_like(red, dtype=bool)
    roi[int(0.15 * H) : int(0.90 * H), int(0.05 * W) : int(0.95 * W)] = True
    red_count = int(np.count_nonzero(red & roi))
    yellow_count = int(np.count_nonzero(yellow & roi))

    label_parts = []
    if red_count >= min_pixels:
        label_parts.append(f"red_pixels={red_count}")
    if yellow_count >= min_pixels:
        label_parts.append(f"yellow_pixels={yellow_count}")
    return bool(label_parts), ";".join(label_parts), red_count + yellow_count


def get_depth_at_bbox(depth: Optional[np.ndarray], bbox: Sequence[float]) -> Optional[float]:
    if depth is None:
        return None
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx1 = int(x1 + 0.25 * (x2 - x1))
    cx2 = int(x1 + 0.75 * (x2 - x1))
    cy1 = int(y1 + 0.25 * (y2 - y1))
    cy2 = int(y1 + 0.75 * (y2 - y1))
    crop = depth[cy1:cy2, cx1:cx2]
    valid = crop[np.isfinite(crop) & (crop > 0.20) & (crop < 15.0)]
    if valid.size < 10:
        return None
    return float(np.median(valid))



def estimate_depth_from_bbox_height(
    bbox: Sequence[float],
    K: np.ndarray,
    physical_height_m: float = 1.05,
    min_depth_m: float = 0.6,
    max_depth_m: float = 12.0,
) -> Optional[float]:
    """Approximate range from apparent barrel height when the depth crop is invalid.

    This is deliberately used only as a fallback.  It lets yellow barrels near
    image edges / floor seams still become trackable instead of only appearing in
    annotated images.  The track merger still requires repeated hits before the
    object contributes to the final count.
    """
    try:
        _x1, y1, _x2, y2 = [float(v) for v in bbox]
        pix_h = max(1.0, y2 - y1)
        fy = float(K[1, 1])
        if not np.isfinite(fy) or fy <= 1.0:
            return None
        d = float(physical_height_m) * fy / pix_h
        if not np.isfinite(d):
            return None
        return float(max(min_depth_m, min(max_depth_m, d)))
    except Exception:
        return None


def crop_colour_group(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    min_pixels: int = 80,
    dominance_ratio: float = 1.20,
) -> Tuple[Optional[str], str]:
    """Infer red/yellow from the detected crop as a safety net.

    YOLO class IDs remain the primary signal, but the crop colour vote prevents
    all-yellow detections being written as red if the model/export has confusing
    names.  It also makes barrel_tracks.csv and detections/ agree with what is
    visible in annotated/.
    """
    h, w = frame_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    except Exception:
        return None, "colour_vote=invalid_bbox"
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, "colour_vote=empty_crop"
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "colour_vote=empty_crop"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_mask = (((hh <= 10) | (hh >= 170)) & (ss >= 45) & (vv >= 35))
    yellow_mask = ((hh >= 16) & (hh <= 48) & (ss >= 35) & (vv >= 45))

    # Ignore very dark/grey pixels because many barrel bodies have black/grey bands.
    red_count = int(np.count_nonzero(red_mask))
    yellow_count = int(np.count_nonzero(yellow_mask))
    info = f"colour_vote=red:{red_count},yellow:{yellow_count}"
    if red_count < min_pixels and yellow_count < min_pixels:
        return None, info
    if yellow_count >= min_pixels and yellow_count >= red_count * dominance_ratio:
        return "yellow", info
    if red_count >= min_pixels and red_count >= yellow_count * dominance_ratio:
        return "red", info
    return None, info

def detection_to_global_ned(bbox: Sequence[float], depth_m: float, pose: Dict[str, float], K: np.ndarray) -> Tuple[float, float]:
    x1, _, x2, _ = bbox
    u = 0.5 * (x1 + x2)
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    x_right = (u - cx) * depth_m / fx
    z_forward = depth_m
    yaw = math.radians(float(pose["yaw_deg"]))
    north = float(pose["north"]) + z_forward * math.cos(yaw) - x_right * math.sin(yaw)
    east = float(pose["east"]) + z_forward * math.sin(yaw) + x_right * math.cos(yaw)
    return north, east


def class_group(
    name: str,
    cls_id: int,
    red_classes: Sequence[str],
    yellow_classes: Sequence[str],
    red_class_ids: Sequence[int] = (),
    yellow_class_ids: Sequence[int] = (),
) -> Optional[str]:
    """Map a YOLO class to red/yellow.

    Some exported YOLO models preserve human-readable class names
    (for example red_barrel / yellow_barrel).  Some ONNX/export paths can expose
    class names as plain numeric strings.  For that reason we support both names
    and class IDs.  For the user's trained model the expected mapping is:

        class 0 -> red
        class 1 -> yellow
    """
    norm = name.strip().lower().replace(" ", "_").replace("-", "_")
    red = {x.strip().lower().replace(" ", "_").replace("-", "_") for x in red_classes if str(x).strip()}
    yellow = {x.strip().lower().replace(" ", "_").replace("-", "_") for x in yellow_classes if str(x).strip()}

    # Prefer explicit numeric class IDs first.  This avoids a common failure mode
    # where an exported model has stale or misleading class-name strings while the
    # class IDs are still correct.  For the user's barrel model the intended map is
    # class 0 -> red, class 1 -> yellow.
    try:
        rid = {int(x) for x in red_class_ids}
        yid = {int(x) for x in yellow_class_ids}
    except Exception:
        rid, yid = set(), set()
    if int(cls_id) in rid:
        return "red"
    if int(cls_id) in yid:
        return "yellow"

    # Fall back to semantic names when the explicit ID map does not cover the class.
    if norm in red or "red" in norm:
        return "red"
    if norm in yellow or "yellow" in norm:
        return "yellow"

    # Last-resort numeric class-name handling.
    if norm.isdigit():
        nid = int(norm)
        if nid in rid:
            return "red"
        if nid in yid:
            return "yellow"
    return None


class BarrelCounter:
    def __init__(
        self,
        model_path: str,
        K: np.ndarray,
        conf_threshold: float,
        merge_radius_m: float,
        red_classes: Sequence[str],
        yellow_classes: Sequence[str],
        output_dir: Path,
        save_annotated: bool = True,
        iou_threshold: float = 0.45,
        max_det: int = 50,
        min_track_hits: int = 2,
        default_depth_m: float = 4.0,
        allow_default_depth: bool = False,
        red_class_ids: Sequence[int] = (0,),
        yellow_class_ids: Sequence[int] = (1,),
        duplicate_suppression_radius_m: float = 1.10,
        confirmed_merge_radius_m: float = 0.45,
        allow_bbox_depth: bool = True,
        barrel_physical_height_m: float = 1.05,
        colour_override: bool = True,
        colour_override_min_pixels: int = 80,
        colour_override_ratio: float = 1.20,
        yolo_imgsz: int = 512,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.model = YOLO(model_path)
        self.K = K
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.merge_radius_m = float(merge_radius_m)
        self.red_classes = red_classes
        self.yellow_classes = yellow_classes
        self.red_class_ids = [int(x) for x in red_class_ids]
        self.yellow_class_ids = [int(x) for x in yellow_class_ids]
        self.min_track_hits = int(min_track_hits)
        self.default_depth_m = float(default_depth_m)
        self.allow_default_depth = bool(allow_default_depth)
        self.duplicate_suppression_radius_m = float(duplicate_suppression_radius_m)
        self.confirmed_merge_radius_m = float(confirmed_merge_radius_m)
        self.allow_bbox_depth = bool(allow_bbox_depth)
        self.barrel_physical_height_m = float(barrel_physical_height_m)
        self.colour_override = bool(colour_override)
        self.colour_override_min_pixels = int(colour_override_min_pixels)
        self.colour_override_ratio = float(colour_override_ratio)
        self.yolo_imgsz = int(yolo_imgsz)
        self.tracks: List[BarrelTrack] = []
        self.next_track_id = 1
        self.raw_seen = {"red": 0, "yellow": 0}
        self.annotated_dir = output_dir / "annotated"
        self.crops_dir = output_dir / "detections"
        self.save_annotated = save_annotated
        if save_annotated:
            self.annotated_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        print(f"[YOLO] Loaded {model_path}")
        print(f"[YOLO] Classes: {self.model.names}")
        with (output_dir / "yolo_model_info.json").open("w") as f:
            json.dump(
                {
                    "model_path": self.model_path,
                    "classes": self.model.names,
                    "red_classes": list(self.red_classes),
                    "yellow_classes": list(self.yellow_classes),
                    "red_class_ids": list(self.red_class_ids),
                    "yellow_class_ids": list(self.yellow_class_ids),
                    "conf": self.conf_threshold,
                    "iou": self.iou_threshold,
                    "merge_radius_m": self.merge_radius_m,
                    "duplicate_suppression_radius_m": self.duplicate_suppression_radius_m,
                    "confirmed_merge_radius_m": self.confirmed_merge_radius_m,
                    "min_track_hits": self.min_track_hits,
                    "allow_bbox_depth": self.allow_bbox_depth,
                    "barrel_physical_height_m": self.barrel_physical_height_m,
                    "colour_override": self.colour_override,
                    "colour_override_min_pixels": self.colour_override_min_pixels,
                    "colour_override_ratio": self.colour_override_ratio,
                    "yolo_imgsz": self.yolo_imgsz,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    def process(self, frame_bgr: np.ndarray, depth: Optional[np.ndarray], pose: Dict[str, float], frame_id: str) -> List[Dict[str, Any]]:
        # Use predict() rather than __call__ so iou / max_det are explicit and repeatable.
        results = self.model.predict(
            source=frame_bgr,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            imgsz=self.yolo_imgsz,
        )
        accepted: List[Dict[str, Any]] = []
        assigned_track_ids: set[int] = set()
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for det_index, box in enumerate(boxes):
                cls_id = int(box.cls[0].cpu().item())
                class_name = str(self.model.names.get(cls_id, cls_id))
                group = class_group(
                    class_name,
                    cls_id,
                    self.red_classes,
                    self.yellow_classes,
                    self.red_class_ids,
                    self.yellow_class_ids,
                )
                if group is None:
                    continue
                conf = float(box.conf[0].cpu().item())
                bbox = [float(x) for x in box.xyxy[0].cpu().numpy().tolist()]

                colour_group, colour_info = crop_colour_group(
                    frame_bgr,
                    bbox,
                    min_pixels=self.colour_override_min_pixels,
                    dominance_ratio=self.colour_override_ratio,
                )
                if self.colour_override and colour_group is not None and colour_group != group:
                    print(f"[YOLO] colour override class={class_name}/{cls_id} {group}->{colour_group} {colour_info}")
                    group = colour_group

                # Save the crop immediately after classification, even if depth is
                # missing.  Previously, no-depth yellow detections appeared only in
                # annotated/ and never in detections/ or barrel_tracks.csv.
                crop_saved = False
                if self.save_annotated:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    h, w = frame_bgr.shape[:2]
                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame_bgr[y1:y2, x1:x2]
                        cv2.imwrite(str(self.crops_dir / f"{frame_id}_{det_index}_{group}_{conf:.2f}_raw.jpg"), crop)
                        crop_saved = True

                d = get_depth_at_bbox(depth, bbox)
                depth_source = "depth"
                if d is None and self.allow_bbox_depth:
                    d = estimate_depth_from_bbox_height(
                        bbox,
                        self.K,
                        physical_height_m=self.barrel_physical_height_m,
                    )
                    if d is not None:
                        depth_source = "bbox_height"
                if d is None and self.allow_default_depth:
                    d = self.default_depth_m
                    depth_source = "default"
                if d is None:
                    # A no-depth detection is still logged as raw evidence, but not counted
                    # as a deduplicated barrel because we cannot place it globally.
                    self.raw_seen[group] = self.raw_seen.get(group, 0) + 1
                    accepted.append(
                        {
                            "group": group,
                            "class_name": class_name,
                            "class_id": cls_id,
                            "confidence": conf,
                            "colour_info": colour_info,
                            "depth_m": -1.0,
                            "north": float("nan"),
                            "east": float("nan"),
                            "bbox": bbox,
                            "depth_source": "missing",
                            "counted": False,
                        }
                    )
                    continue
                n, e = detection_to_global_ned(bbox, d, pose, self.K)
                track, is_new, crossed_count_threshold, match_dist = self._merge_or_add(
                    group, n, e, conf, assigned_track_ids
                )
                assigned_track_ids.add(track.track_id)
                self._consolidate_tracks()
                self.raw_seen[group] = self.raw_seen.get(group, 0) + 1
                accepted.append(
                    {
                        "group": group,
                        "class_name": class_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "colour_info": colour_info,
                        "depth_m": d,
                        "north": n,
                        "east": e,
                        "bbox": bbox,
                        "depth_source": depth_source,
                        "track_id": track.track_id,
                        "is_new_track": is_new,
                        "counted": crossed_count_threshold,
                        "duplicate_of_existing": not is_new,
                        "match_distance_m": match_dist,
                    }
                )
                if self.save_annotated and not crop_saved:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    h, w = frame_bgr.shape[:2]
                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame_bgr[y1:y2, x1:x2]
                        cv2.imwrite(str(self.crops_dir / f"{frame_id}_{det_index}_{group}_{conf:.2f}_{depth_source}.jpg"), crop)
            if self.save_annotated and accepted:
                cv2.imwrite(str(self.annotated_dir / f"{frame_id}.jpg"), result.plot())
        return accepted

    def _new_track(self, group: str, north: float, east: float, confidence: float) -> BarrelTrack:
        tr = BarrelTrack(
            group=group,
            north=north,
            east=east,
            confidence=confidence,
            track_id=self.next_track_id,
        )
        self.next_track_id += 1
        self.tracks.append(tr)
        return tr

    def _merge_or_add(
        self,
        group: str,
        north: float,
        east: float,
        confidence: float,
        assigned_track_ids: Optional[set[int]] = None,
    ) -> Tuple[BarrelTrack, bool, bool, float]:
        """Merge this detection into an existing physical-barrel track or add one.

        The important anti-double-counting rule is:
        - repeated sightings near an existing track update that track only;
        - a track can only consume one detection per frame, so two adjacent same-colour
          barrels in the same image can still become two separate tracks;
        - weak one-off tracks near a confirmed track are later consolidated away.
        """
        assigned_track_ids = assigned_track_ids or set()
        best: Optional[BarrelTrack] = None
        best_dist = float("inf")
        for tr in self.tracks:
            if tr.group != group or tr.track_id in assigned_track_ids:
                continue
            d = math.hypot(tr.north - north, tr.east - east)
            if d < best_dist:
                best_dist = d
                best = tr

        # Confirmed tracks get a larger duplicate suppression radius because their
        # global position is already reliable; pending tracks use the normal merge radius.
        if best is not None:
            allowed = self.merge_radius_m
            if best.hits >= self.min_track_hits:
                allowed = max(allowed, self.duplicate_suppression_radius_m)
            if best_dist <= allowed:
                was_counted = best.hits >= self.min_track_hits
                best.update(north, east, confidence)
                now_counted = best.hits >= self.min_track_hits
                return best, False, (not was_counted and now_counted), best_dist

        tr = self._new_track(group, north, east, confidence)
        return tr, True, tr.hits >= self.min_track_hits, float("inf")

    def _consolidate_tracks(self) -> None:
        """Merge obvious duplicate tracks from noisy depth/yaw projection.

        We avoid blindly merging all close tracks because adjacent barrels can be
        physically close.  The rule below mainly removes *weak* duplicate tracks
        that appear near an already confirmed track.  Two confirmed tracks are only
        merged if they are extremely close.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(self.tracks)):
                if changed:
                    break
                a = self.tracks[i]
                for j in range(i + 1, len(self.tracks)):
                    b = self.tracks[j]
                    if a.group != b.group:
                        continue
                    d = math.hypot(a.north - b.north, a.east - b.east)
                    a_confirmed = a.hits >= self.min_track_hits
                    b_confirmed = b.hits >= self.min_track_hits
                    should_merge = False
                    if d <= self.confirmed_merge_radius_m:
                        should_merge = True
                    elif d <= self.duplicate_suppression_radius_m and (a_confirmed != b_confirmed):
                        should_merge = True
                    elif d <= self.merge_radius_m and (a.hits <= 1 or b.hits <= 1):
                        should_merge = True
                    if not should_merge:
                        continue

                    # Keep the stronger/older track ID.
                    keep, drop = (a, b)
                    if (b.hits, b.confidence) > (a.hits, a.confidence):
                        keep, drop = (b, a)
                    total_hits = max(1, keep.hits + drop.hits)
                    keep.north = (keep.north * keep.hits + drop.north * drop.hits) / total_hits
                    keep.east = (keep.east * keep.hits + drop.east * drop.hits) / total_hits
                    keep.confidence = max(keep.confidence, drop.confidence)
                    keep.hits = total_hits
                    keep.first_seen_ms = min(keep.first_seen_ms, drop.first_seen_ms)
                    keep.last_seen_ms = max(keep.last_seen_ms, drop.last_seen_ms)
                    self.tracks.remove(drop)
                    changed = True
                    break

    def counts(self) -> Dict[str, int]:
        self._consolidate_tracks()
        red = sum(1 for t in self.tracks if t.group == "red" and t.hits >= self.min_track_hits)
        yellow = sum(1 for t in self.tracks if t.group == "yellow" and t.hits >= self.min_track_hits)
        return {"red": red, "yellow": yellow, "total": red + yellow}

    def raw_counts(self) -> Dict[str, int]:
        red = int(self.raw_seen.get("red", 0))
        yellow = int(self.raw_seen.get("yellow", 0))
        return {"raw_red_detections": red, "raw_yellow_detections": yellow, "raw_total_detections": red + yellow}

    def save_tracks(self, output_dir: Path) -> Path:
        self._consolidate_tracks()
        path = output_dir / "barrel_tracks.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "group", "counted", "north", "east", "confidence", "hits", "first_seen_ms", "last_seen_ms"])
            for t in sorted(self.tracks, key=lambda x: (x.group, x.track_id)):
                counted = int(t.hits >= self.min_track_hits)
                w.writerow([t.track_id, t.group, counted, t.north, t.east, t.confidence, t.hits, t.first_seen_ms, t.last_seen_ms])
        return path


def _candidate_model_paths(base: Path) -> List[Path]:
    """Return likely YOLO weights in preference order."""
    candidates = [
        base / "train" / "weights" / "best.pt",
        base / "weights" / "best.pt",
        base / "best.pt",
        base / "my_model.pt",
        base / "train" / "weights" / "last.pt",
        base / "last.pt",
        base / "train" / "weights" / "best.onnx",
        base / "weights" / "best.onnx",
        base / "best.onnx",
    ]
    # Also include recursive matches, with .pt preferred over .onnx.
    recursive = sorted(base.rglob("*.pt")) + sorted(base.rglob("*.onnx"))
    out: List[Path] = []
    seen: Set[Path] = set()
    for c in candidates + recursive:
        try:
            r = c.resolve()
        except Exception:
            r = c
        if r not in seen and c.exists() and c.is_file():
            out.append(c)
            seen.add(r)
    return out


def resolve_yolo_model_path(args: argparse.Namespace, output_dir: Path) -> str:
    """Resolve --model, --model-zip, or common local my_model paths to a usable file."""
    if getattr(args, "model", ""):
        model = Path(args.model).expanduser()
        if model.suffix.lower() == ".zip":
            args.model_zip = str(model)
        elif model.exists():
            return str(model.resolve())
        else:
            raise FileNotFoundError(f"YOLO model path does not exist: {model}")

    if getattr(args, "model_zip", ""):
        zpath = Path(args.model_zip).expanduser().resolve()
        if not zpath.exists():
            raise FileNotFoundError(f"YOLO model zip does not exist: {zpath}")
        extract_dir = Path(args.model_extract_dir).expanduser().resolve() if getattr(args, "model_extract_dir", "") else output_dir / "model_extracted"
        marker = extract_dir / ".extracted_from_zip"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"[YOLO] Extracting {zpath} -> {extract_dir}")
            with zipfile.ZipFile(str(zpath), "r") as zf:
                zf.extractall(str(extract_dir))
            marker.write_text(str(zpath))
        candidates = _candidate_model_paths(extract_dir)
        if candidates:
            print(f"[YOLO] Auto-selected weights: {candidates[0]}")
            return str(candidates[0].resolve())
        raise FileNotFoundError(f"No .pt or .onnx YOLO weights found inside {zpath}")

    search_roots = []
    if getattr(args, "model_dir", ""):
        search_roots.append(Path(args.model_dir).expanduser())
    cwd = Path.cwd()
    search_roots += [cwd, cwd / "my_model", cwd / "train", cwd / "runs" / "detect" / "train"]
    for root in search_roots:
        if root.exists():
            candidates = _candidate_model_paths(root)
            if candidates:
                print(f"[YOLO] Auto-selected weights: {candidates[0]}")
                return str(candidates[0].resolve())
    raise FileNotFoundError(
        "Count mode needs a trained YOLO model. Use --model path/to/best.pt, "
        "or --model-zip my_model.zip, or place train/weights/best.pt under the current folder."
    )


# -----------------------------------------------------------------------------
# Mission controller
# -----------------------------------------------------------------------------
class QualifierMission:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.K = K_DEFAULT.copy()

        self.drone = Drone()
        self.state = SharedState()
        self.stop_event = asyncio.Event()
        self.monitor_task: Optional[asyncio.Task] = None

        self.rgb = RGBReceiver(args.rgb_topic)
        self.depth = DepthReceiver(args.depth_topic)
        self.mapper = OccupancyGridMapper(
            self.K,
            resolution_m=args.grid_resolution_m,
            ray_max_m=args.map_ray_max_m,
            safety_radius_m=args.map_safety_radius_m,
            sample_cols=args.map_sample_cols,
            occ_threshold=args.map_occ_threshold,
            occ_update=args.map_occ_update,
        )
        self.mapper.path_stride_m = float(getattr(args, "path_stride_m", 1.20))
        # v30: optional geometry regularisation layer. This converts noisy raw
        # occupied cells into line/rectangle primitives before planning/debug save.
        for _name in (
            "map_regularize", "map_regularize_use_for_planning", "map_regularize_close_radius_m",
            "map_regularize_open_radius_m", "map_regularize_min_component_cells", "map_regularize_min_line_m",
            "map_regularize_line_gap_m", "map_regularize_wall_thickness_m", "map_regularize_hough_threshold",
            "map_regularize_manhattan", "map_regularize_min_rect_side_m", "map_regularize_max_rect_side_m",
            "map_regularize_rectangles", "map_regularize_rect_min_fill_ratio", "map_regularize_rect_max_aspect",
            "map_regularize_rect_thickness_m", "map_regularize_keep_strong_raw", "map_regularize_strong_raw_threshold",
            "map_regularize_min_output_cells", "map_regularize_clean_display", "no_diagonal_corner_cutting", "path_turn_padding_cells",
            "frontier_unknown_radius_m", "frontier_unknown_lookahead_m", "frontier_unknown_width_m",
            "frontier_unknown_weight", "frontier_unknown_ahead_weight", "frontier_progress_weight", "frontier_path_cost_weight", "frontier_local_loop_penalty", "frontier_local_loop_path_cells",
            "topological_gateway_mode", "topo_frontier_min_unknown_ahead", "topo_frontier_min_unknown_near",
            "topo_frontier_gateway_bonus", "topo_frontier_depth_bonus", "topo_frontier_perimeter_penalty",
            "map_prune_diagonal_ghosts", "map_diagonal_prune_min_axis_run_cells",
            "map_diagonal_prune_max_component_cells", "map_diagonal_prune_apply_decay", "map_diagonal_prune_decay",
            "self_clear_enabled", "self_clear_radius_m", "self_clear_free_log"
        ):
            setattr(self.mapper, _name, getattr(args, _name))
        self.planner = FrontierPlanner(self.mapper)
        self.safety = DepthSafetyFilter(self.K, args)
        self.logger = FrameLogger(self.output_dir)

        self.counter: Optional[BarrelCounter] = None
        if args.mode == "count":
            resolved_model = resolve_yolo_model_path(args, self.output_dir)
            self.counter = BarrelCounter(
                model_path=resolved_model,
                K=self.K,
                conf_threshold=args.conf,
                merge_radius_m=args.merge_radius_m,
                red_classes=args.red_classes,
                yellow_classes=args.yellow_classes,
                output_dir=self.output_dir,
                save_annotated=not args.no_annotated,
                iou_threshold=args.iou,
                max_det=args.max_det,
                min_track_hits=args.min_track_hits,
                default_depth_m=args.default_detection_depth_m,
                allow_default_depth=args.count_without_depth,
                red_class_ids=args.red_class_ids,
                yellow_class_ids=args.yellow_class_ids,
                duplicate_suppression_radius_m=args.duplicate_suppression_radius_m,
                confirmed_merge_radius_m=args.confirmed_merge_radius_m,
                allow_bbox_depth=args.count_with_bbox_depth,
                barrel_physical_height_m=args.barrel_physical_height_m,
                colour_override=args.colour_override,
                colour_override_min_pixels=args.colour_override_min_pixels,
                colour_override_ratio=args.colour_override_ratio,
                yolo_imgsz=args.yolo_imgsz,
            )

        self.current_path: List[Tuple[float, float]] = []
        self.last_plan_ms = 0
        self.last_photo_ms = 0
        self.last_detection_ms = 0
        self.last_saved_pose: Optional[Dict[str, float]] = None
        self.last_interest_sweep_ms = 0
        self.last_scan_ms = 0
        self.last_map_save_ms = 0
        self.start_pose: Optional[Dict[str, float]] = None

        # Stuck / recovery state. A frontier that repeatedly causes a blocked first
        # segment is temporarily blacklisted so replanning does not choose it again.
        self.rejected_frontiers: Dict[Cell, int] = {}
        self.segment_block_count = 0
        self.no_path_count = 0
        self.avoid_block_count = 0
        self.last_recovery_ms = 0

        # v24 local centreline exploration state.  This is separate from the global
        # frontier path: it is used when the planner has a nominal path but the
        # reactive corridor gate keeps refusing to translate.
        self.local_corridor_yaw: Optional[float] = None
        self.local_corridor_until_ms = 0
        self.local_soft_block_count = 0
        self.local_corridor_candidates: List[Dict[str, float]] = []

        # v29 express corridor mode.  When the depth camera sees a straight path
        # with a left and right boundary, stop treating the route as many tiny A*
        # waypoints.  Lock onto the corridor centreline, record the corridor entry,
        # and drive it at a higher speed until the corridor ends, an obstacle blocks
        # it, or the maximum commit distance/time expires.
        self.express_corridor_yaw: Optional[float] = None
        self.express_corridor_until_ms = 0
        self.express_corridor_start_pose: Optional[Dict[str, float]] = None
        self.express_corridor_start_ms = 0
        self.express_corridor_block_count = 0

        # v26 front-block / vertical recovery state.  A true front obstacle should
        # trigger bypass planning, not an infinite brake loop.
        self.front_block_count = 0
        self.last_vertical_escape_ms = 0
        self.vertical_escape_count = 0

        # v32 dead-end / breadcrumb escape state.  In a dead-end corridor the old
        # behaviour could spend too long trying side scans or vertical escape.
        # Keep a lightweight breadcrumb trail so a confirmed dead-end can be
        # exited by reversing the travelled centreline back to the last useful
        # open area, then resume fast exploration.
        self.travel_history: List[Dict[str, float]] = []
        self.last_history_ms = 0
        self.backtrack_targets: List[Tuple[float, float]] = []
        self.backtrack_until_ms = 0
        self.last_dead_end_escape_ms = 0
        self.dead_end_escape_count = 0
        # v42: remember confirmed dead-end directions so the drone does not
        # immediately choose the same dead-end after backtracking out.
        self.dead_end_blocked_cells: Dict[Cell, int] = {}

        # v38 room/zone escape.  If the drone has been driving a loop inside the
        # same zone, force a gateway-style scan that scores unknown space highly.
        self.last_zone_escape_ms = 0

        # v39 online topological gateway exploration state.  We do not know the
        # global map in advance, so the drone records local junction/gateway
        # decisions online.  The policy prefers unvisited gateways that lead to
        # unknown space and treats ordinary frontier planning as a fallback.
        self.gateway_attempts: Dict[Tuple[int, int, int], int] = {}
        self.gateway_success: Dict[Tuple[int, int, int], int] = {}
        self.gateway_blacklist_until_ms: Dict[Tuple[int, int, int], int] = {}
        self.current_topo_node: Optional[Tuple[int, int]] = None
        self.last_gateway_scan_ms = 0
        self.last_gateway_selected_key: Optional[Tuple[int, int, int]] = None
        self.topo_loop_score = 0

        # v41 progress commitment / v42 corner cleanup.  v39/v41 could select a gateway, move only a few
        # centimetres, then immediately re-scan/select a different gateway.  That
        # created decision oscillation at the start area.  When a gateway/centreline
        # is selected, commit to it for a small distance/time unless a real safety
        # emergency occurs.
        self.progress_commit_until_ms = 0
        self.progress_commit_start_pose: Optional[Dict[str, float]] = None
        self.progress_commit_min_m = 0.0
        self.progress_commit_yaw: Optional[float] = None
        self.progress_commit_reason = ""

        self.last_waypoint_pose: Optional[Dict[str, float]] = None
        self.waypoint_csv_path = self.output_dir / "waypoints.csv"
        self.waypoint_csv_file = self.waypoint_csv_path.open("w", newline="")
        self.waypoint_writer = csv.DictWriter(
            self.waypoint_csv_file,
            fieldnames=["timestamp_ms", "north", "east", "down", "yaw_deg", "event", "target_yaw_deg", "clearance_center_m", "clearance_left_m", "clearance_right_m"],
        )
        self.waypoint_writer.writeheader()

        self.gateway_csv_path = self.output_dir / "gateway_graph.csv"
        self.gateway_csv_file = self.gateway_csv_path.open("w", newline="")
        self.gateway_writer = csv.DictWriter(
            self.gateway_csv_file,
            fieldnames=["timestamp_ms", "event", "node_i", "node_j", "gateway_i", "gateway_j", "yaw_deg", "unknown", "clearance_center_m", "clearance_left_m", "clearance_right_m", "score", "attempts"],
        )
        self.gateway_writer.writeheader()

    def pose(self) -> Optional[Dict[str, float]]:
        if self.state.latest_position is None or self.state.latest_yaw is None:
            return None
        p = self.state.latest_position
        return {
            "north": float(p.north_m),
            "east": float(p.east_m),
            "down": float(p.down_m),
            "yaw_deg": float(self.state.latest_yaw),
        }

    async def wait_for_ready_data(self, timeout_s: float = 20.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self.pose() is not None and self.rgb.get_frame() is not None and self.depth.get_frame() is not None:
                return
            await asyncio.sleep(0.1)
        print(
            "[WARN] Sensor wait timed out: "
            f"pose={self.pose() is not None}, rgb={self.rgb.get_frame() is not None}, depth={self.depth.get_frame() is not None}"
        )

    def clearances(self, depth_frame: Optional[np.ndarray]) -> Tuple[float, float, float]:
        if depth_frame is None:
            return self.args.map_ray_max_m, self.args.map_ray_max_m, self.args.map_ray_max_m
        h, w = depth_frame.shape[:2]
        band = depth_frame[int(0.30 * h) : int(0.70 * h), :]
        left = safe_percentile(band[:, : w // 3], 15, self.args.map_ray_max_m)
        center = safe_percentile(band[:, w // 3 : 2 * w // 3], 15, self.args.map_ray_max_m)
        right = safe_percentile(band[:, 2 * w // 3 :], 15, self.args.map_ray_max_m)
        return left, center, right

    def topo_node_key(self, pose: Dict[str, float]) -> Tuple[int, int]:
        scale = max(1, int(round(float(getattr(self.args, "topo_node_radius_m", 3.0)) / self.mapper.res)))
        ci, cj = self.mapper.world_to_cell(pose["north"], pose["east"])
        return (int(round(ci / scale)), int(round(cj / scale)))

    def gateway_key(self, pose: Dict[str, float], yaw_deg: float, distance_m: Optional[float] = None) -> Tuple[int, int, int]:
        if distance_m is None:
            distance_m = float(getattr(self.args, "topo_gateway_key_distance_m", 4.0))
        yaw_rad = math.radians(float(yaw_deg))
        n = float(pose["north"]) + float(distance_m) * math.cos(yaw_rad)
        e = float(pose["east"]) + float(distance_m) * math.sin(yaw_rad)
        ci, cj = self.mapper.world_to_cell(n, e)
        # 30-degree bins are deliberate: the graph should remember a branch, not a
        # noisy exact yaw.
        yaw_bin = int(round(wrap_deg(float(yaw_deg)) / float(getattr(self.args, "topo_gateway_yaw_bin_deg", 30.0))))
        return (ci, cj, yaw_bin)

    def gateway_blacklisted(self, key: Tuple[int, int, int]) -> bool:
        until = self.gateway_blacklist_until_ms.get(key, 0)
        if until <= now_ms():
            self.gateway_blacklist_until_ms.pop(key, None)
            return False
        return True

    def record_gateway_event(self, event: str, pose: Optional[Dict[str, float]], yaw_deg: float, unknown: float = 0.0, clearances: Optional[Tuple[float, float, float]] = None, score: float = 0.0, key: Optional[Tuple[int, int, int]] = None) -> None:
        if pose is None:
            return
        if key is None:
            key = self.gateway_key(pose, yaw_deg)
        node = self.topo_node_key(pose)
        l, c, r = clearances if clearances is not None else (float("nan"), float("nan"), float("nan"))
        try:
            self.gateway_writer.writerow({
                "timestamp_ms": now_ms(),
                "event": event,
                "node_i": node[0],
                "node_j": node[1],
                "gateway_i": key[0],
                "gateway_j": key[1],
                "yaw_deg": f"{wrap_deg(yaw_deg):.1f}",
                "unknown": f"{float(unknown):.1f}",
                "clearance_center_m": f"{c:.2f}" if math.isfinite(c) else "",
                "clearance_left_m": f"{l:.2f}" if math.isfinite(l) else "",
                "clearance_right_m": f"{r:.2f}" if math.isfinite(r) else "",
                "score": f"{float(score):.2f}",
                "attempts": self.gateway_attempts.get(key, 0),
            })
            self.gateway_csv_file.flush()
        except Exception:
            pass

    def start_progress_commitment(self, pose: Optional[Dict[str, float]], yaw_deg: float, reason: str, duration_s: Optional[float] = None, min_dist_m: Optional[float] = None) -> None:
        """Temporarily suppress non-essential re-decisions after choosing a route.

        This is intentionally not a safety override: emergency braking, front-block
        bypass, and dead-end escape still run.  It only prevents gateway/zone-loop
        scanners from stealing control after the drone has moved a few centimetres.
        """
        if pose is None:
            return
        if duration_s is None:
            duration_s = float(getattr(self.args, "progress_commit_s", 7.0))
        if min_dist_m is None:
            min_dist_m = float(getattr(self.args, "progress_commit_min_m", 3.0))
        self.progress_commit_until_ms = now_ms() + int(max(0.1, float(duration_s)) * 1000)
        self.progress_commit_start_pose = pose.copy()
        self.progress_commit_min_m = max(0.0, float(min_dist_m))
        self.progress_commit_yaw = wrap_deg(float(yaw_deg))
        self.progress_commit_reason = str(reason)
        print(f"[COMMIT] route yaw={self.progress_commit_yaw:.0f} reason={reason} min_m={self.progress_commit_min_m:.1f} time_s={duration_s:.1f}")

    def progress_commit_active(self, pose: Optional[Dict[str, float]] = None) -> bool:
        """Return True while a selected route should be given a chance to make progress."""
        if not bool(getattr(self.args, "progress_commitment", True)):
            return False
        now = now_ms()
        if now >= self.progress_commit_until_ms:
            return False
        if pose is None:
            pose = self.pose()
        if pose is None or self.progress_commit_start_pose is None:
            return True
        travelled = math.hypot(
            float(pose["north"]) - float(self.progress_commit_start_pose["north"]),
            float(pose["east"]) - float(self.progress_commit_start_pose["east"]),
        )
        # Release the lock once either enough distance or enough time has passed.
        if travelled >= float(self.progress_commit_min_m):
            return False
        return True

    async def scan_topological_gateways(self, reason: str = "topo_gateway") -> Optional[float]:
        """Scan a small set of branch headings and choose an unvisited gateway.

        This is the v39 high-level route policy.  It is intentionally different
        from generic frontier exploration: it chooses doorway/corridor openings
        that lead to unknown space, and it penalizes directions that overlap the
        recent path unless this is an explicit dead-end/backtrack recovery.
        """
        pose = self.pose()
        if pose is None:
            return None
        base = float(pose["yaw_deg"])
        step = float(getattr(self.args, "topo_scan_step_deg", 45.0))
        if "front_block" in reason or "corner" in reason:
            deltas = [90.0, -90.0, 45.0, -45.0, 135.0, -135.0, 0.0]
            max_headings = int(getattr(self.args, "topo_front_block_max_headings", 7))
        elif "zone" in reason or "loop" in reason or "gateway" in reason:
            deltas = [0.0, 90.0, -90.0, 45.0, -45.0, 135.0, -135.0, 180.0]
            max_headings = int(getattr(self.args, "topo_gateway_scan_max_headings", 8))
        else:
            # Normal scans check forward and both orthogonal exits; this is enough
            # to find most doorways without wasting 3-4 minutes spinning.
            deltas = [0.0, 90.0, -90.0, 45.0, -45.0, 135.0, -135.0]
            max_headings = int(getattr(self.args, "topo_normal_scan_max_headings", 5))
        headings: List[float] = []
        seen: Set[int] = set()
        for dlt in deltas:
            y = wrap_deg(base + dlt)
            k = int(round(y / 5.0))
            if k not in seen:
                seen.add(k)
                headings.append(y)
        if max_headings > 0:
            headings = headings[:max_headings]

        candidates: List[Dict[str, float]] = []
        print(f"[TOPO] {reason}; scanning {len(headings)} gateway headings from yaw={base:.0f}")
        for y in headings:
            p_snap, d_snap = await self.mapping_yaw_snapshot(
                y,
                timeout_s=float(getattr(self.args, "topo_scan_yaw_timeout_s", self.args.recovery_yaw_timeout_s)),
                tolerance_deg=float(getattr(self.args, "topo_scan_yaw_tolerance_deg", self.args.recovery_yaw_tolerance_deg)),
            )
            if p_snap is not None:
                self.mapper.update_from_depth(d_snap, p_snap)
            left, center, right = self.clearances(d_snap)
            usable_dist = min(float(center), float(getattr(self.args, "topo_unknown_lookahead_m", 8.5)))
            unknown = self._unknown_gain_ahead(
                pose,
                y,
                max(usable_dist, float(getattr(self.args, "topo_unknown_lookahead_m", 8.5))),
                float(getattr(self.args, "topo_unknown_width_m", 2.8)),
            )
            side_balance = min(float(left), float(right))
            visited = self._visited_penalty_ahead(pose, y, max(usable_dist, 2.0))
            key = self.gateway_key(pose, y)
            attempts = self.gateway_attempts.get(key, 0)
            blacklisted = self.gateway_blacklisted(key)
            recent_block = False
            if bool(getattr(self.args, "avoid_recent_path", True)) and not ("dead_end" in reason or "backtrack" in reason):
                recent_block = self.heading_points_into_recent_path(pose, y, min(max(usable_dist, 2.0), float(getattr(self.args, "recent_heading_lookahead_m", 4.0))))
            viable = (
                center >= float(getattr(self.args, "topo_gateway_min_center_m", 2.6))
                and side_balance >= float(getattr(self.args, "topo_gateway_min_side_m", 0.55))
                and unknown >= int(getattr(self.args, "topo_gateway_min_unknown_cells", 10))
                and not blacklisted
            )
            if recent_block and unknown < int(getattr(self.args, "topo_recent_override_unknown_cells", 35)):
                viable = False
            turn_cost = abs(yaw_error_deg(y, base)) / 90.0
            side_turn_bonus = 0.0
            if "front_block" in reason or "corner" in reason:
                if 60.0 <= abs(yaw_error_deg(y, base)) <= 125.0:
                    side_turn_bonus = float(getattr(self.args, "topo_side_branch_bonus", 8.0))
            score = (
                float(getattr(self.args, "topo_unknown_weight", 0.22)) * math.sqrt(max(0.0, float(unknown)))
                + float(getattr(self.args, "topo_clearance_weight", 0.85)) * min(center, 10.0)
                + float(getattr(self.args, "topo_side_balance_weight", 0.50)) * min(side_balance, 5.0)
                + side_turn_bonus
                - float(getattr(self.args, "topo_visited_weight", 2.6)) * visited
                - float(getattr(self.args, "topo_attempt_penalty", 5.0)) * attempts
                - float(getattr(self.args, "topo_turn_weight", 0.45)) * turn_cost
            )
            if recent_block:
                score -= float(getattr(self.args, "topo_recent_penalty", 10.0))
            if blacklisted:
                score -= 20.0
            if not viable:
                score -= 12.0
            candidates.append({"yaw": y, "left": left, "center": center, "right": right, "unknown": float(unknown), "visited": visited, "score": score, "viable": 1.0 if viable else 0.0, "key": key})
            print(f"[TOPO] yaw={y:.0f} L={left:.2f} C={center:.2f} R={right:.2f} unknown={unknown} visited={visited:.1f} attempts={attempts} recent={recent_block} viable={viable} score={score:.2f}")
            self.record_gateway_event("scan_candidate", pose, y, unknown, (left, center, right), score, key)
            await self.maybe_save_or_detect(note="topo_gateway_scan")

        viable_candidates = [c for c in candidates if c["viable"] > 0.5]
        if not viable_candidates:
            print("[TOPO] no viable unvisited gateway; falling back to legacy centreline/frontier")
            return None
        best = max(viable_candidates, key=lambda c: c["score"])
        key = best["key"]
        self.gateway_attempts[key] = self.gateway_attempts.get(key, 0) + 1
        self.last_gateway_selected_key = key
        self.local_corridor_yaw = wrap_deg(float(best["yaw"]))
        self.local_corridor_until_ms = now_ms() + int(float(getattr(self.args, "topo_gateway_commit_s", 12.0)) * 1000)
        self.current_path = []
        self.start_progress_commitment(pose, self.local_corridor_yaw, f"topo:{reason}", duration_s=float(getattr(self.args, "topo_gateway_commit_s", 12.0)), min_dist_m=float(getattr(self.args, "progress_commit_min_m", 3.0)))
        print(f"[TOPO] selected gateway yaw={self.local_corridor_yaw:.0f} unknown={best['unknown']:.0f} C={best['center']:.2f} score={best['score']:.2f}")
        self.record_gateway_event("selected", pose, self.local_corridor_yaw, best["unknown"], (best["left"], best["center"], best["right"]), best["score"], key)
        self.record_waypoint(pose, event=f"topo_gateway_selected:{reason}", target_yaw=self.local_corridor_yaw, clearances=(best["left"], best["center"], best["right"]), force=True)
        if bool(getattr(self.args, "auto_express_after_scan", True)) and best["center"] >= float(getattr(self.args, "express_corridor_min_center_m", 4.2)):
            self.start_express_corridor(pose, self.local_corridor_yaw, reason=f"topo:{reason}")
        await self.yaw_to_mapping_heading(
            self.local_corridor_yaw,
            timeout_s=float(getattr(self.args, "topo_scan_yaw_timeout_s", self.args.recovery_yaw_timeout_s)),
            tolerance_deg=float(getattr(self.args, "topo_scan_yaw_tolerance_deg", self.args.recovery_yaw_tolerance_deg)),
        )
        return self.local_corridor_yaw

    async def stop_motion(self, seconds: float = 0.05) -> None:
        pose = self.pose()
        yaw = 0.0 if pose is None else pose["yaw_deg"]
        await self.drone.send_velocity(0.0, 0.0, 0.0, yaw)
        await asyncio.sleep(seconds)

    def autosave_map_if_due(self, pose: Optional[Dict[str, float]]) -> None:
        interval_ms = int(float(self.args.map_save_interval_s) * 1000)
        if interval_ms <= 0:
            return
        if now_ms() - self.last_map_save_ms >= interval_ms:
            path = self.mapper.save_debug_map(self.output_dir, pose)
            self.last_map_save_ms = now_ms()
            if path:
                print(f"[MAP] autosaved {path}")

    def active_rejected_frontiers(self) -> Set[Cell]:
        """Return non-expired temporarily rejected frontier cells."""
        t = now_ms()
        self.rejected_frontiers = {c: expiry for c, expiry in self.rejected_frontiers.items() if expiry > t}
        return set(self.rejected_frontiers.keys())

    def reject_frontier_world(self, target_ne: Tuple[float, float], reason: str) -> None:
        cell = self.mapper.world_to_cell(target_ne[0], target_ne[1])
        ttl_ms = int(float(self.args.reject_frontier_ttl_s) * 1000)
        self.rejected_frontiers[cell] = now_ms() + ttl_ms
        print(f"[PLAN] temporarily rejected frontier cell={cell} reason={reason} ttl={self.args.reject_frontier_ttl_s:.0f}s")

    def record_waypoint(self, pose: Optional[Dict[str, float]], event: str, target_yaw: Optional[float] = None, clearances: Optional[Tuple[float, float, float]] = None, force: bool = False) -> None:
        """Record travelled waypoints / decision points for post-run review."""
        if pose is None:
            return
        if not force and self.last_waypoint_pose is not None:
            moved = math.hypot(pose["north"] - self.last_waypoint_pose["north"], pose["east"] - self.last_waypoint_pose["east"])
            if moved < float(getattr(self.args, "waypoint-record-every-m", getattr(self.args, "waypoint_record_every_m", 0.8))):
                return
        l = c = r = float("nan")
        if clearances is not None:
            l, c, r = clearances
        self.waypoint_writer.writerow({
            "timestamp_ms": now_ms(),
            "north": f"{pose['north']:.3f}",
            "east": f"{pose['east']:.3f}",
            "down": f"{pose['down']:.3f}",
            "yaw_deg": f"{pose['yaw_deg']:.2f}",
            "event": event,
            "target_yaw_deg": "" if target_yaw is None else f"{wrap_deg(target_yaw):.2f}",
            "clearance_center_m": f"{c:.3f}" if math.isfinite(c) else "",
            "clearance_left_m": f"{l:.3f}" if math.isfinite(l) else "",
            "clearance_right_m": f"{r:.3f}" if math.isfinite(r) else "",
        })
        self.waypoint_csv_file.flush()
        self.last_waypoint_pose = pose.copy()

    def update_travel_history(self, pose: Optional[Dict[str, float]]) -> None:
        """Record a sparse breadcrumb trail for fast dead-end escape."""
        if pose is None:
            return
        t = now_ms()
        min_gap_ms = int(float(getattr(self.args, "history_record_gap_s", 0.35)) * 1000)
        min_dist = float(getattr(self.args, "history_record_every_m", 0.55))
        if self.travel_history:
            last = self.travel_history[-1]
            moved = math.hypot(pose["north"] - last["north"], pose["east"] - last["east"])
            if moved < min_dist and t - self.last_history_ms < min_gap_ms:
                return
        self.travel_history.append({
            "north": float(pose["north"]),
            "east": float(pose["east"]),
            "down": float(pose.get("down", 0.0)),
            "yaw_deg": float(pose.get("yaw_deg", 0.0)),
            "timestamp_ms": t,
        })
        max_len = int(getattr(self.args, "history_max_points", 450))
        if len(self.travel_history) > max_len:
            self.travel_history = self.travel_history[-max_len:]
        self.last_history_ms = t


    def recent_path_cells(self, pose: Optional[Dict[str, float]] = None) -> Set[Cell]:
        """Cells near the recent travelled path that normal exploration should avoid reusing.

        This is not used during explicit dead-end breadcrumb escape.  It prevents the
        frontier planner from choosing the path the drone just came from when a ghost
        map obstacle appears ahead.
        """
        if not bool(getattr(self.args, "avoid_recent_path", True)):
            return set()
        if not self.travel_history:
            return set()
        lookback_m = float(getattr(self.args, "recent_path_lookback_m", 18.0))
        radius_m = float(getattr(self.args, "recent_path_radius_m", 0.9))
        exclude_current_m = float(getattr(self.args, "recent_path_exclude_current_m", 1.8))
        rad_cells = max(1, int(math.ceil(radius_m / self.mapper.res)))
        cells: Set[Cell] = set()
        total = 0.0
        last: Optional[Dict[str, float]] = None
        cur_n = None if pose is None else float(pose["north"])
        cur_e = None if pose is None else float(pose["east"])
        for h in reversed(self.travel_history[:-1]):
            if last is not None:
                total += math.hypot(float(last["north"]) - float(h["north"]), float(last["east"]) - float(h["east"]))
            last = h
            if total > lookback_m:
                break
            if cur_n is not None and math.hypot(float(h["north"]) - cur_n, float(h["east"]) - cur_e) < exclude_current_m:
                continue
            base = self.mapper.world_to_cell(float(h["north"]), float(h["east"]))
            for di in range(-rad_cells, rad_cells + 1):
                for dj in range(-rad_cells, rad_cells + 1):
                    if di * di + dj * dj <= rad_cells * rad_cells:
                        cells.add((base[0] + di, base[1] + dj))
        return cells

    def heading_points_into_recent_path(self, pose: Dict[str, float], yaw_deg: float, lookahead_m: Optional[float] = None) -> bool:
        """Return True if a candidate heading mainly points back into recent breadcrumbs."""
        if not bool(getattr(self.args, "avoid_recent_path", True)):
            return False
        lookahead = float(getattr(self.args, "recent_heading_lookahead_m", 4.0)) if lookahead_m is None else float(lookahead_m)
        recent = self.recent_path_cells(pose)
        if not recent:
            return False
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        samples = max(3, int(math.ceil(lookahead / max(0.4, self.mapper.res))))
        hits = 0
        for k in range(1, samples + 1):
            r = (k / samples) * lookahead
            cell = self.mapper.world_to_cell(pose["north"] + r * c, pose["east"] + r * s)
            if cell in recent:
                hits += 1
        return hits >= int(getattr(self.args, "recent_heading_max_hits", 2))

    def remember_dead_end_direction(self, pose: Dict[str, float], reason: str = "dead_end") -> None:
        """Temporarily blacklist cells just ahead of a confirmed dead end."""
        ttl_ms = int(float(getattr(self.args, "dead_end_memory_ttl_s", 55.0)) * 1000)
        until = now_ms() + ttl_ms
        yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
        c, s = math.cos(yaw), math.sin(yaw)
        dist_m = float(getattr(self.args, "dead_end_memory_distance_m", 4.2))
        radius_m = float(getattr(self.args, "dead_end_memory_radius_m", 1.0))
        rad_cells = max(1, int(math.ceil(radius_m / self.mapper.res)))
        d = max(0.8, self.mapper.res)
        marked = 0
        while d <= dist_m:
            base = self.mapper.world_to_cell(pose["north"] + d * c, pose["east"] + d * s)
            for di in range(-rad_cells, rad_cells + 1):
                for dj in range(-rad_cells, rad_cells + 1):
                    if di * di + dj * dj <= rad_cells * rad_cells:
                        self.dead_end_blocked_cells[(base[0] + di, base[1] + dj)] = until
                        marked += 1
            d += max(0.6, self.mapper.res)
        print(f"[DEAD_END] remembered blocked direction reason={reason} cells={marked} ttl_s={ttl_ms/1000:.0f}")

    def heading_points_to_dead_end_memory(self, pose: Dict[str, float], yaw_deg: float, lookahead_m: Optional[float] = None) -> bool:
        if not self.dead_end_blocked_cells:
            return False
        t = now_ms()
        for c, until in list(self.dead_end_blocked_cells.items()):
            if until <= t:
                self.dead_end_blocked_cells.pop(c, None)
        if not self.dead_end_blocked_cells:
            return False
        lookahead = float(getattr(self.args, "dead_end_memory_check_m", 4.5)) if lookahead_m is None else float(lookahead_m)
        yaw = math.radians(float(yaw_deg))
        c, s = math.cos(yaw), math.sin(yaw)
        samples = max(3, int(math.ceil(lookahead / max(0.4, self.mapper.res))))
        hits = 0
        for k in range(1, samples + 1):
            r = (k / samples) * lookahead
            cell = self.mapper.world_to_cell(pose["north"] + r * c, pose["east"] + r * s)
            if cell in self.dead_end_blocked_cells:
                hits += 1
        return hits >= int(getattr(self.args, "dead_end_memory_max_hits", 2))

    async def verify_and_clear_ghost_segment(self, pose: Dict[str, float], target: Tuple[float, float], reason: str = "blocked_segment") -> bool:
        """Verify a map blockage with a settled depth look, then clear ghost cells if open.

        This handles the case where the occupancy grid contains black ghost artefacts
        at a corner but the current camera view down the intended route is actually
        clear.  The drone yaws toward the route, takes a stable depth snapshot, and
        only clears map cells when the depth view confirms a viable forward corridor.
        """
        if not bool(getattr(self.args, "ghost_clearance_enabled", True)):
            return False
        dn = float(target[0]) - float(pose["north"])
        de = float(target[1]) - float(pose["east"])
        dist = math.hypot(dn, de)
        if dist < 0.4:
            return False
        desired_yaw = yaw_from_vector_deg(dn, de)
        yaw_err = abs(yaw_error_deg(desired_yaw, pose["yaw_deg"]))
        if yaw_err > float(getattr(self.args, "ghost_verify_max_turn_deg", 75.0)):
            # A huge turn probably really is a new branch, not just a false wall ahead.
            return False
        print(f"[GHOST] verifying map blockage reason={reason} target_yaw={desired_yaw:.0f} dist={dist:.1f}m")
        p2, d2 = await self.mapping_yaw_snapshot(
            desired_yaw,
            timeout_s=float(getattr(self.args, "ghost_verify_yaw_timeout_s", 0.9)),
            tolerance_deg=float(getattr(self.args, "ghost_verify_yaw_tolerance_deg", 14.0)),
        )
        if p2 is None or d2 is None:
            return False
        self.mapper.update_from_depth(d2, p2)
        left, center, right = self.clearances(d2)
        supported = self.safety.front_obstacle_supported(d2, float(getattr(self.args, "emergency_stop_m", 1.05)))
        side_ok = min(left, right) >= float(getattr(self.args, "ghost_clear_min_side_m", 0.75))
        center_ok = center >= float(getattr(self.args, "ghost_clear_center_m", 2.6))
        if center_ok and side_ok and not supported:
            cleared = self.mapper.clear_ghost_cells_near_segment(
                (p2["north"], p2["east"]),
                target,
                radius_m=float(getattr(self.args, "ghost_clear_radius_m", 0.75)),
                amount=int(getattr(self.args, "ghost_clear_amount", 7)),
                ignore_start_m=float(getattr(self.args, "path_ignore_start_m", 1.5)),
            )
            print(f"[GHOST] cleared {cleared} cells; L={left:.2f} C={center:.2f} R={right:.2f}")
            self.record_waypoint(p2, event=f"ghost_clear:{reason}", target_yaw=desired_yaw, clearances=(left, center, right), force=True)
            return cleared > 0
        print(f"[GHOST] blockage confirmed/uncertain; L={left:.2f} C={center:.2f} R={right:.2f} supported={supported}")
        return False

    def is_dead_end_view(self, left: float, center: float, right: float, depth_frame: Optional[np.ndarray]) -> bool:
        """True when the current view looks like a dead-end corridor rather than a bypassable obstacle.

        A dead-end has a close supported front wall and no wide side opening.  If only
        a single box/barrel is in front, one side should normally be open enough for
        the existing bypass/vertical logic to handle it.
        """
        if not bool(getattr(self.args, "dead_end_escape", True)):
            return False
        front_m = float(getattr(self.args, "dead_end_front_m", 1.75))
        side_m = float(getattr(self.args, "dead_end_side_open_m", 2.15))
        if center > front_m:
            return False
        if max(left, right) > side_m:
            return False
        if depth_frame is None:
            return False
        return bool(self.safety.front_obstacle_supported(depth_frame, min(front_m, float(getattr(self.args, "emergency_stop_m", 1.05)) + 0.45)))

    def choose_backtrack_targets(self, pose: Dict[str, float]) -> List[Tuple[float, float]]:
        """Choose one or more old breadcrumbs behind the drone for exiting a dead-end."""
        if not self.travel_history:
            return []
        min_back = float(getattr(self.args, "dead_end_backtrack_min_m", 2.8))
        max_back = float(getattr(self.args, "dead_end_backtrack_max_m", 8.0))
        spacing = float(getattr(self.args, "dead_end_backtrack_spacing_m", 1.8))
        cur_n, cur_e = pose["north"], pose["east"]
        candidates: List[Dict[str, float]] = []
        for h in reversed(self.travel_history[:-2]):
            d = math.hypot(cur_n - h["north"], cur_e - h["east"])
            if d >= min_back:
                candidates.append(h)
            if d >= max_back:
                break
        if not candidates:
            return []
        # Pick a sparse chain back through the breadcrumb trail, nearest useful point first.
        targets: List[Tuple[float, float]] = []
        last_n, last_e = cur_n, cur_e
        for h in candidates:
            d = math.hypot(last_n - h["north"], last_e - h["east"])
            if not targets or d >= spacing:
                targets.append((float(h["north"]), float(h["east"])))
                last_n, last_e = float(h["north"]), float(h["east"])
            if len(targets) >= int(getattr(self.args, "dead_end_backtrack_max_targets", 4)):
                break
        return targets

    async def start_dead_end_escape(self, pose: Dict[str, float], reason: str = "dead_end") -> bool:
        """Exit a dead-end quickly using breadcrumbs instead of repeated scans."""
        now = now_ms()
        min_gap = int(float(getattr(self.args, "dead_end_escape_gap_s", 2.0)) * 1000)
        if now - self.last_dead_end_escape_ms < min_gap:
            return False
        self.last_dead_end_escape_ms = now
        self.dead_end_escape_count += 1
        self.remember_dead_end_direction(pose, reason=reason)
        targets = self.choose_backtrack_targets(pose)
        self.current_path = []
        self.local_corridor_yaw = None
        self.stop_express_corridor(pose, reason="dead_end_escape")
        if targets:
            self.backtrack_targets = targets
            self.backtrack_until_ms = now + int(float(getattr(self.args, "dead_end_backtrack_timeout_s", 10.0)) * 1000)
            first = targets[0]
            yaw = yaw_from_vector_deg(first[0] - pose["north"], first[1] - pose["east"])
            print(f"[DEAD_END] {reason}: breadcrumb backtrack targets={len(targets)} first=N{first[0]:.1f},E{first[1]:.1f}, yaw={yaw:.0f}")
            self.record_waypoint(pose, event=f"dead_end_backtrack_start:{reason}", target_yaw=yaw, force=True)
            await self.drone.send_velocity(0.0, 0.0, 0.0, yaw)
            return True

        # Last resort if no history exists: rotate 180 and move out cautiously.
        yaw = wrap_deg(pose["yaw_deg"] + 180.0)
        self.local_corridor_yaw = yaw
        self.local_corridor_until_ms = now + int(float(getattr(self.args, "local_corridor_commit_s", 8.0)) * 1000)
        print(f"[DEAD_END] {reason}: no breadcrumbs; U-turn yaw={yaw:.0f}")
        self.record_waypoint(pose, event=f"dead_end_uturn:{reason}", target_yaw=yaw, force=True)
        await self.drone.send_velocity(0.0, 0.0, 0.0, yaw)
        return True

    async def drive_backtrack_step(self, pose: Dict[str, float], depth_frame: Optional[np.ndarray]) -> bool:
        """Follow breadcrumbs out of a dead end. Returns True if this step handled control."""
        if not self.backtrack_targets:
            return False
        if now_ms() > self.backtrack_until_ms:
            print("[DEAD_END] backtrack timeout; clearing and rescanning")
            self.backtrack_targets = []
            await self.scan_viable_pathways(reason="dead_end_timeout")
            return True

        target = self.backtrack_targets[0]
        dist = math.hypot(target[0] - pose["north"], target[1] - pose["east"])
        if dist <= float(getattr(self.args, "dead_end_backtrack_waypoint_radius_m", 0.75)):
            self.backtrack_targets.pop(0)
            if not self.backtrack_targets:
                print("[DEAD_END] backtrack complete; quick scan for new route")
                self.record_waypoint(pose, event="dead_end_backtrack_complete", force=True)
                await self.scan_viable_pathways(reason="dead_end_exit")
                return True
            target = self.backtrack_targets[0]
            dist = math.hypot(target[0] - pose["north"], target[1] - pose["east"])

        desired_yaw = yaw_from_vector_deg(target[0] - pose["north"], target[1] - pose["east"])
        yaw_err = yaw_error_deg(desired_yaw, pose["yaw_deg"])
        if abs(yaw_err) > float(getattr(self.args, "dead_end_turn_first_angle_deg", 35.0)):
            print(f"[DEAD_END] align yaw current={pose['yaw_deg']:.0f} target={desired_yaw:.0f} err={yaw_err:.0f}")
            await self.drone.send_velocity(0.0, 0.0, 0.0, desired_yaw)
            return True

        speed = min(float(getattr(self.args, "dead_end_backtrack_speed_m_s", 0.42)), float(self.args.cruise_speed_m_s))
        if dist < 1.2:
            speed *= clamp(dist / 1.2, 0.35, 1.0)
        vn = speed * (target[0] - pose["north"]) / max(dist, 1e-6)
        ve = speed * (target[1] - pose["east"]) / max(dist, 1e-6)
        safe_vn, safe_ve, info = self.safety.filter_velocity_ned(vn, ve, pose, depth_frame)
        if info.get("emergency"):
            print(f"[DEAD_END] backtrack emergency {info.get('reason')}; scanning")
            self.backtrack_targets = []
            await self.scan_viable_pathways(reason="dead_end_backtrack_blocked")
            return True
        # If the filter only slows/crawls, keep moving; dead-end escape is intentionally decisive.
        if math.hypot(safe_vn, safe_ve) < float(getattr(self.args, "min_progress_speed_m_s", 0.06)):
            safe_vn, safe_ve = body_to_ned(float(getattr(self.args, "corridor_crawl_speed_m_s", 0.12)), 0.0, pose["yaw_deg"])
        await self.drone.send_velocity(safe_vn, safe_ve, 0.0, desired_yaw)
        self.record_waypoint(pose, event="dead_end_backtrack_move", target_yaw=desired_yaw, force=False)
        return True

    def _visited_penalty_ahead(self, pose: Dict[str, float], yaw_deg: float, distance_m: float) -> float:
        """Approximate how travelled a candidate corridor is."""
        yaw_rad = math.radians(yaw_deg)
        steps = max(2, int(distance_m / max(0.2, self.mapper.res)))
        penalty = 0.0
        for k in range(1, steps + 1):
            d = distance_m * k / steps
            n = pose["north"] + d * math.cos(yaw_rad)
            e = pose["east"] + d * math.sin(yaw_rad)
            penalty += self.mapper.visit_count.get(self.mapper.world_to_cell(n, e), 0)
        return penalty / steps

    def _unknown_gain_ahead(self, pose: Dict[str, float], yaw_deg: float, distance_m: float, width_m: Optional[float] = None) -> int:
        """Count unknown cells in a lookahead corridor from the current pose.

        This is the local-scan version of frontier information gain.  It helps the
        drone choose doorways/exits from the current zone instead of re-driving a
        clear but already explored wall-following route.
        """
        if width_m is None:
            width_m = float(getattr(self.args, "local_unknown_width_m", 2.0))
        yaw_rad = math.radians(float(yaw_deg))
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        step = max(self.mapper.res, 0.40)
        lateral_step = max(self.mapper.res, 0.40)
        seen: Set[Cell] = set()
        count = 0
        d = step
        while d <= max(step, float(distance_m)):
            half = max(0.4, 0.5 * float(width_m) + 0.08 * d)
            lat = -half
            while lat <= half:
                n = pose["north"] + d * c - lat * s
                e = pose["east"] + d * s + lat * c
                cell = self.mapper.world_to_cell(n, e)
                if cell not in seen:
                    seen.add(cell)
                    if cell not in self.mapper.logodds:
                        count += 1
                lat += lateral_step
            d += step
        return count

    def zone_loop_detected(self, pose: Optional[Dict[str, float]] = None) -> bool:
        """Detect that we are orbiting the same room/zone instead of exiting.

        This is deliberately lightweight: if the last N breadcrumbs contain a long
        travelled distance but remain inside a bounded area and the local cells have
        been visited many times, force a gateway/unknown-space scan.
        """
        if not bool(getattr(self.args, "zone_loop_escape", True)):
            return False
        pts = self.travel_history[-int(getattr(self.args, "zone_loop_history_points", 70)):]
        if len(pts) < 12:
            return False
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b["north"] - a["north"], b["east"] - a["east"])
        if total < float(getattr(self.args, "zone_loop_min_travel_m", 24.0)):
            return False
        ns = [q["north"] for q in pts]
        es = [q["east"] for q in pts]
        span = max(max(ns) - min(ns), max(es) - min(es))
        if span > float(getattr(self.args, "zone_loop_max_span_m", 16.0)):
            return False
        if pose is None:
            pose = self.pose()
        current_visit = 0 if pose is None else self.mapper.visit_count.get(self.mapper.world_to_cell(pose["north"], pose["east"]), 0)
        closed = False
        if pose is not None:
            for q in pts[:-8]:
                if math.hypot(pose["north"] - q["north"], pose["east"] - q["east"]) < float(getattr(self.args, "zone_loop_close_m", 2.0)):
                    closed = True
                    break
        return closed or current_visit >= int(getattr(self.args, "zone_loop_min_current_visit", 5))

    async def maybe_take_side_opening(self, pose: Dict[str, float], left: float, center: float, right: float, reason: str = "side_opening") -> bool:
        """Turn into a promising side doorway/opening before continuing a room loop.

        Express corridor mode is excellent for long straight travel, but in an open
        room it can follow the perimeter and miss doorways.  If a side sector opens
        and the cells beyond that heading are mostly unknown, commit to that branch.
        """
        if not bool(getattr(self.args, "side_opening_branch_mode", True)):
            return False
        # v41: do not let side-opening checks flip-flop between left/right while a
        # freshly selected corridor/gateway is still being given a chance to make progress.
        if self.progress_commit_active(pose) and not any(tok in reason for tok in ("front_block", "dead_end", "blocked", "recovery")):
            return False
        if center < float(getattr(self.args, "side_opening_min_front_m", 2.2)):
            return False
        threshold = float(getattr(self.args, "side_opening_clear_m", 4.2))
        options: List[Tuple[str, float, float]] = []
        base = float(pose["yaw_deg"])
        # Positive yaw is a right turn in the NED yaw convention used by the rest of the file.
        if right > threshold:
            options.append(("right", wrap_deg(base + 90.0), right))
        if left > threshold:
            options.append(("left", wrap_deg(base - 90.0), left))
        if not options:
            return False
        best = None
        lookahead = float(getattr(self.args, "side_opening_unknown_lookahead_m", 7.0))
        for label, yaw, clear in options:
            if bool(getattr(self.args, "avoid_recent_path", True)) and self.heading_points_into_recent_path(pose, yaw, min(lookahead, float(getattr(self.args, "recent_heading_lookahead_m", 4.0)))):
                continue
            unknown = self._unknown_gain_ahead(pose, yaw, lookahead, float(getattr(self.args, "side_opening_unknown_width_m", 2.2)))
            visited = self._visited_penalty_ahead(pose, yaw, min(lookahead, clear))
            key = self.gateway_key(pose, yaw) if bool(getattr(self.args, "topological_gateway_mode", True)) else None
            attempts = 0 if key is None else self.gateway_attempts.get(key, 0)
            if key is not None and self.gateway_blacklisted(key):
                continue
            score = unknown - float(getattr(self.args, "side_opening_visited_weight", 10.0)) * visited + 2.0 * clear - float(getattr(self.args, "topo_attempt_penalty", 5.0)) * attempts
            if best is None or score > best[0]:
                best = (score, label, yaw, clear, unknown, visited, key)
        if best is None:
            return False
        score, label, yaw, clear, unknown, visited, key = best
        if unknown < int(getattr(self.args, "side_opening_min_unknown_cells", 18)):
            return False
        if key is not None:
            self.gateway_attempts[key] = self.gateway_attempts.get(key, 0) + 1
            self.last_gateway_selected_key = key
            self.record_gateway_event("side_opening_selected", pose, yaw, unknown, (left, center, right), score, key)
        self.stop_express_corridor(pose, f"side_opening_{label}")
        self.current_path = []
        self.local_corridor_yaw = yaw
        self.local_corridor_until_ms = now_ms() + int(float(getattr(self.args, "side_opening_commit_s", 14.0)) * 1000)
        self.start_progress_commitment(pose, yaw, f"side_opening:{label}:{reason}", duration_s=float(getattr(self.args, "side_opening_commit_s", 14.0)), min_dist_m=float(getattr(self.args, "progress_commit_min_m", 3.0)))
        print(f"[GATEWAY] side opening {label} selected yaw={yaw:.0f} clear={clear:.1f} unknown={unknown} visited={visited:.1f} score={score:.1f}")
        self.record_waypoint(pose, event=f"gateway_side_opening:{label}:{reason}", target_yaw=yaw, clearances=(left, center, right), force=True)
        await self.yaw_to_fast(yaw, timeout_s=float(getattr(self.args, "side_opening_yaw_timeout_s", 1.0)), tolerance_deg=float(getattr(self.args, "side_opening_yaw_tolerance_deg", 28.0)))
        return True

    def _express_corridor_diagnostics(
        self,
        pose: Dict[str, float],
        left: float,
        center: float,
        right: float,
        yaw_deg: Optional[float] = None,
        active: bool = False,
    ) -> Tuple[bool, str, float]:
        """Decide whether the current view looks like a straight corridor.

        This deliberately uses simple, fast depth-sector evidence rather than a
        full scan.  A corridor is defined as: clear forward depth plus two
        side boundaries at sensible distances.  If already in express mode,
        the check is looser so the drone does not exit just because one side
        briefly opens at a doorway or a barrel occludes a few pixels.
        """
        if not bool(getattr(self.args, "express_corridor_mode", True)):
            return False, "express_disabled", 999.0

        yaw = float(pose.get("yaw_deg", 0.0) if yaw_deg is None else yaw_deg)
        min_center = float(getattr(self.args, "express_corridor_min_center_m", 4.2))
        min_side = float(getattr(self.args, "express_corridor_min_side_m", 0.75))
        side_wall_max = float(getattr(self.args, "express_corridor_side_wall_max_m", 3.8))
        balance_min = float(getattr(self.args, "express_corridor_balance_ratio_min", 0.35))
        unvisited_lookahead = min(center, float(getattr(self.args, "express_corridor_unvisited_lookahead_m", 5.5)))
        visited_penalty = self._visited_penalty_ahead(pose, yaw, max(1.0, unvisited_lookahead))

        if center < min_center:
            return False, f"center_too_short:{center:.2f}", visited_penalty
        if min(left, right) < min_side:
            return False, f"side_too_close:L{left:.2f}_R{right:.2f}", visited_penalty

        # Require side boundaries, not a fully open room, when entering express
        # mode.  While active, allow one side to open briefly so the drone can pass
        # doorways/intersections without stopping immediately.
        both_walls = left <= side_wall_max and right <= side_wall_max
        one_wall = left <= side_wall_max or right <= side_wall_max
        if bool(getattr(self.args, "express_corridor_require_two_walls", True)):
            if not (both_walls or (active and one_wall)):
                return False, f"not_two_side_walls:L{left:.2f}_R{right:.2f}", visited_penalty

        ratio = min(left, right) / max(0.1, max(left, right))
        if ratio < balance_min and not active:
            return False, f"side_unbalanced:{ratio:.2f}", visited_penalty

        if not active and visited_penalty > float(getattr(self.args, "express_corridor_unvisited_max_penalty", 2.5)):
            return False, f"already_visited:{visited_penalty:.1f}", visited_penalty

        return True, "corridor_like", visited_penalty

    def start_express_corridor(self, pose: Dict[str, float], yaw_deg: float, reason: str = "auto") -> None:
        self.express_corridor_yaw = wrap_deg(float(yaw_deg))
        self.express_corridor_start_pose = pose.copy()
        self.express_corridor_start_ms = now_ms()
        self.express_corridor_until_ms = now_ms() + int(float(getattr(self.args, "express_corridor_commit_s", 14.0)) * 1000)
        self.express_corridor_block_count = 0
        # While express mode is active, do not let a stale global A* path pull the
        # drone back toward the wall.  It will replan at the corridor exit.
        self.current_path = []
        self.local_corridor_yaw = self.express_corridor_yaw
        self.local_corridor_until_ms = self.express_corridor_until_ms
        self.start_progress_commitment(pose, self.express_corridor_yaw, f"express:{reason}", duration_s=min(float(getattr(self.args, "express_corridor_commit_s", 14.0)), float(getattr(self.args, "progress_commit_s", 7.0)) + 4.0), min_dist_m=float(getattr(self.args, "progress_commit_min_m", 3.0)))
        print(f"[EXPRESS] start reason={reason} yaw={self.express_corridor_yaw:.0f}")
        self.record_waypoint(pose, event=f"express_corridor_start:{reason}", target_yaw=self.express_corridor_yaw, force=True)

    def stop_express_corridor(self, pose: Optional[Dict[str, float]], reason: str = "end") -> None:
        if self.express_corridor_yaw is not None:
            print(f"[EXPRESS] stop reason={reason}")
            self.record_waypoint(pose, event=f"express_corridor_stop:{reason}", target_yaw=self.express_corridor_yaw, force=True)
        self.express_corridor_yaw = None
        self.express_corridor_until_ms = 0
        self.express_corridor_start_pose = None
        self.express_corridor_start_ms = 0
        self.express_corridor_block_count = 0

    async def drive_express_corridor_step(self, pose: Dict[str, float], depth_frame: Optional[np.ndarray], reason: str = "express") -> bool:
        """Fast centreline driving through an already-visible corridor.

        This is the speed layer for the 5-minute round.  It only activates when
        the forward sector is clear and side sectors look like corridor walls.
        It still runs the normal safety filter and the v26 front-block handler, so
        it can go around/over/under an obstacle if the corridor is blocked.
        """
        if not bool(getattr(self.args, "express_corridor_mode", True)):
            return False
        if depth_frame is None:
            self.stop_express_corridor(pose, "no_depth")
            return False

        left, center, right = self.clearances(depth_frame)
        now = now_ms()
        active = self.express_corridor_yaw is not None and now < self.express_corridor_until_ms

        # Start express mode opportunistically from the current heading if a
        # corridor is already visible.  This avoids waiting for a new global plan.
        if not active:
            ok, why, visit_penalty = self._express_corridor_diagnostics(pose, left, center, right, pose["yaw_deg"], active=False)
            if not ok:
                return False
            self.start_express_corridor(pose, pose["yaw_deg"], reason=f"detected:{why}:visited{visit_penalty:.1f}")
            active = True

        target_yaw = float(self.express_corridor_yaw)

        # Hard limits: do not stay in express mode forever or drive past the end of
        # the visible corridor.  A side opening on both sides usually means an
        # intersection/room; exit express and let the planner or a quick scan choose.
        if now >= self.express_corridor_until_ms:
            self.stop_express_corridor(pose, "commit_timeout")
            return False
        if self.express_corridor_start_pose is not None:
            travelled = math.hypot(pose["north"] - self.express_corridor_start_pose["north"], pose["east"] - self.express_corridor_start_pose["east"])
            if travelled > float(getattr(self.args, "express_corridor_max_distance_m", 10.0)):
                self.stop_express_corridor(pose, f"max_distance:{travelled:.1f}")
                return False

        open_side = float(getattr(self.args, "express_corridor_end_open_side_m", 5.2))
        if left > open_side and right > open_side and center > float(getattr(self.args, "express_corridor_min_center_m", 4.2)):
            self.stop_express_corridor(pose, "both_sides_open")
            return False

        # v38: do not let express mode blindly orbit the same room perimeter.
        # If one side opens into unknown space, turn into that gateway instead of
        # staying on the already-travelled wall-following track.
        if (not self.progress_commit_active(pose)) and await self.maybe_take_side_opening(pose, left, center, right, reason="express"):
            return True

        ok, why, _visit_penalty = self._express_corridor_diagnostics(pose, left, center, right, target_yaw, active=True)
        if not ok and not active:
            return False

        # If the corridor is truly blocked directly ahead, switch to the v26
        # around/above/below recovery instead of just braking in place.
        if center < float(getattr(self.args, "express_corridor_front_block_m", 1.55)) and self.safety.front_obstacle_supported(depth_frame, float(getattr(self.args, "emergency_stop_m", 1.05))):
            self.stop_express_corridor(pose, "supported_front_block")
            await self.handle_front_blocked(pose, depth_frame, reason="express_front_block")
            return True

        yaw_err = yaw_error_deg(target_yaw, pose["yaw_deg"])
        if abs(yaw_err) > float(getattr(self.args, "express_corridor_yaw_tolerance_deg", 16.0)):
            print(f"[EXPRESS] aligning yaw current={pose['yaw_deg']:.0f} target={target_yaw:.0f} err={yaw_err:.0f}")
            await self.drone.send_velocity(0.0, 0.0, 0.0, target_yaw)
            return True

        # Centreline control: drift away from the closer wall.  Positive right_cmd
        # moves to vehicle-right; if the right side has more clearance, move right.
        balance_error = clamp((right - left) / max(1.0, left + right), -1.0, 1.0)
        right_cmd = clamp(
            float(getattr(self.args, "express_corridor_centerline_gain", 0.30)) * balance_error,
            -float(getattr(self.args, "express_corridor_max_lateral_m_s", 0.16)),
            float(getattr(self.args, "express_corridor_max_lateral_m_s", 0.16)),
        )

        side_min = min(left, right)
        if side_min < float(getattr(self.args, "express_corridor_min_side_m", 0.75)):
            # Stronger nudge if very close to a wall.
            if left < right:
                right_cmd = max(right_cmd, min(0.18, float(self.args.cruise_speed_m_s) * 0.30))
            else:
                right_cmd = min(right_cmd, -min(0.18, float(self.args.cruise_speed_m_s) * 0.30))

        # Speed scheduling: go fast only when the corridor is visibly long and side
        # clearance is reasonable.  cruise_speed_m_s remains the hard cap inside
        # DepthSafetyFilter, so set --cruise-speed-m-s high enough for express mode.
        express_speed = min(float(getattr(self.args, "express_corridor_speed_m_s", 0.62)), float(self.args.cruise_speed_m_s))
        if center < float(getattr(self.args, "express_corridor_slow_center_m", 3.2)):
            express_speed = min(express_speed, float(self.args.slow_speed_m_s))
        elif center < float(getattr(self.args, "express_corridor_fast_center_m", 5.5)):
            express_speed *= 0.72
        if side_min < float(getattr(self.args, "express_corridor_slow_side_m", 1.0)):
            express_speed *= 0.75

        desired_vn, desired_ve = body_to_ned(express_speed, right_cmd, pose["yaw_deg"])
        safe_vn, safe_ve, avoid_info = self.safety.filter_velocity_ned(desired_vn, desired_ve, pose, depth_frame)
        safe_speed = math.hypot(safe_vn, safe_ve)

        if avoid_info.get("active"):
            print(
                f"[EXPRESS_AVOID] {avoid_info.get('reason')} desired=({desired_vn:.2f},{desired_ve:.2f}) "
                f"safe=({safe_vn:.2f},{safe_ve:.2f}) L={avoid_info.get('left',0):.2f} C={avoid_info.get('center',0):.2f} R={avoid_info.get('right',0):.2f} "
                f"corridor={avoid_info.get('corridor',0):.2f} count={avoid_info.get('corridor_count',0)}"
            )

        if avoid_info.get("emergency"):
            self.express_corridor_block_count += 1
            if self.express_corridor_block_count >= int(getattr(self.args, "express_corridor_block_limit", 2)):
                self.stop_express_corridor(pose, "emergency_block")
                await self.handle_front_blocked(pose, depth_frame, reason="express_emergency")
            else:
                await self.stop_motion(0.05)
            return True

        if safe_speed < float(getattr(self.args, "min_progress_speed_m_s", 0.06)):
            self.express_corridor_block_count += 1
            if self.express_corridor_block_count >= int(getattr(self.args, "express_corridor_block_limit", 2)):
                self.stop_express_corridor(pose, "no_progress")
                await self.scan_viable_pathways(reason="express_no_progress")
            else:
                await self.stop_motion(0.05)
            return True

        self.express_corridor_block_count = 0
        await self.drone.send_velocity(safe_vn, safe_ve, 0.0, target_yaw)
        self.record_waypoint(pose, event="express_corridor_move", target_yaw=target_yaw, clearances=(left, center, right))
        return True

    def altitude_m(self, pose: Optional[Dict[str, float]] = None) -> float:
        """Return altitude above takeoff point from NED down coordinate."""
        p = pose if pose is not None else self.pose()
        if p is None:
            return float(getattr(self.args, "takeoff_altitude_m", 3.0))
        return max(0.0, -float(p.get("down", -float(getattr(self.args, "takeoff_altitude_m", 3.0)))))

    async def move_to_altitude(self, target_alt_m: float, reason: str = "vertical") -> bool:
        """Climb/descend safely using NED down velocity while holding x/y/yaw.

        NED convention: negative down velocity means climb, positive means descend.
        This is intentionally slow and bounded because we do not have a dedicated
        upward/downward range sensor in the organiser helpers.
        """
        pose0 = self.pose()
        if pose0 is None:
            return False
        min_alt = float(getattr(self.args, "min_flying_height_m", 2.2))
        max_alt = float(getattr(self.args, "max_flying_height_m", 7.0))
        target = clamp(float(target_alt_m), min_alt, max_alt)
        speed = abs(float(getattr(self.args, "vertical_speed_m_s", 0.35)))
        tol = float(getattr(self.args, "vertical_alt_tolerance_m", 0.18))
        timeout_s = float(getattr(self.args, "vertical_escape_timeout_s", 8.0))
        start = time.monotonic()
        print(f"[VERTICAL] {reason}: moving altitude {self.altitude_m(pose0):.2f} -> {target:.2f} m")
        while time.monotonic() - start < timeout_s:
            p = self.pose()
            if p is None:
                await asyncio.sleep(0.05)
                continue
            alt = self.altitude_m(p)
            err = target - alt
            if abs(err) <= tol:
                await self.drone.send_velocity(0.0, 0.0, 0.0, p["yaw_deg"])
                self.record_waypoint(p, event=f"altitude_reached:{reason}", target_yaw=p["yaw_deg"], force=True)
                return True
            vd = -speed if err > 0.0 else speed
            await self.drone.send_velocity(0.0, 0.0, vd, p["yaw_deg"])
            # Keep collecting map/photo data during vertical manoeuvres.
            d = self.depth.get_frame()
            if d is not None:
                self.mapper.update_from_depth(d, p)
            await self.maybe_save_or_detect(note="vertical_escape")
            await asyncio.sleep(0.10)
        p = self.pose()
        if p is not None:
            await self.drone.send_velocity(0.0, 0.0, 0.0, p["yaw_deg"])
        print(f"[VERTICAL] {reason}: altitude move timed out")
        return False

    async def vertical_escape(self, reason: str = "front_block") -> bool:
        """Try to get out of a true front blockage by changing altitude.

        Order of preference:
        1. climb by one step, up to --max-flying-height-m;
        2. if needed, climb again / to max;
        3. if already high and still blocked, try descending toward a safe minimum.

        After each altitude change, do a quick local centreline scan so the drone
        can continue around/over the obstacle instead of staying in a brake loop.
        """
        if not bool(getattr(self.args, "enable_vertical_avoidance", True)):
            return False
        now = now_ms()
        min_gap_ms = int(float(getattr(self.args, "vertical_escape_gap_s", 2.0)) * 1000)
        if now - self.last_vertical_escape_ms < min_gap_ms:
            return False
        self.last_vertical_escape_ms = now
        self.vertical_escape_count += 1

        p = self.pose()
        if p is None:
            return False
        alt = self.altitude_m(p)
        max_alt = float(getattr(self.args, "max_flying_height_m", 7.0))
        min_alt = float(getattr(self.args, "min_flying_height_m", 2.2))
        step = float(getattr(self.args, "vertical_escape_step_m", 1.0))
        clear_target = float(getattr(self.args, "vertical_clearance_target_m", 2.2))

        candidates: List[float] = []
        # Prefer climbing: in this map most problematic obstacles are boxes/barrels/walls.
        if alt + 0.25 < max_alt:
            candidates.append(min(max_alt, alt + step))
        if alt + step + 0.25 < max_alt:
            candidates.append(min(max_alt, alt + 2.0 * step))
        if abs(max_alt - alt) > 0.35:
            candidates.append(max_alt)
        # Descending is a secondary option for overhang-like obstacles; keep a safe floor margin.
        if alt - step > min_alt + 0.1:
            candidates.append(max(min_alt, alt - step))

        # De-duplicate while preserving order.
        uniq: List[float] = []
        for c in candidates:
            if min_alt <= c <= max_alt and all(abs(c - u) > 0.25 for u in uniq):
                uniq.append(c)

        if not uniq:
            print(f"[VERTICAL] {reason}: no legal altitude candidate from alt={alt:.2f} m")
            return False

        print(f"[VERTICAL] {reason}: trying altitude candidates {[round(c, 2) for c in uniq]} m")
        self.current_path = []
        self.local_corridor_yaw = None

        for target_alt in uniq:
            ok = await self.move_to_altitude(target_alt, reason=reason)
            if not ok:
                continue
            await self.stop_motion(float(getattr(self.args, "vertical_settle_s", 0.25)))
            p2 = self.pose()
            d2 = self.depth.get_frame()
            if p2 is not None and d2 is not None:
                self.mapper.update_from_depth(d2, p2)
                left, center, right = self.clearances(d2)
                print(f"[VERTICAL] at {self.altitude_m(p2):.2f} m: L={left:.2f} C={center:.2f} R={right:.2f}")
                if center >= clear_target and not self.safety.front_obstacle_supported(d2, self.args.emergency_stop_m):
                    # Commit to current heading briefly; global planner will take over on the next loop.
                    self.local_corridor_yaw = p2["yaw_deg"]
                    self.local_corridor_until_ms = now_ms() + int(float(getattr(self.args, "local_corridor_commit_s", 4.0)) * 1000)
                    self.record_waypoint(p2, event=f"vertical_escape_clear:{reason}", target_yaw=p2["yaw_deg"], clearances=(left, center, right), force=True)
                    return True

            # If straight ahead is still blocked, search around at this altitude.
            yaw = await self.scan_viable_pathways(reason=f"vertical_{reason}")
            if yaw is not None:
                return True

        print(f"[VERTICAL] {reason}: no altitude gave a clear path")
        return False

    async def handle_front_blocked(self, pose: Dict[str, float], depth_frame: Optional[np.ndarray], reason: str = "front_block") -> None:
        """Resolve a true obstacle directly ahead.

        This replaces the old behaviour where the drone repeatedly sent zero
        velocity forever.  It first tries a local around-the-obstacle scan; if no
        viable side route exists, it attempts vertical avoidance up to the configured
        maximum altitude.
        """
        self.front_block_count += 1
        await self.stop_motion(float(getattr(self.args, "front_block_stop_s", 0.05)))
        print(f"[FRONT_BLOCK] {reason} count={self.front_block_count}")
        if self.front_block_count < int(getattr(self.args, "front_block_bypass_limit", 2)):
            return

        self.current_path = []
        self.local_corridor_yaw = None

        left, center, right = self.clearances(depth_frame)

        # v37: branch-first front-block handling.  At the end of a corridor the
        # forward sector often looks blocked, but a side branch is reachable after a
        # 90-degree turn.  Older versions classified this as a dead end too early
        # and backtracked along the blue trail.  First scan side headings (+/-90,
        # then +/-45) and commit to a fresh branch if one is visible.
        yaw = await self.scan_viable_pathways(reason="front_block_bypass")
        if yaw is not None:
            p = self.pose() or pose
            d = self.depth.get_frame() if self.depth.get_frame() is not None else depth_frame
            print(f"[FRONT_BLOCK] side/branch bypass selected yaw={yaw:.0f}; not backtracking")
            await self.drive_centerline_step(p, d, reason="front_block_bypass")
            self.front_block_count = 0
            return

        # Only after a side-branch scan fails should this be treated as a true dead
        # end requiring breadcrumb backtracking.
        if self.is_dead_end_view(left, center, right, depth_frame):
            if await self.start_dead_end_escape(pose, reason=reason):
                self.front_block_count = 0
                return

        # Second preference: go above/below it, bounded by max/min altitude.
        if await self.vertical_escape(reason=reason):
            self.front_block_count = 0
            return

        # Last fallback: recovery scan, but do not stay in an emergency loop.
        await self.recovery_scan(f"{reason}_fallback")
        self.front_block_count = 0

    async def scan_viable_pathways(self, reason: str = "local_scan") -> Optional[float]:
        """Scan headings, update the map, and choose a clear, preferably unvisited pathway.

        This is the v23 local-centreline layer. It does not rely on a full global
        A* path. It simply asks: "which yaw direction gives me a wide, clear corridor
        and leads toward cells I have not already driven through?"
        """
        if not bool(getattr(self.args, "local_centerline_mode", True)):
            return None
        pose = self.pose()
        if pose is None:
            return None

        # v41: do not spend minutes doing topological yaw scans for startup/ordinary
        # replans.  Gateway scans are reserved for actual decision points: front
        # blockage, confirmed loops, dead-end exit, or explicit gateway recovery.
        topo_scan_reason = any(tok in reason for tok in ("front_block", "corner", "zone", "loop", "dead_end", "gateway", "no_reachable"))
        if bool(getattr(self.args, "topological_gateway_mode", True)) and topo_scan_reason and not reason.startswith("legacy_"):
            topo_yaw = await self.scan_topological_gateways(reason=reason)
            if topo_yaw is not None:
                return topo_yaw

        await self.stop_motion(float(getattr(self.args, "pre_scan_stop_s", 0.03)))
        base = pose["yaw_deg"]
        step = float(getattr(self.args, "local_scan_step_deg", 45.0))
        headings: List[float] = []
        seen: Set[int] = set()
        # v24: fast heading order.  Straight and shallow alternatives first, then
        # wider options only if needed.  The older v23 scanned all 8 headings, which
        # gave good maps but could waste minutes in a 5-minute qualifier.
        if reason == "startup":
            deltas = [0.0, step, -step, 2.0 * step, -2.0 * step, 180.0]
            max_headings = int(getattr(self.args, "startup_scan_max_headings", 3))
        elif "front_block" in reason or "corner" in reason:
            # v37: when front is blocked, do not waste the limited scan budget on
            # the blocked forward heading.  Probe side branches first.  This fixes
            # the common case where the drone reaches a wall/corridor end and the
            # correct action is to turn left/right, not backtrack.
            deltas = [2.0 * step, -2.0 * step, step, -step, 0.0, 3.0 * step, -3.0 * step, 180.0]
            max_headings = int(getattr(self.args, "front_block_scan_max_headings", 5))
        elif "zone_escape" in reason or "gateway" in reason or "room_loop" in reason:
            # v38: when we suspect we are orbiting the same room, spend a little
            # more scan budget and choose the heading with maximum unknown gain.
            # This targets corridor openings rather than the already-driven room perimeter.
            deltas = [0.0, step, -step, 2.0 * step, -2.0 * step, 3.0 * step, -3.0 * step, 180.0]
            max_headings = int(getattr(self.args, "zone_escape_scan_max_headings", 8))
        else:
            deltas = [0.0, step, -step, 2.0 * step, -2.0 * step, 3.0 * step, -3.0 * step, 180.0]
            max_headings = int(getattr(self.args, "local_scan_max_headings", 4))
        for delta in deltas:
            y = wrap_deg(base + delta)
            key = int(round(y))
            if key not in seen:
                seen.add(key)
                headings.append(y)
        if max_headings > 0:
            headings = headings[:max_headings]

        candidates: List[Dict[str, float]] = []
        print(f"[CENTERLINE] {reason}; quick-scanning {len(headings)} headings from yaw={base:.0f}")
        for y in headings:
            p, d = await self.mapping_yaw_snapshot(
                y,
                timeout_s=float(getattr(self.args, "local_scan_yaw_timeout_s", self.args.recovery_yaw_timeout_s)),
                tolerance_deg=float(getattr(self.args, "local_scan_yaw_tolerance_deg", self.args.recovery_yaw_tolerance_deg)),
            )
            if p is not None:
                self.mapper.update_from_depth(d, p)
            left, center, right = self.clearances(d)
            usable_dist = min(center, float(getattr(self.args, "local_corridor_lookahead_m", 5.0)))
            visit_penalty = self._visited_penalty_ahead(pose, y, usable_dist) if usable_dist > 0.5 else 0.0
            unknown_gain = self._unknown_gain_ahead(
                pose,
                y,
                max(usable_dist, float(getattr(self.args, "local_unknown_lookahead_m", 5.5))),
                float(getattr(self.args, "local_unknown_width_m", 2.0)),
            )
            turn_penalty = abs(yaw_error_deg(y, base)) / 180.0
            side_balance = min(left, right)
            is_viable = (
                center >= float(getattr(self.args, "local_corridor_min_center_m", 2.4))
                and side_balance >= float(getattr(self.args, "local_corridor_min_side_m", 0.75))
            )
            backtrack_blocked = False
            if bool(getattr(self.args, "avoid_recent_path", True)) and not ("dead_end" in reason or "backtrack" in reason):
                # A pathway that simply points back down the blue trail should not
                # be selected during normal exploration.  Confirmed dead-end escape
                # uses breadcrumbs separately and is allowed to backtrack.
                recent_thresh = float(getattr(self.args, "local_recent_penalty_block", 2.2))
                backtrack_blocked = visit_penalty >= recent_thresh or self.heading_points_into_recent_path(pose, y, min(usable_dist, float(getattr(self.args, "recent_heading_lookahead_m", 4.0))))
                if backtrack_blocked:
                    is_viable = False
            unknown_weight = float(getattr(self.args, "local_unknown_gain_weight", 0.06))
            if "zone_escape" in reason or "gateway" in reason or "room_loop" in reason:
                unknown_weight = float(getattr(self.args, "zone_escape_unknown_gain_weight", 0.13))
            score = center + 0.45 * side_balance - 0.80 * visit_penalty - 0.45 * turn_penalty + unknown_weight * math.sqrt(float(unknown_gain))
            if "front_block" in reason or "corner" in reason:
                # Prefer true side exits over trying to continue into the front wall
                # or reversing along the path we came from.  A 70-120 degree turn is
                # exactly what we want at many corridor ends/corners.
                abs_turn = abs(yaw_error_deg(y, base))
                if 65.0 <= abs_turn <= 125.0:
                    score += float(getattr(self.args, "front_block_side_turn_bonus", 3.0))
                if abs_turn < 25.0:
                    score -= float(getattr(self.args, "front_block_forward_penalty", 5.0))
            dead_end_memory_blocked = False
            if not ("dead_end" in reason or "backtrack" in reason):
                dead_end_memory_blocked = self.heading_points_to_dead_end_memory(
                    pose, y, min(usable_dist, float(getattr(self.args, "dead_end_memory_check_m", 4.5)))
                )
                if dead_end_memory_blocked:
                    is_viable = False
            if backtrack_blocked:
                score -= 12.0
            if dead_end_memory_blocked:
                score -= 18.0
            if not is_viable:
                score -= 8.0
            candidates.append({"yaw": y, "left": left, "center": center, "right": right, "score": score, "visited": visit_penalty, "unknown": float(unknown_gain), "viable": 1.0 if is_viable else 0.0})
            print(f"[CENTERLINE] yaw={y:.0f} L={left:.2f} C={center:.2f} R={right:.2f} unknown={unknown_gain} visited={visit_penalty:.1f} backtrack={backtrack_blocked} deadmem={dead_end_memory_blocked} viable={is_viable} score={score:.2f}")
            await self.maybe_save_or_detect(note="centerline_scan")

            # v24: early exit.  Once a heading is clearly usable, do not keep
            # scanning all alternatives.  Preference for unvisited paths is still
            # used when no excellent candidate is found immediately.
            if (
                bool(getattr(self.args, "fast_scan_early_exit", True))
                and is_viable
                and not ("zone_escape" in reason or "gateway" in reason or "room_loop" in reason)
                and center >= float(getattr(self.args, "local_scan_good_center_m", 4.0))
                and side_balance >= float(getattr(self.args, "local_scan_good_side_m", 0.9))
            ):
                print(f"[CENTERLINE] early-exit: heading {y:.0f} is already clear enough")
                break

        self.local_corridor_candidates = candidates
        viable = [c for c in candidates if c["viable"] > 0.5]
        if not viable:
            print("[CENTERLINE] no viable local pathway found")
            self.local_corridor_yaw = None
            return None

        best = max(viable, key=lambda c: c["score"])
        self.local_corridor_yaw = wrap_deg(best["yaw"])
        self.local_corridor_until_ms = now_ms() + int(float(getattr(self.args, "local_corridor_commit_s", 5.0)) * 1000)
        p_commit = self.pose()
        if p_commit is not None:
            self.start_progress_commitment(p_commit, self.local_corridor_yaw, f"centerline:{reason}", duration_s=float(getattr(self.args, "local_corridor_commit_s", 5.0)), min_dist_m=float(getattr(self.args, "progress_commit_min_m", 3.0)))
        print(f"[CENTERLINE] selected yaw={self.local_corridor_yaw:.0f} C={best['center']:.2f} L={best['left']:.2f} R={best['right']:.2f} score={best['score']:.2f}")
        self.record_waypoint(self.pose(), event=f"selected_centerline:{reason}", target_yaw=self.local_corridor_yaw, clearances=(best["left"], best["center"], best["right"]), force=True)
        # v29: if the scan found a corridor-like heading, immediately promote it
        # to express mode.  The express step will align yaw, centre the drone, and
        # drive through the corridor without returning to global A* every second.
        if bool(getattr(self.args, "auto_express_after_scan", True)) and best["center"] >= float(getattr(self.args, "express_corridor_min_center_m", 4.2)):
            p_now = self.pose()
            if p_now is not None:
                self.start_express_corridor(p_now, self.local_corridor_yaw, reason=f"scan:{reason}")
        await self.yaw_to_mapping_heading(
            self.local_corridor_yaw,
            timeout_s=float(getattr(self.args, "local_scan_yaw_timeout_s", self.args.recovery_yaw_timeout_s)),
            tolerance_deg=float(getattr(self.args, "local_scan_yaw_tolerance_deg", self.args.recovery_yaw_tolerance_deg)),
        )
        return self.local_corridor_yaw

    async def drive_centerline_step(self, pose: Dict[str, float], depth_frame: Optional[np.ndarray], reason: str = "centerline") -> bool:
        """Move along the selected local corridor centreline for one control step."""
        if self.local_corridor_yaw is None or now_ms() > self.local_corridor_until_ms:
            yaw = await self.scan_viable_pathways(reason=reason)
            if yaw is None:
                return False

        target_yaw = float(self.local_corridor_yaw)
        yaw_err = yaw_error_deg(target_yaw, pose["yaw_deg"])
        if abs(yaw_err) > float(getattr(self.args, "local_corridor_yaw_tolerance_deg", 18.0)):
            print(f"[CENTERLINE] aligning yaw current={pose['yaw_deg']:.0f} target={target_yaw:.0f} err={yaw_err:.0f}")
            await self.drone.send_velocity(0.0, 0.0, 0.0, target_yaw)
            return True

        left, center, right = self.clearances(depth_frame)

        # v38: if committed centreline travel passes a doorway/opening to unknown
        # space, branch through it rather than continuing a perimeter loop.
        if (not self.progress_commit_active(pose)) and await self.maybe_take_side_opening(pose, left, center, right, reason=reason):
            return True

        if center < float(getattr(self.args, "local_corridor_min_continue_m", 1.7)):
            hard_block = center < float(getattr(self.args, "local_corridor_hard_block_m", 1.15))
            supported = False if depth_frame is None else bool(self.safety.front_obstacle_supported(depth_frame, min(float(getattr(self.args, "emergency_stop_m", 1.05)) + 0.35, float(getattr(self.args, "local_corridor_min_continue_m", 1.7)))))
            if hard_block or supported:
                print(f"[CENTERLINE] corridor ended/blocked C={center:.2f}; rescanning")
                self.local_corridor_yaw = None
                await self.scan_viable_pathways(reason="corridor_blocked")
                return True
            # v41: marginal depth in a cluttered start corner should not trigger a
            # decision loop. Creep forward and let the safety filter handle true hazards.
            print(f"[CENTERLINE] marginal front C={center:.2f}; committing/crawling instead of rescanning")

        # Centre the drone between visible side boundaries. Positive right_cmd moves
        # to the vehicle's right. If right side has more clearance than left, drift
        # right; if left has more clearance, drift left.
        balance_error = clamp((right - left) / max(1.0, left + right), -1.0, 1.0)
        right_cmd = clamp(
            float(getattr(self.args, "centerline_gain", 0.22)) * balance_error,
            -float(getattr(self.args, "centerline_max_lateral_m_s", 0.12)),
            float(getattr(self.args, "centerline_max_lateral_m_s", 0.12)),
        )
        # If both sides are very open, do not chase noisy side readings.
        if left > float(getattr(self.args, "centerline_open_side_m", 4.0)) and right > float(getattr(self.args, "centerline_open_side_m", 4.0)):
            right_cmd = 0.0

        forward = min(float(getattr(self.args, "local_corridor_speed_m_s", self.args.cruise_speed_m_s)), float(self.args.cruise_speed_m_s))
        if center < float(getattr(self.args, "front_slow_m", 2.8)):
            forward = min(forward, float(self.args.slow_speed_m_s))

        desired_vn, desired_ve = body_to_ned(forward, right_cmd, pose["yaw_deg"])
        safe_vn, safe_ve, avoid_info = self.safety.filter_velocity_ned(desired_vn, desired_ve, pose, depth_frame)
        safe_speed = math.hypot(safe_vn, safe_ve)
        if avoid_info.get("active"):
            print(
                f"[CENTERLINE_AVOID] {avoid_info.get('reason')} desired=({desired_vn:.2f},{desired_ve:.2f}) "
                f"safe=({safe_vn:.2f},{safe_ve:.2f}) L={avoid_info.get('left',0):.2f} C={avoid_info.get('center',0):.2f} R={avoid_info.get('right',0):.2f} "
                f"corridor={avoid_info.get('corridor',0):.2f} count={avoid_info.get('corridor_count',0)}"
            )
        if avoid_info.get("emergency") or safe_speed < float(getattr(self.args, "min_progress_speed_m_s", 0.06)):
            self.local_soft_block_count += 1
            if self.local_soft_block_count >= int(getattr(self.args, "local_block_scan_limit", 4)):
                print(f"[CENTERLINE] movement blocked {self.local_soft_block_count} times; rescanning")
                self.local_soft_block_count = 0
                self.local_corridor_yaw = None
                await self.scan_viable_pathways(reason="centerline_avoid_blocked")
            else:
                await self.stop_motion(0.08)
            return True

        self.local_soft_block_count = 0
        await self.drone.send_velocity(safe_vn, safe_ve, 0.0, target_yaw)
        self.record_waypoint(pose, event="centerline_move", target_yaw=target_yaw, clearances=(left, center, right))
        return True

    async def yaw_to_fast(self, yaw_deg: float, timeout_s: float = 1.5, tolerance_deg: float = 8.0) -> None:
        """Fast yaw helper for non-mapping turns."""
        yaw_deg = wrap_deg(yaw_deg)
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            pose = self.pose()
            current = yaw_deg if pose is None else pose["yaw_deg"]
            await self.drone.send_velocity(0.0, 0.0, 0.0, yaw_deg)
            if abs(yaw_error_deg(yaw_deg, current)) <= tolerance_deg:
                break
            await asyncio.sleep(0.08)
        await asyncio.sleep(self.args.yaw_settle_s)

    async def yaw_to_mapping_heading(self, yaw_deg: float, timeout_s: float, tolerance_deg: float) -> None:
        """
        Turn to a scan heading and wait until the yaw is actually stable.

        This is deliberately slower than yaw_to_fast(). Mapping with a depth frame
        captured during a fast turn produces smeared/rotated obstacle points and can
        create ghost walls in the occupancy grid.
        """
        yaw_deg = wrap_deg(yaw_deg)
        start = time.monotonic()
        stable_since: Optional[float] = None
        last_yaw: Optional[float] = None
        last_t = time.monotonic()

        while time.monotonic() - start < timeout_s:
            pose = self.pose()
            current = yaw_deg if pose is None else float(pose["yaw_deg"])
            err = abs(yaw_error_deg(yaw_deg, current))

            now_t = time.monotonic()
            yaw_rate = 999.0
            if last_yaw is not None:
                dt = max(1e-3, now_t - last_t)
                yaw_rate = abs(yaw_error_deg(current, last_yaw)) / dt
            last_yaw = current
            last_t = now_t

            await self.drone.send_velocity(0.0, 0.0, 0.0, yaw_deg)

            if err <= tolerance_deg and yaw_rate <= float(self.args.scan_stable_yaw_rate_deg_s):
                if stable_since is None:
                    stable_since = now_t
                elif now_t - stable_since >= float(self.args.scan_stable_time_s):
                    break
            else:
                stable_since = None

            await asyncio.sleep(0.08)

        # Keep commanding the same yaw during the settle interval so the depth frame
        # used for mapping is less likely to be from the previous heading.
        settle_until = time.monotonic() + float(self.args.mapping_settle_s)
        while time.monotonic() < settle_until:
            await self.drone.send_velocity(0.0, 0.0, 0.0, yaw_deg)
            await asyncio.sleep(0.08)

    async def settled_depth_snapshot(self) -> Optional[np.ndarray]:
        """Collect a few depth frames after settling and median-filter them."""
        frames: List[np.ndarray] = []
        samples = max(1, int(self.args.mapping_depth_samples))
        gap = max(0.0, float(self.args.mapping_depth_sample_gap_s))
        for _ in range(samples):
            await asyncio.sleep(gap)
            d = self.depth.get_frame()
            if d is not None:
                frames.append(d.astype(np.float32, copy=False))
        if not frames:
            return self.depth.get_frame()
        if len(frames) == 1:
            return frames[0].copy()
        return np.nanmedian(np.stack(frames, axis=0), axis=0).astype(np.float32)

    async def mapping_yaw_snapshot(self, yaw_deg: float, timeout_s: float, tolerance_deg: float) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
        """Yaw to a heading, wait for stabilization, then return pose + depth."""
        await self.yaw_to_mapping_heading(yaw_deg, timeout_s=timeout_s, tolerance_deg=tolerance_deg)
        p = self.pose()
        d = await self.settled_depth_snapshot()
        if p is not None:
            p = p.copy()
        return p, d

    async def maybe_save_or_detect(self, note: str = "interval", force: bool = False) -> None:
        pose = self.pose()
        frame = self.rgb.get_frame()
        depth_frame = self.depth.get_frame()
        if pose is None or frame is None:
            return

        interesting, interest_label, _ = color_interest(frame, self.args.interest_min_pixels)
        elapsed_ms = now_ms() - self.last_photo_ms
        moved_enough = False
        if self.last_saved_pose is not None:
            moved = math.hypot(pose["north"] - self.last_saved_pose["north"], pose["east"] - self.last_saved_pose["east"])
            moved_enough = moved >= self.args.sparse_photo_every_m
        else:
            moved_enough = True

        should_save = force
        if self.args.mode == "collect":
            if interesting and elapsed_ms >= int(self.args.interesting_save_gap_s * 1000):
                should_save = True
                note = f"{note}_color_interest"
            elif self.args.save_sparse_context and moved_enough and elapsed_ms >= int(self.args.photo_interval_s * 1000):
                should_save = True
                note = f"{note}_sparse_context"
            elif elapsed_ms >= int(self.args.max_photo_gap_s * 1000):
                should_save = True
                note = f"{note}_max_gap"
        else:
            # Count mode: run YOLO periodically, but do not save every raw frame.
            should_save = force or elapsed_ms >= int(self.args.photo_interval_s * 1000)

        frame_id = ""
        if should_save:
            frame_id = self.logger.save(frame, pose, self.args.mode, note=note, interest=interest_label)
            self.last_photo_ms = now_ms()
            self.last_saved_pose = pose.copy()

        if self.counter is not None:
            detect_due = now_ms() - self.last_detection_ms >= int(self.args.detect_interval_s * 1000)
            if detect_due or force:
                if not frame_id:
                    frame_id = f"det_{now_ms()}"
                detections = self.counter.process(frame, depth_frame, pose, Path(frame_id).stem)
                self.last_detection_ms = now_ms()
                if detections:
                    compact = [
                        (
                            d["group"],
                            f"id={d.get('track_id', '-')}",
                            round(d["confidence"], 2),
                            round(d["depth_m"], 2),
                            d.get("depth_source", ""),
                            "new" if d.get("is_new_track") else "dup",
                            "counted" if d.get("counted") else "seen",
                        )
                        for d in detections
                    ]
                    print(f"[DETECT] {compact} counts={self.counter.counts()} raw={self.counter.raw_counts()}")

        # Collect mode: when a red/yellow-looking blob appears, take a tiny fast yaw bracket, not a full 360.
        if (
            self.args.mode == "collect"
            and self.args.enable_interest_sweep
            and interesting
            and now_ms() - self.last_interest_sweep_ms >= int(self.args.interest_sweep_gap_s * 1000)
        ):
            self.last_interest_sweep_ms = now_ms()
            await self.micro_sweep(base_note="interest_sweep")

    async def micro_sweep(self, base_note: str) -> None:
        pose = self.pose()
        if pose is None:
            return
        base = pose["yaw_deg"]
        print(f"[SWEEP] micro sweep around possible barrel at yaw {base:.0f}")
        for delta in (-25.0, 0.0, 25.0):
            await self.yaw_to_fast(base + delta, timeout_s=1.0, tolerance_deg=10.0)
            await self.maybe_save_or_detect(note=f"{base_note}_{int(delta)}", force=True)
        await self.yaw_to_fast(base, timeout_s=1.0, tolerance_deg=10.0)

    async def scan_for_frontiers(self, force: bool = False) -> None:
        """Depth-only yaw scan when the planner cannot find reachable frontiers."""
        if not force and now_ms() - self.last_scan_ms < int(self.args.frontier_scan_gap_s * 1000):
            return
        pose = self.pose()
        if pose is None:
            return
        self.last_scan_ms = now_ms()
        base = pose["yaw_deg"]
        max_headings = int(getattr(self.args, "frontier_scan_max_headings", 4))
        deltas = [0.0, 60.0, -60.0, 120.0, -120.0, 180.0]
        if max_headings > 0:
            deltas = deltas[:max_headings]
        print(f"[SCAN] doing {len(deltas)} settled depth snapshots")
        for delta in deltas:
            p, d = await self.mapping_yaw_snapshot(
                base + delta,
                timeout_s=self.args.scan_yaw_timeout_s,
                tolerance_deg=self.args.scan_yaw_tolerance_deg,
            )
            self.mapper.update_from_depth(d, p)
            await self.maybe_save_or_detect(note="frontier_scan")
        await self.yaw_to_mapping_heading(base, timeout_s=self.args.scan_yaw_timeout_s, tolerance_deg=self.args.scan_yaw_tolerance_deg)

    async def recovery_scan(self, reason: str) -> None:
        """
        Active unsticking behaviour: stop, yaw through candidate headings, update the
        depth map at each heading, choose the heading with the best forward clearance,
        then make a short cautious forward move if it is safe.
        """
        if now_ms() - self.last_recovery_ms < int(self.args.recovery_scan_gap_s * 1000):
            print(f"[RECOVERY] cooldown active; reason={reason}")
            await self.scan_for_frontiers(force=True)
            return

        pose = self.pose()
        if pose is None:
            return

        self.last_recovery_ms = now_ms()
        await self.stop_motion(0.15)
        base_yaw = pose["yaw_deg"]
        step = float(self.args.recovery_yaw_step_deg)
        deltas = [0.0, step, -step, 2 * step, -2 * step, 3 * step, -3 * step, 180.0]
        # Remove duplicates after wrapping, while preserving order.
        headings: List[float] = []
        seen: Set[int] = set()
        for d in deltas:
            y = wrap_deg(base_yaw + d)
            key = int(round(y))
            if key not in seen:
                seen.add(key)
                headings.append(y)
        max_recovery = int(getattr(self.args, "recovery_scan_max_headings", 4))
        if max_recovery > 0:
            headings = headings[:max_recovery]

        best_yaw = base_yaw
        best_score = -1e9
        best_clearance = 0.0
        print(f"[RECOVERY] reason={reason}; scanning {len(headings)} headings from yaw={base_yaw:.0f}")

        for y in headings:
            p, d = await self.mapping_yaw_snapshot(
                y,
                timeout_s=self.args.recovery_yaw_timeout_s,
                tolerance_deg=self.args.recovery_yaw_tolerance_deg,
            )
            self.mapper.update_from_depth(d, p)
            left, center, right = self.clearances(d)
            turn_penalty = abs(yaw_error_deg(y, base_yaw)) / 180.0
            score = center + 0.25 * min(left, right) - 0.35 * turn_penalty
            if center < self.args.recovery_min_clearance_m:
                score -= 4.0
            print(f"[RECOVERY] yaw={y:.0f} L={left:.2f} C={center:.2f} R={right:.2f} score={score:.2f}")
            if score > best_score:
                best_score = score
                best_yaw = y
                best_clearance = center

        await self.yaw_to_mapping_heading(best_yaw, timeout_s=self.args.recovery_yaw_timeout_s, tolerance_deg=self.args.recovery_yaw_tolerance_deg)
        print(f"[RECOVERY] selected yaw={best_yaw:.0f} center_clearance={best_clearance:.2f}")

        if best_clearance >= self.args.recovery_min_clearance_m and self.args.recovery_forward_s > 0.0:
            t_end = time.monotonic() + float(self.args.recovery_forward_s)
            while time.monotonic() < t_end:
                p = self.pose()
                d = self.depth.get_frame()
                if p is None or d is None:
                    break
                self.mapper.update_from_depth(d, p)
                _, center, _ = self.clearances(d)
                if center < self.args.emergency_stop_m:
                    print("[RECOVERY] aborting forward nudge: obstacle too close")
                    break
                desired_vn, desired_ve = body_to_ned(float(self.args.recovery_forward_speed_m_s), 0.0, p["yaw_deg"])
                safe_vn, safe_ve, avoid_info = self.safety.filter_velocity_ned(desired_vn, desired_ve, p, d)
                await self.drone.send_velocity(safe_vn, safe_ve, 0.0, p["yaw_deg"])
                if avoid_info.get("emergency"):
                    print(f"[RECOVERY] avoidance emergency during nudge: {avoid_info.get('reason')}")
                    break
                await asyncio.sleep(0.10)

        await self.stop_motion(0.10)
        self.current_path = []
        self.last_plan_ms = 0

    def need_replan(self) -> bool:
        if not self.current_path:
            return True
        return now_ms() - self.last_plan_ms >= int(self.args.replan_interval_s * 1000)

    def replan(self, pose: Dict[str, float]) -> None:
        rejected = self.active_rejected_frontiers()
        recent_cells = self.recent_path_cells(pose) if bool(getattr(self.args, "avoid_recent_path", True)) else set()

        # v35: first try to plan without reusing the recent breadcrumb trail.
        # Backtracking is handled by explicit dead-end escape, not by ordinary frontier planning.
        path = self.planner.plan_to_best_frontier(
            pose,
            min_frontier_dist_m=self.args.min_frontier_dist_m,
            max_frontiers_to_try=self.args.max_frontiers_to_try,
            rejected_cells=rejected,
            recent_cells=recent_cells,
            avoid_recent=bool(getattr(self.args, "avoid_recent_path", True)),
            max_recent_path_hits=int(getattr(self.args, "recent_path_max_hits", 0)),
            recent_penalty_weight=float(getattr(self.args, "recent_path_penalty", 10.0)),
            turnback_penalty_weight=float(getattr(self.args, "turnback_penalty", 5.0)),
        )

        if path is None and bool(getattr(self.args, "allow_recent_path_fallback", False)):
            # Optional fallback for debugging only.  The recommended competition mode
            # leaves this disabled so the drone scans forward instead of returning
            # along the same blue trail unless a dead-end was confirmed.
            path = self.planner.plan_to_best_frontier(
                pose,
                min_frontier_dist_m=self.args.min_frontier_dist_m,
                max_frontiers_to_try=self.args.max_frontiers_to_try,
                rejected_cells=rejected,
                recent_cells=recent_cells,
                avoid_recent=False,
                recent_penalty_weight=float(getattr(self.args, "recent_path_penalty", 10.0)),
                turnback_penalty_weight=float(getattr(self.args, "turnback_penalty", 5.0)),
            )
            if path:
                print("[PLAN] fallback reused recent path; consider leaving --allow-recent-path-fallback disabled")

        self.last_plan_ms = now_ms()
        if path:
            self.current_path = path
            goal = path[-1]
            if path:
                first = path[0]
                yaw0 = yaw_from_vector_deg(first[0] - pose["north"], first[1] - pose["east"])
                self.start_progress_commitment(pose, yaw0, "global_plan", duration_s=float(getattr(self.args, "progress_commit_s", 7.0)), min_dist_m=max(1.5, 0.65 * float(getattr(self.args, "progress_commit_min_m", 3.0))))
            print(f"[PLAN] path_len={len(path)} goal N={goal[0]:.1f} E={goal[1]:.1f} known_cells={len(self.mapper.logodds)} rejected={len(rejected)} recent_cells={len(recent_cells)}")
        else:
            self.current_path = []
            print(f"[PLAN] no non-backtracking frontier; known_cells={len(self.mapper.logodds)} rejected={len(rejected)} recent_cells={len(recent_cells)}")

    async def drive_path_step(self, pose: Dict[str, float], depth_frame: Optional[np.ndarray]) -> None:
        left, center, right = self.clearances(depth_frame)
        self.mapper.clear_pose_footprint(pose, reason="pre_navigation")
        self.update_travel_history(pose)
        print(
            f"[NAV] N={pose['north']:.1f} E={pose['east']:.1f} yaw={pose['yaw_deg']:.0f} "
            f"L={left:.2f} C={center:.2f} R={right:.2f} path={len(self.current_path)} saved={self.logger.saved_count}"
        )

        # v32: if a dead-end escape is active, follow breadcrumbs out before doing
        # any new frontier planning.  This prevents the drone from repeatedly
        # trying to plan deeper into a corridor that has already terminated.
        if await self.drive_backtrack_step(pose, depth_frame):
            return

        # No depth means no safe flight. Stop and wait for the sensor to recover.
        if depth_frame is None:
            print("[NAV] no depth frame; holding position")
            self.current_path = []
            await self.stop_motion(0.2)
            return

        # Immediate collision protection before planning logic.
        if center < self.args.emergency_stop_m and self.safety.front_obstacle_supported(depth_frame, self.args.emergency_stop_m):
            # v32: if this is a real dead-end corridor, backtrack decisively instead
            # of spending time scanning/vertical-escaping into a wall.
            if self.is_dead_end_view(left, center, right, depth_frame):
                if await self.start_dead_end_escape(pose, reason="supported_front_dead_end"):
                    return
            # v26: a true obstacle in front should trigger around/over recovery,
            # not an infinite brake loop.
            front_stats = self.safety._front_obstacle_stats(depth_frame, self.args.emergency_stop_m)
            print(
                "[NAV] front blocked: supported obstacle too close "
                f"count={front_stats.get('count', 0)} area={front_stats.get('area', 0)} "
                f"span=({front_stats.get('row_span', 0)}x{front_stats.get('col_span', 0)})"
            )
            await self.handle_front_blocked(pose, depth_frame, reason="supported_front_obstacle")
            return
        else:
            # Decay the counter after clean front frames so one old blockage does not
            # trigger a vertical escape much later.
            self.front_block_count = max(0, self.front_block_count - 1)

        # v43: progress-first routing.  Earlier versions allowed local gateway scans,
        # express mode, and centreline mode to override the global route every few
        # centimetres.  That created start-corner decision loops.  Here we plan early
        # and, if a route exists, follow it before non-essential local behaviours.
        # Emergency braking, confirmed front blockage, dead-end escape, and ghost
        # clearing still run below.
        if bool(getattr(self.args, "global_path_priority", True)):
            if self.need_replan() and not (self.current_path and self.progress_commit_active(pose)):
                self.replan(pose)

        path_priority_active = bool(getattr(self.args, "global_path_priority", True)) and bool(self.current_path)

        # v38: if we are looping inside a room/zone, pause the perimeter-following
        # behaviour and deliberately scan for a gateway into unknown space.
        if (not path_priority_active) and (not self.progress_commit_active(pose)) and self.zone_loop_detected(pose) and now_ms() - self.last_zone_escape_ms > int(float(getattr(self.args, "zone_loop_cooldown_s", 18.0)) * 1000):
            self.last_zone_escape_ms = now_ms()
            print("[ZONE] probable local loop; forcing gateway/unknown-space scan")
            self.current_path = []
            self.stop_express_corridor(pose, "zone_loop_escape")
            self.local_corridor_yaw = None
            await self.scan_viable_pathways(reason="zone_escape")
            return

        # v39: online topological gateway policy.  If a side doorway/opening is visible
        # while moving, branch into it before the global frontier planner can pull the
        # drone into another loop around the current room.  This is a cheap check; the
        # more expensive yaw scans are only used at junctions, dead ends, or loops.
        if (not path_priority_active) and bool(getattr(self.args, "topological_gateway_mode", True)) and not self.progress_commit_active(pose):
            if now_ms() - self.last_gateway_scan_ms > int(float(getattr(self.args, "topo_side_check_gap_s", 1.2)) * 1000):
                if await self.maybe_take_side_opening(pose, left, center, right, reason="topo_side_gateway"):
                    self.last_gateway_scan_ms = now_ms()
                    return

        # v29: high-speed corridor behaviour.  If a straight corridor is visible,
        # align to its centreline and drive it directly.  This is much faster than
        # repeatedly choosing short frontier waypoints inside the same corridor.
        if (not path_priority_active) and bool(getattr(self.args, "express_corridor_mode", True)):
            moved_express = await self.drive_express_corridor_step(pose, depth_frame, reason="nav")
            if moved_express:
                return

        # v24: after a quick scan has selected a clear centreline, actually use it
        # for a short committed interval instead of immediately falling back into
        # repeated global replanning. This is what lets the drone leave the start
        # area quickly while still using the accurate occupancy mapper.
        if (
            (not path_priority_active)
            and bool(getattr(self.args, "prefer_committed_centerline", True))
            and self.local_corridor_yaw is not None
            and now_ms() < self.local_corridor_until_ms
        ):
            moved = await self.drive_centerline_step(pose, depth_frame, reason="committed_centerline")
            if moved:
                return

        if (not bool(getattr(self.args, "global_path_priority", True))) and self.need_replan():
            self.replan(pose)

        if not self.current_path:
            if bool(getattr(self.args, "local_centerline_mode", True)):
                moved = await self.drive_centerline_step(pose, depth_frame, reason="no_global_path")
                if moved:
                    return
            self.no_path_count += 1
            await self.stop_motion(0.1)
            if self.no_path_count >= self.args.no_path_recovery_limit:
                await self.recovery_scan("no_reachable_frontier")
                self.no_path_count = 0
            else:
                await self.scan_for_frontiers()
            return

        # Drop waypoints already reached.
        while self.current_path:
            target = self.current_path[0]
            dist = math.hypot(target[0] - pose["north"], target[1] - pose["east"])
            if dist > self.args.waypoint_radius_m:
                break
            self.current_path.pop(0)

        if not self.current_path:
            self.replan(pose)
            return

        target = self.current_path[0]

        # We have a candidate path, so reset the no-path counter.
        self.no_path_count = 0

        # v35: avoid reusing the same path unless an explicit dead-end backtrack is active.
        # If the next waypoint points back down the recent breadcrumb trail, discard this
        # path and do a quick local scan instead of turning around.
        if bool(getattr(self.args, "avoid_recent_path", True)) and not self.backtrack_targets:
            dn_tmp = target[0] - pose["north"]
            de_tmp = target[1] - pose["east"]
            dist_tmp = math.hypot(dn_tmp, de_tmp)
            if dist_tmp > float(getattr(self.args, "recent_path_exclude_current_m", 1.8)):
                yaw_tmp = yaw_from_vector_deg(dn_tmp, de_tmp)
                if self.heading_points_into_recent_path(pose, yaw_tmp, min(dist_tmp, float(getattr(self.args, "recent_heading_lookahead_m", 4.0)))):
                    print(f"[PLAN] refusing to reuse recent path toward yaw={yaw_tmp:.0f}; scanning for a fresh branch")
                    self.current_path = []
                    await self.scan_viable_pathways(reason="avoid_recent_path")
                    return

        # Do not fly a straight segment through an inflated mapped obstacle. This catches corner cuts.
        segment_clear = self.mapper.path_segment_collision_free(
            (pose["north"], pose["east"]),
            target,
            inflation_extra_m=self.args.path_extra_inflation_m,
            ignore_start_m=self.args.path_ignore_start_m,
        )
        if not segment_clear:
            # A false occupied cell under or immediately beside the vehicle can
            # invalidate the first segment even though the corridor ahead is open.
            # Clear the self footprint and re-test once before doing any recovery
            # behaviour that might reverse or turn the drone around.
            cleared_self = self.mapper.clear_pose_footprint(pose, reason="blocked_segment_self_check")
            if cleared_self:
                segment_clear = self.mapper.path_segment_collision_free(
                    (pose["north"], pose["east"]),
                    target,
                    inflation_extra_m=self.args.path_extra_inflation_m,
                    ignore_start_m=max(self.args.path_ignore_start_m, float(getattr(self.args, "self_clear_path_ignore_start_m", 1.9))),
                )
        if not segment_clear:
            # v35: before rejecting the route and turning back, verify whether the
            # blockage is just a map ghost.  If the depth camera confirms the route
            # is open, clear the offending cells and keep the forward route.
            if await self.verify_and_clear_ghost_segment(pose, target, reason="path_segment_blocked"):
                self.segment_block_count = 0
                self.last_plan_ms = 0
                self.replan(self.pose() or pose)
                return

            self.segment_block_count += 1
            reject_target = self.current_path[-1] if self.current_path else target
            self.reject_frontier_world(reject_target, reason="blocked_first_segment")
            print(f"[NAV] next path segment intersects inflated obstacle; replanning/stuck_count={self.segment_block_count}")
            self.current_path = []
            await self.stop_motion(0.1)
            if self.segment_block_count >= self.args.stuck_replan_limit:
                await self.recovery_scan("blocked_path_segment")
                self.segment_block_count = 0
            else:
                self.replan(pose)
            return

        self.segment_block_count = 0

        dn = target[0] - pose["north"]
        de = target[1] - pose["east"]
        dist = math.hypot(dn, de)
        desired_yaw = yaw_from_vector_deg(dn, de)

        speed = self.args.cruise_speed_m_s
        if dist < 1.5:
            speed *= clamp(dist / 1.5, 0.30, 1.0)
        if center < self.args.front_slow_m:
            speed = min(speed, self.args.slow_speed_m_s)

        # v20/v21: nose-first navigation.  The forward-facing depth camera is much
        # more reliable when the drone moves mostly forward.  In v19, the drone
        # could start in a corner, pick a diagonal waypoint, then command a large
        # sideways component while still facing the old yaw.  The direction-aware
        # collision corridor would correctly reject that sideways/diagonal motion,
        # but the planner would keep asking for the same impossible command.  Here
        # we first rotate toward the waypoint, then translate.
        yaw_err_signed = yaw_error_deg(desired_yaw, pose["yaw_deg"])
        yaw_err_abs = abs(yaw_err_signed)
        turn_first_angle = float(getattr(self.args, "turn_first_angle_deg", 18.0))
        if yaw_err_abs > turn_first_angle:
            if now_ms() - getattr(self, "last_turn_first_log_ms", 0) > 700:
                self.last_turn_first_log_ms = now_ms()
                print(
                    f"[NAV] turn-first yaw current={pose['yaw_deg']:.0f} "
                    f"target={desired_yaw:.0f} err={yaw_err_signed:.0f}; holding translation"
                )
            await self.drone.send_velocity(0.0, 0.0, 0.0, desired_yaw)
            return

        desired_vn = speed * dn / max(dist, 1e-6)
        desired_ve = speed * de / max(dist, 1e-6)

        safe_vn, safe_ve, avoid_info = self.safety.filter_velocity_ned(desired_vn, desired_ve, pose, depth_frame)
        if avoid_info.get("active"):
            print(
                f"[AVOID] {avoid_info.get('reason')} "
                f"desired=({desired_vn:.2f},{desired_ve:.2f}) safe=({safe_vn:.2f},{safe_ve:.2f}) "
                f"L={avoid_info.get('left', 0):.2f} C={avoid_info.get('center', 0):.2f} R={avoid_info.get('right', 0):.2f} "
                f"corridor={avoid_info.get('corridor', 0):.2f} count={avoid_info.get('corridor_count', 0)}"
            )
            safe_speed = math.hypot(safe_vn, safe_ve)
            if avoid_info.get("emergency"):
                self.avoid_block_count += 1
                self.current_path = []
                reason = str(avoid_info.get("reason", ""))
                if "front" in reason and self.is_dead_end_view(left, center, right, depth_frame):
                    if await self.start_dead_end_escape(pose, reason="safety_front_dead_end"):
                        self.avoid_block_count = 0
                        return
                if "front" in reason and self.avoid_block_count >= int(getattr(self.args, "front_block_bypass_limit", 2)):
                    print(f"[FRONT_BLOCK] safety filter blocked forward motion {self.avoid_block_count} times; trying bypass/vertical escape")
                    self.avoid_block_count = 0
                    await self.handle_front_blocked(pose, depth_frame, reason="safety_front_emergency")
                    return
                if self.avoid_block_count >= int(getattr(self.args, "avoid_block_recovery_limit", 5)):
                    print(f"[RECOVERY] avoidance blocked movement {self.avoid_block_count} times; doing active scan")
                    self.avoid_block_count = 0
                    await self.recovery_scan("avoidance_blocked_path")
                    return
            elif safe_speed < float(getattr(self.args, "min_progress_speed_m_s", 0.06)) and bool(getattr(self.args, "local_centerline_mode", True)):
                self.local_soft_block_count += 1
                print(f"[CENTERLINE] global path command slowed to zero count={self.local_soft_block_count}; switching to local centreline")
                if self.local_soft_block_count >= int(getattr(self.args, "local_block_scan_limit", 3)):
                    self.current_path = []
                    self.local_soft_block_count = 0
                    await self.scan_viable_pathways(reason="global_corridor_slowdown")
                    await self.drive_centerline_step(pose, depth_frame, reason="global_corridor_slowdown")
                    return
            else:
                self.local_soft_block_count = 0
        else:
            self.avoid_block_count = 0
            self.local_soft_block_count = 0

        # Near obstacles, do not combine translation with a large yaw change. Sudden yaw
        # can make depth/pose alignment worse and can cause controller drift near walls.
        yaw_err = abs(yaw_error_deg(desired_yaw, pose["yaw_deg"]))
        near_obstacle = (
            avoid_info.get("emergency")
            or center < self.args.avoid_safe_m
            or left < self.args.side_critical_m
            or right < self.args.side_critical_m
            or float(avoid_info.get("corridor", self.args.map_ray_max_m)) < self.args.avoid_safe_m
        )
        if near_obstacle and yaw_err > float(getattr(self.args, "avoid_hold_yaw_angle_deg", 20.0)):
            cmd_yaw = pose["yaw_deg"]
        else:
            cmd_yaw = pose["yaw_deg"] if avoid_info.get("emergency") else desired_yaw
        await self.drone.send_velocity(safe_vn, safe_ve, 0.0, cmd_yaw)
        self.record_waypoint(pose, event="global_path_move", target_yaw=cmd_yaw, clearances=(left, center, right))

    async def arm_and_takeoff_at_configured_altitude(self) -> None:
        """Set MAVSDK takeoff altitude from the CLI before using the organiser Drone helper.

        This keeps compatibility with the original drone_control.py, whose
        arm_and_takeoff() method takes no arguments, while also working if you
        later patch drone_control.py to accept an altitude argument.
        """
        altitude_m = max(0.5, float(self.args.takeoff_altitude_m))

        # The organiser Drone wrapper stores the MAVSDK System as self.drone.drone.
        # MAVSDK set_takeoff_altitude uses positive metres above the takeoff point,
        # not NED down.
        try:
            await self.drone.drone.action.set_takeoff_altitude(altitude_m)
            print(f"[MISSION] requested takeoff altitude={altitude_m:.2f} m")
        except Exception as exc:
            print(f"[WARN] could not set takeoff altitude via MAVSDK: {exc}")

        # Compatibility path:
        # - original Drone.arm_and_takeoff() accepts no arguments;
        # - patched versions may accept arm_and_takeoff(altitude_m).
        try:
            await self.drone.arm_and_takeoff(altitude_m)
        except TypeError:
            await self.drone.arm_and_takeoff()

    async def run(self) -> None:
        print(f"[MISSION] mode={self.args.mode} output_dir={self.output_dir}")
        print("[MISSION] Connecting...")
        await self.drone.connect()
        self.monitor_task = asyncio.create_task(position_monitor_task(self.drone, self.state, self.stop_event))
        await asyncio.sleep(1.0)

        print(f"[MISSION] Arming and taking off to {self.args.takeoff_altitude_m:.2f} m...")
        await self.arm_and_takeoff_at_configured_altitude()
        await self.wait_for_ready_data(timeout_s=20.0)

        pose = self.pose()
        if pose is not None:
            self.start_pose = pose.copy()
            print(f"[MISSION] start pose N={pose['north']:.2f} E={pose['east']:.2f} D={pose['down']:.2f} yaw={pose['yaw_deg']:.1f}")

        # v41: speedrun startup.  In a 5-minute qualifier, even quick yaw scans can cause
        # moving is too expensive.  Default is a quick centreline scan over only a
        # few headings; default is now none, so it plans from the first forward depth view
        # and keeps mapping while moving. Use --startup-scan-mode quick/full for debugging.
        startup_mode = str(getattr(self.args, "startup_scan_mode", "quick")).lower()
        if bool(getattr(self.args, "no_startup_scan", False)):
            startup_mode = "none"
        if startup_mode == "full":
            await self.scan_for_frontiers(force=True)
            if bool(getattr(self.args, "local_centerline_mode", True)):
                await self.scan_viable_pathways(reason="startup")
        elif startup_mode == "quick":
            if bool(getattr(self.args, "local_centerline_mode", True)):
                await self.scan_viable_pathways(reason="startup")
            else:
                await self.scan_for_frontiers(force=True)
        else:
            print("[MISSION] startup scan skipped")

        start_time = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.args.duration_s:
                    print("[STOP] duration reached")
                    break

                pose = self.pose()
                depth_frame = self.depth.get_frame()
                if pose is None:
                    await self.stop_motion(0.1)
                    continue

                self.mapper.update_from_depth(depth_frame, pose)
                self.autosave_map_if_due(pose)
                await self.maybe_save_or_detect(note="nav")
                await self.drive_path_step(pose, depth_frame)
                await asyncio.sleep(1.0 / self.args.loop_hz)

        except asyncio.CancelledError:
            print("[MISSION] cancelled")
        except KeyboardInterrupt:
            print("[MISSION] keyboard interrupt")
        finally:
            await self.stop_motion(0.5)
            await self.maybe_save_or_detect(note="final", force=True)
            map_path = self.mapper.save_debug_map(self.output_dir, self.pose())
            print(f"[MAP] saved {map_path}")

            self.stop_event.set()
            if self.monitor_task is not None:
                self.monitor_task.cancel()
                await asyncio.gather(self.monitor_task, return_exceptions=True)

            if self.counter is not None:
                tracks_path = self.counter.save_tracks(self.output_dir)
                counts = self.counter.counts()
                raw_counts = self.counter.raw_counts()
                result_payload = {
                    "red": counts["red"],
                    "yellow": counts["yellow"],
                    "total": counts["total"],
                    **raw_counts,
                    "tracks_csv": str(tracks_path),
                    "model_path": self.counter.model_path,
                }
                print("QUALIFIER_RESULT " + json.dumps(result_payload, sort_keys=True))
            else:
                print(f"[COLLECT] saved_images={self.logger.saved_count}")
                print(f"[COLLECT] metadata={self.logger.csv_path}")

            try:
                self.waypoint_csv_file.close()
                print(f"[MISSION] waypoints={self.waypoint_csv_path}")
            except Exception:
                pass
            try:
                self.gateway_csv_file.close()
                print(f"[MISSION] gateway_graph={self.gateway_csv_path}")
            except Exception:
                pass
            self.logger.close()

            if not self.args.no_land:
                print("[MISSION] Landing...")
                await self.drone.land()
            else:
                print("[MISSION] --no-land set; leaving drone hovering/offboard")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast progress-first online exploration qualifier mission with YOLO barrel counting and self-footprint cleanup")

    p.add_argument("--mode", choices=["collect", "count"], default="collect")
    p.add_argument("--model", default="", help="YOLO model path for count mode. Use best.pt if possible; ONNX also works with Ultralytics if runtime support is installed.")
    p.add_argument("--model-zip", default="", help="Zip containing trained YOLO weights. The script will extract it and auto-select train/weights/best.pt.")
    p.add_argument("--model-dir", default="", help="Directory to search recursively for best.pt / my_model.pt / best.onnx if --model is not supplied")
    p.add_argument("--model-extract-dir", default="", help="Optional folder where --model-zip should be extracted")
    p.add_argument("--output-dir", default="qualifier_output_v44")
    p.add_argument("--rgb-topic", default=DEFAULT_RGB_TOPIC)
    p.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)

    # Mission and control timing.
    p.add_argument("--duration-s", type=float, default=900.0)
    p.add_argument("--takeoff-altitude-m", type=float, default=3.0, help="Target takeoff altitude in metres above the start point. Use 2.5 or 3.0 for this map.")
    p.add_argument("--max-flying-height-m", type=float, default=7.0, help="Maximum allowed flight altitude above takeoff point for vertical obstacle avoidance")
    p.add_argument("--min-flying-height-m", type=float, default=2.2, help="Minimum allowed flight altitude above takeoff point for vertical obstacle avoidance")
    p.add_argument("--enable-vertical-avoidance", action=argparse.BooleanOptionalAction, default=True, help="When a real front obstacle blocks progress, try climbing/descending within min/max height before giving up")
    p.add_argument("--vertical-escape-step-m", type=float, default=1.0, help="Altitude step used when trying to climb over or descend under an obstacle")
    p.add_argument("--vertical-speed-m-s", type=float, default=0.35, help="Vertical speed used for obstacle escape; NED sign is handled internally")
    p.add_argument("--vertical-clearance-target-m", type=float, default=2.2, help="Required center clearance after changing altitude before committing forward")
    p.add_argument("--vertical-escape-timeout-s", type=float, default=8.0)
    p.add_argument("--vertical-alt-tolerance-m", type=float, default=0.18)
    p.add_argument("--vertical-settle-s", type=float, default=0.25)
    p.add_argument("--vertical-escape-gap-s", type=float, default=2.0)
    p.add_argument("--front-block-bypass-limit", type=int, default=1, help="Consecutive supported front blocks before trying around/vertical recovery")
    p.add_argument("--front-block-stop-s", type=float, default=0.05)
    p.add_argument("--loop-hz", type=float, default=18.0)
    p.add_argument("--cruise-speed-m-s", type=float, default=1.22)
    p.add_argument("--slow-speed-m-s", type=float, default=0.32)
    p.add_argument("--front-slow-m", type=float, default=2.8)
    p.add_argument("--emergency-stop-m", type=float, default=1.15)
    p.add_argument("--yaw-settle-s", type=float, default=0.08)
    p.add_argument("--no-land", action="store_true")

    # Occupancy grid / frontier planning.
    p.add_argument("--grid-resolution-m", type=float, default=0.40)
    p.add_argument("--map-ray-max-m", type=float, default=8.0)
    p.add_argument("--map-safety-radius-m", type=float, default=0.85)
    p.add_argument("--self-clear-enabled", action=argparse.BooleanOptionalAction, default=True, help="Clear a small occupancy bubble around the drone so the map never treats the drone itself as an obstacle")
    p.add_argument("--self-clear-radius-m", type=float, default=0.72, help="Radius around the current pose to force free in the occupancy grid")
    p.add_argument("--self-clear-free-log", type=int, default=4, help="Free log-odds strength used when clearing the current vehicle footprint")
    p.add_argument("--self-clear-path-ignore-start-m", type=float, default=1.9, help="Extra first-segment ignore distance after clearing an own-footprint obstacle")
    p.add_argument("--map-sample-cols", type=int, default=104)
    p.add_argument("--map-occ-threshold", type=int, default=4, help="Log-odds threshold before a cell is treated as occupied; higher reduces ghost walls")
    p.add_argument("--map-occ-update", type=int, default=2, help="Log-odds increment for one obstacle observation")
    p.add_argument("--map-prune-diagonal-ghosts", action=argparse.BooleanOptionalAction, default=True, help="Suppress thin diagonal occupied chains that are usually depth projection artefacts at corners")
    p.add_argument("--map-diagonal-prune-min-axis-run-cells", type=int, default=3, help="Keep occupied cells that are part of at least this many-cell horizontal/vertical wall run")
    p.add_argument("--map-diagonal-prune-max-component-cells", type=int, default=14, help="Only prune weak diagonal components up to this size")
    p.add_argument("--map-diagonal-prune-apply-decay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--map-diagonal-prune-decay", type=int, default=2)
    p.add_argument("--map-regularize", action=argparse.BooleanOptionalAction, default=True, help="v31: regularize occupied cells into conservative straight wall lines; rectangles are optional and off by default")
    p.add_argument("--map-regularize-use-for-planning", action=argparse.BooleanOptionalAction, default=True, help="Use regularized line/rectangle occupied cells for A* inflation instead of raw speckle cells")
    p.add_argument("--map-regularize-manhattan", action=argparse.BooleanOptionalAction, default=True, help="Snap fitted wall lines to north/east grid axes; suitable for the rectangular competition map")
    p.add_argument("--map-regularize-close-radius-m", type=float, default=0.35, help="Bridge small gaps in wall observations before line fitting; lower values avoid closing corners/doorways")
    p.add_argument("--map-regularize-open-radius-m", type=float, default=0.10, help="Remove tiny occupied speckles before line/rectangle fitting")
    p.add_argument("--map-regularize-min-component-cells", type=int, default=4, help="Ignore occupied components smaller than this during primitive fitting")
    p.add_argument("--map-regularize-min-line-m", type=float, default=1.2, help="Minimum length of fitted wall/boundary line")
    p.add_argument("--map-regularize-line-gap-m", type=float, default=0.55, help="Maximum gap that can be bridged when fitting wall lines")
    p.add_argument("--map-regularize-wall-thickness-m", type=float, default=0.24, help="Thickness of regularized wall lines on the grid")
    p.add_argument("--map-regularize-hough-threshold", type=int, default=5, help="Hough vote threshold for regularized line fitting")
    p.add_argument("--map-regularize-min-rect-side-m", type=float, default=0.45, help="Minimum rectangle side length for compact box/obstacle fitting")
    p.add_argument("--map-regularize-max-rect-side-m", type=float, default=4.5, help="Maximum rectangle side length; larger components are treated as walls/boundaries, not filled boxes")
    p.add_argument("--map-regularize-rectangles", action=argparse.BooleanOptionalAction, default=False, help="Enable rectangle fitting for compact obstacle islands. Off by default because wall corners can look like false boxes.")
    p.add_argument("--map-regularize-rect-min-fill-ratio", type=float, default=0.22, help="If rectangle fitting is enabled, require this occupied/bbox area ratio")
    p.add_argument("--map-regularize-rect-max-aspect", type=float, default=4.0, help="If rectangle fitting is enabled, reject long thin components")
    p.add_argument("--map-regularize-rect-thickness-m", type=float, default=0.30, help="Thickness of rectangle obstacle outlines")
    p.add_argument("--map-regularize-keep-strong-raw", action=argparse.BooleanOptionalAction, default=False, help="Also keep very strong raw occupied cells so partial box faces are not deleted")
    p.add_argument("--map-regularize-strong-raw-threshold", type=int, default=6, help="Raw occupied log-odds threshold preserved even if not fitted to a primitive")
    p.add_argument("--map-regularize-min-output-cells", type=int, default=8, help="Fallback to raw occupied cells if regularization finds fewer cells than this")
    p.add_argument("--map-regularize-clean-display", action=argparse.BooleanOptionalAction, default=True, help="Show raw occupied evidence as grey and accepted regularized primitives as black in occupancy_grid.png")
    p.add_argument("--no-diagonal-corner-cutting", action=argparse.BooleanOptionalAction, default=True, help="Prevent A* from using diagonal moves through blocked cell corners")
    p.add_argument("--replan-interval-s", type=float, default=1.6)
    p.add_argument("--min-frontier-dist-m", type=float, default=4.0)
    p.add_argument("--max-frontiers-to-try", type=int, default=180)
    p.add_argument("--frontier-unknown-radius-m", type=float, default=3.0, help="v38: radius around a frontier used to estimate information gain / doorway quality")
    p.add_argument("--frontier-unknown-lookahead-m", type=float, default=7.0, help="v38: unknown-space lookahead beyond a frontier")
    p.add_argument("--frontier-unknown-width-m", type=float, default=2.4, help="v38: width of the unknown-space lookahead corridor")
    p.add_argument("--frontier-unknown-weight", type=float, default=0.38, help="v38: reward for frontiers bordering large unknown regions")
    p.add_argument("--frontier-unknown-ahead-weight", type=float, default=0.85, help="v38: reward for unknown space beyond the frontier direction")
    p.add_argument("--frontier-progress-weight", type=float, default=0.18, help="v38: small reward for choosing farther exits over nearby perimeter edges")
    p.add_argument("--frontier-path-cost-weight", type=float, default=0.18, help="v41: lower than 1.0 makes the planner prefer far high-information exits instead of nearest local edges")
    p.add_argument("--frontier-local-loop-penalty", type=float, default=42.0, help="v41: penalty for short low-information perimeter frontiers near the current zone")
    p.add_argument("--frontier-local-loop-path-cells", type=int, default=14, help="v41: path length below this can be treated as a local-loop frontier if information gain is weak")
    # v39 topological gateway exploration.  These make the high-level policy prefer
    # doorway/corridor exits into unknown zones over short perimeter frontiers inside
    # the current room.  Frontier planning remains available as fallback.
    p.add_argument("--topological-gateway-mode", action=argparse.BooleanOptionalAction, default=True, help="v39: prefer online gateway/doorway graph exploration over nearest-frontier looping")
    p.add_argument("--topo-frontier-min-unknown-ahead", type=int, default=24, help="Minimum unknown cells ahead for a frontier to be treated as a gateway")
    p.add_argument("--topo-frontier-min-unknown-near", type=int, default=18)
    p.add_argument("--topo-frontier-gateway-bonus", type=float, default=16.0)
    p.add_argument("--topo-frontier-depth-bonus", type=float, default=0.16)
    p.add_argument("--topo-frontier-perimeter-penalty", type=float, default=55.0)
    p.add_argument("--topo-node-radius-m", type=float, default=3.0)
    p.add_argument("--topo-gateway-key-distance-m", type=float, default=4.0)
    p.add_argument("--topo-gateway-yaw-bin-deg", type=float, default=30.0)
    p.add_argument("--topo-scan-step-deg", type=float, default=45.0)
    p.add_argument("--topo-normal-scan-max-headings", type=int, default=3)
    p.add_argument("--topo-gateway-scan-max-headings", type=int, default=4)
    p.add_argument("--topo-front-block-max-headings", type=int, default=5)
    p.add_argument("--topo-scan-yaw-timeout-s", type=float, default=0.75)
    p.add_argument("--topo-scan-yaw-tolerance-deg", type=float, default=18.0)
    p.add_argument("--topo-gateway-min-center-m", type=float, default=2.6)
    p.add_argument("--topo-gateway-min-side-m", type=float, default=0.55)
    p.add_argument("--topo-gateway-min-unknown-cells", type=int, default=10)
    p.add_argument("--topo-recent-override-unknown-cells", type=int, default=38)
    p.add_argument("--topo-unknown-lookahead-m", type=float, default=8.5)
    p.add_argument("--topo-unknown-width-m", type=float, default=2.8)
    p.add_argument("--topo-unknown-weight", type=float, default=0.25)
    p.add_argument("--topo-clearance-weight", type=float, default=0.85)
    p.add_argument("--topo-side-balance-weight", type=float, default=0.50)
    p.add_argument("--topo-visited-weight", type=float, default=3.0)
    p.add_argument("--topo-attempt-penalty", type=float, default=5.0)
    p.add_argument("--topo-turn-weight", type=float, default=0.45)
    p.add_argument("--topo-side-branch-bonus", type=float, default=8.0)
    p.add_argument("--topo-recent-penalty", type=float, default=12.0)
    p.add_argument("--topo-gateway-commit-s", type=float, default=14.0)
    p.add_argument("--topo-side-check-gap-s", type=float, default=4.0)
    p.add_argument("--waypoint-radius-m", type=float, default=1.80)
    p.add_argument("--path-stride-m", type=float, default=2.75, help="Distance between simplified A* waypoints on straight segments. Corners keep extra waypoints.")
    p.add_argument("--path-turn-padding-cells", type=int, default=2, help="Keep this many extra A* cells around a turn so the drone does not cut/collide with corners")
    p.add_argument("--frontier-scan-gap-s", type=float, default=6.0)
    p.add_argument("--scan-yaw-timeout-s", type=float, default=0.95)
    p.add_argument("--scan-yaw-tolerance-deg", type=float, default=12.0)
    p.add_argument("--scan-stable-yaw-rate-deg-s", type=float, default=16.0)
    p.add_argument("--scan-stable-time-s", type=float, default=0.06)
    p.add_argument("--mapping-settle-s", type=float, default=0.08)
    p.add_argument("--mapping-depth-samples", type=int, default=1)
    p.add_argument("--mapping-depth-sample-gap-s", type=float, default=0.07)
    p.add_argument("--map-save-interval-s", type=float, default=4.0, help="Autosave occupancy_grid.png during flight; 0 disables")
    p.add_argument("--path-extra-inflation-m", type=float, default=0.0, help="Extra inflation when checking next path segment")
    p.add_argument("--path-ignore-start-m", type=float, default=1.45, help="Ignore this much of the path segment near the current pose during inflated-obstacle checks")
    p.add_argument("--reject-frontier-ttl-s", type=float, default=20.0, help="How long to avoid a frontier that repeatedly caused a blocked segment")
    p.add_argument("--stuck-replan-limit", type=int, default=3, help="Blocked-segment replans before active yaw recovery")
    p.add_argument("--no-path-recovery-limit", type=int, default=2, help="No-path loops before active yaw recovery")
    p.add_argument("--recovery-scan-gap-s", type=float, default=4.0)
    p.add_argument("--recovery-yaw-step-deg", type=float, default=45.0)
    p.add_argument("--recovery-yaw-timeout-s", type=float, default=0.95)
    p.add_argument("--recovery-yaw-tolerance-deg", type=float, default=12.0)
    p.add_argument("--recovery-settle-s", type=float, default=0.03)
    p.add_argument("--recovery-min-clearance-m", type=float, default=2.0)
    p.add_argument("--recovery-forward-s", type=float, default=1.4)
    p.add_argument("--recovery-forward-speed-m-s", type=float, default=0.36)
    p.add_argument("--turn-first-angle-deg", type=float, default=45.0, help="Yaw in place before translating if waypoint heading differs by more than this. This avoids diagonal/sideways motion with a forward-facing depth camera.")

    # Reactive avoidance. These wrap the organiser AvoidancePlanner.py / avoid.py logic.
    p.add_argument("--avoid-safe-m", type=float, default=2.0, help="Start slowing/blending inside this front/corridor distance")
    p.add_argument("--avoid-critical-m", type=float, default=1.20, help="Critical distance used by the reactive safety filter")
    p.add_argument("--side-critical-m", type=float, default=0.78, help="Cancel motion toward side obstacles inside this distance")
    p.add_argument("--side-hard-m", type=float, default=0.42, help="Hard side clearance threshold for immediate braking/cancelling sideways motion")
    p.add_argument("--emergency-action", choices=["brake", "sidestep"], default="brake", help="Emergency behaviour. Brake is safest because there is no rear camera.")
    p.add_argument("--emergency-confirm-frames", type=int, default=1)
    p.add_argument("--sidestep-min-clearance-m", type=float, default=1.8)
    p.add_argument("--clearance-percentile", type=float, default=25.0)
    p.add_argument("--avoid-band-y1", type=float, default=0.22)
    p.add_argument("--avoid-band-y2", type=float, default=0.58)
    p.add_argument("--avoid-hard-band-y2", type=float, default=0.52, help="Lower edge of the narrower image band used only for hard-front close-pixel support")
    p.add_argument("--front-close-x1", type=float, default=0.38, help="Left edge of the narrow central front support band as fraction of image width")
    p.add_argument("--front-close-x2", type=float, default=0.62, help="Right edge of the narrow central front support band as fraction of image width")
    p.add_argument("--front-close-min-pixels", type=int, default=260, help="Hard front brake requires at least this many close pixels in the narrow central support band")
    p.add_argument("--front-slow-min-pixels", type=int, default=420, help="Front slowdown requires this many close pixels, preventing a few bad pixels from freezing the drone")
    p.add_argument("--front-component-min-area-px", type=int, default=90, help="v25: hard front brake requires a connected close-depth component of at least this area")
    p.add_argument("--front-component-min-row-span-px", type=int, default=12, help="v25: hard front brake component must span this many image rows")
    p.add_argument("--front-component-min-col-span-px", type=int, default=8, help="v25: hard front brake component must span this many image columns")
    p.add_argument("--front-hard-confirm-frames", type=int, default=2, help="v25: require this many consecutive supported front-obstacle frames before hard braking")
    p.add_argument("--collision-radius-m", type=float, default=0.34, help="Direction-aware safety corridor half-width including drone/prop margin")
    p.add_argument("--collision-corridor-growth", type=float, default=0.03)
    p.add_argument("--collision-lookahead-m", type=float, default=2.6)
    p.add_argument("--collision-depth-stride", type=int, default=6)
    p.add_argument("--collision-min-points", type=int, default=12)
    p.add_argument("--collision-reaction-s", type=float, default=0.35)
    p.add_argument("--collision-brake-accel-m-s2", type=float, default=1.20)
    p.add_argument("--collision-extra-margin-m", type=float, default=0.12)
    p.add_argument("--corridor-hard-brake-clearance-m", type=float, default=0.75, help="Only hard-brake for corridor obstacles closer than this; farther corridor hits slow the drone instead")
    p.add_argument("--corridor-hard-brake-min-points", type=int, default=35, help="Minimum supported depth points before a corridor hit can become a hard brake")
    p.add_argument("--corridor-clear-front-override-m", type=float, default=3.0, help="If center/side sectors are this open, marginal corridor hits are treated as slowdowns")
    p.add_argument("--corridor-min-scale-front-clear", type=float, default=0.45, help="Minimum crawl speed fraction when corridor is suspicious but the front view is clear")
    p.add_argument("--avoid-block-recovery-limit", type=int, default=5, help="After this many consecutive avoidance hard-stops, trigger an active recovery scan")
    p.add_argument("--no-startup-scan", action="store_true", help="Skip the settled yaw scan before first movement")
    p.add_argument("--startup-scan-mode", choices=["quick", "full", "none"], default="none", help="quick=scan only a few headings and move; full=v23-style full scan; none=move immediately")
    p.add_argument("--startup-scan-max-headings", type=int, default=1, help="Number of headings for the quick startup centreline scan")
    p.add_argument("--frontier-scan-max-headings", type=int, default=2, help="Limit expensive frontier yaw scans")
    p.add_argument("--recovery-scan-max-headings", type=int, default=2, help="Limit expensive recovery yaw scans")
    p.add_argument("--absolute-min-depth-m", type=float, default=0.38)
    p.add_argument("--avoid-smoothing-alpha", type=float, default=0.40)
    p.add_argument("--avoid-hold-yaw-angle-deg", type=float, default=20.0, help="Near obstacles, hold current yaw if the target yaw differs by more than this")
    p.add_argument("--no-reactive-avoidance", action="store_true", help="Disable the reactive AvoidancePlanner safety filter")

    # v23 local corridor-centreline exploration.
    p.add_argument("--local-centerline-mode", action=argparse.BooleanOptionalAction, default=True, help="Use local scanned corridor centrelines when global frontier/collision gating gets stuck")
    p.add_argument("--local-scan-step-deg", type=float, default=45.0)
    p.add_argument("--local-scan-max-headings", type=int, default=2, help="Maximum headings for local centreline scans; keeps scans short")
    p.add_argument("--local-unknown-lookahead-m", type=float, default=6.0, help="v38: lookahead used to score local scan headings by unknown-space gain")
    p.add_argument("--local-unknown-width-m", type=float, default=2.2, help="v38: width used to score unknown-space gain for local scan headings")
    p.add_argument("--local-unknown-gain-weight", type=float, default=0.08, help="v38: score weight for unknown-space gain during ordinary local scans")
    p.add_argument("--zone-escape-scan-max-headings", type=int, default=8, help="v38: headings checked when escaping a room/zone loop")
    p.add_argument("--zone-escape-unknown-gain-weight", type=float, default=0.18, help="v38: stronger unknown-space reward during zone escape scans")
    p.add_argument("--front-block-scan-max-headings", type=int, default=5, help="When front is blocked, scan side headings first before declaring a dead end/backtracking")
    p.add_argument("--front-block-side-turn-bonus", type=float, default=3.0, help="Score bonus for +/-90 degree side branches during front-block bypass")
    p.add_argument("--front-block-forward-penalty", type=float, default=5.0, help="Score penalty for trying to continue forward during front-block bypass")
    p.add_argument("--fast-scan-early-exit", action=argparse.BooleanOptionalAction, default=True, help="Stop scanning once a clearly viable centreline is found")
    p.add_argument("--local-scan-good-center-m", type=float, default=3.2, help="Early-exit center clearance threshold")
    p.add_argument("--local-scan-good-side-m", type=float, default=0.90, help="Early-exit side clearance threshold")
    p.add_argument("--prefer-committed-centerline", action=argparse.BooleanOptionalAction, default=True, help="After selecting a centreline, drive it for the commit time before returning to global planning")
    p.add_argument("--pre-scan-stop-s", type=float, default=0.03, help="Short stop before a quick scan")
    p.add_argument("--local-scan-yaw-timeout-s", type=float, default=0.65)
    p.add_argument("--local-scan-yaw-tolerance-deg", type=float, default=18.0)
    p.add_argument("--local-corridor-min-center-m", type=float, default=2.2)
    p.add_argument("--local-corridor-min-side-m", type=float, default=0.70)
    p.add_argument("--local-corridor-min-continue-m", type=float, default=1.7)
    p.add_argument("--local-corridor-lookahead-m", type=float, default=5.5)
    p.add_argument("--local-corridor-commit-s", type=float, default=12.0)
    p.add_argument("--local-corridor-speed-m-s", type=float, default=0.92)
    p.add_argument("--local-corridor-yaw-tolerance-deg", type=float, default=24.0)
    p.add_argument("--centerline-gain", type=float, default=0.22)
    p.add_argument("--centerline-max-lateral-m-s", type=float, default=0.10)
    p.add_argument("--centerline-open-side-m", type=float, default=4.0)
    p.add_argument("--local-block-scan-limit", type=int, default=5)
    p.add_argument("--express-corridor-mode", action=argparse.BooleanOptionalAction, default=True, help="v29: when a straight corridor is visible, lock to its centreline and drive it faster instead of repeatedly replanning short waypoints")
    p.add_argument("--auto-express-after-scan", action=argparse.BooleanOptionalAction, default=True, help="Promote a good scanned centreline directly into express corridor mode")
    p.add_argument("--express-corridor-speed-m-s", type=float, default=1.45, help="Target speed inside a detected straight corridor; capped by --cruise-speed-m-s")
    p.add_argument("--express-corridor-min-center-m", type=float, default=3.0, help="Minimum forward clearance required to enter express corridor mode")
    p.add_argument("--express-corridor-min-side-m", type=float, default=0.65, help="Minimum side clearance required for express corridor mode")
    p.add_argument("--express-corridor-side-wall-max-m", type=float, default=3.8, help="Side clearances below this are treated as continuous corridor walls")
    p.add_argument("--express-corridor-require-two-walls", action=argparse.BooleanOptionalAction, default=True, help="Require both left and right boundaries before entering express mode")
    p.add_argument("--express-corridor-balance-ratio-min", type=float, default=0.32, help="Minimum min(left,right)/max(left,right) ratio when entering express mode")
    p.add_argument("--express-corridor-centerline-gain", type=float, default=0.30)
    p.add_argument("--express-corridor-max-lateral-m-s", type=float, default=0.16)
    p.add_argument("--express-corridor-yaw-tolerance-deg", type=float, default=24.0)
    p.add_argument("--express-corridor-commit-s", type=float, default=32.0, help="Maximum time to stay committed to one express corridor before replanning")
    p.add_argument("--express-corridor-max-distance-m", type=float, default=28.0, help="Maximum distance to drive on one express corridor before replanning")
    p.add_argument("--express-corridor-front-block-m", type=float, default=1.35, help="If center clearance is below this in express mode, trigger bypass/vertical avoidance")
    p.add_argument("--express-corridor-block-limit", type=int, default=2)
    p.add_argument("--express-corridor-unvisited-lookahead-m", type=float, default=5.5)
    p.add_argument("--express-corridor-unvisited-max-penalty", type=float, default=2.5, help="Do not enter express mode along a heavily travelled corridor unless already committed")
    p.add_argument("--express-corridor-end-open-side-m", type=float, default=5.2, help="If both sides open beyond this, treat it as an intersection/room and stop express mode")
    p.add_argument("--side-opening-branch-mode", action=argparse.BooleanOptionalAction, default=True, help="v38: turn into promising side openings/gateways instead of orbiting the same room")
    p.add_argument("--side-opening-clear-m", type=float, default=4.0, help="Side clearance required before checking a side opening as a branch")
    p.add_argument("--side-opening-min-front-m", type=float, default=2.2, help="Require this much front clearance before committing to a side opening")
    p.add_argument("--side-opening-unknown-lookahead-m", type=float, default=7.0)
    p.add_argument("--side-opening-unknown-width-m", type=float, default=2.3)
    p.add_argument("--side-opening-min-unknown-cells", type=int, default=14)
    p.add_argument("--side-opening-visited-weight", type=float, default=10.0)
    p.add_argument("--side-opening-commit-s", type=float, default=14.0)
    p.add_argument("--side-opening-yaw-timeout-s", type=float, default=1.0)
    p.add_argument("--side-opening-yaw-tolerance-deg", type=float, default=28.0)
    p.add_argument("--express-corridor-slow-center-m", type=float, default=2.3)
    p.add_argument("--express-corridor-fast-center-m", type=float, default=4.0)
    p.add_argument("--express-corridor-slow-side-m", type=float, default=0.80)
    p.add_argument("--min-progress-speed-m-s", type=float, default=0.06)
    p.add_argument("--waypoint-record-every-m", type=float, default=0.65)
    p.add_argument("--corridor-allow-crawl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--corridor-min-scale-always", type=float, default=0.25)
    p.add_argument("--force-crawl-on-corridor-slowdown", action=argparse.BooleanOptionalAction, default=True, help="Prevent non-emergency corridor slowdowns from freezing the drone")
    p.add_argument("--corridor-crawl-speed-m-s", type=float, default=0.12)

    # v32 dead-end escape / breadcrumb backtracking.
    p.add_argument("--dead-end-escape", action=argparse.BooleanOptionalAction, default=True, help="Detect true dead-end corridors and backtrack along recorded breadcrumbs instead of scanning/climbing repeatedly")
    p.add_argument("--dead-end-front-m", type=float, default=1.75, help="Center clearance below this can be considered a dead-end front wall")
    p.add_argument("--dead-end-side-open-m", type=float, default=2.15, help="If either side is more open than this, treat the blockage as bypassable rather than a dead end")
    p.add_argument("--dead-end-backtrack-min-m", type=float, default=3.2)
    p.add_argument("--dead-end-backtrack-max-m", type=float, default=8.0)
    p.add_argument("--dead-end-backtrack-spacing-m", type=float, default=1.8)
    p.add_argument("--dead-end-backtrack-max-targets", type=int, default=1)
    p.add_argument("--dead-end-backtrack-speed-m-s", type=float, default=0.66)
    p.add_argument("--dead-end-backtrack-timeout-s", type=float, default=10.0)
    p.add_argument("--dead-end-backtrack-waypoint-radius-m", type=float, default=1.05)
    p.add_argument("--dead-end-memory-ttl-s", type=float, default=55.0, help="How long to avoid re-entering a confirmed dead-end direction")
    p.add_argument("--dead-end-memory-distance-m", type=float, default=4.2)
    p.add_argument("--dead-end-memory-radius-m", type=float, default=1.0)
    p.add_argument("--dead-end-memory-check-m", type=float, default=4.5)
    p.add_argument("--dead-end-memory-max-hits", type=int, default=2)
    p.add_argument("--dead-end-turn-first-angle-deg", type=float, default=35.0)
    p.add_argument("--avoid-recent-path", action=argparse.BooleanOptionalAction, default=True, help="Normal exploration avoids reusing the recent breadcrumb trail; dead-end escape may still backtrack explicitly")
    p.add_argument("--allow-recent-path-fallback", action=argparse.BooleanOptionalAction, default=False, help="Allow global frontier planner to reuse recent path when no fresh route is found. Keep disabled for competition exploration.")
    p.add_argument("--recent-path-lookback-m", type=float, default=18.0, help="Length of recent breadcrumb trail to avoid during normal exploration")
    p.add_argument("--recent-path-radius-m", type=float, default=0.90, help="Radius around recent breadcrumbs considered already travelled")
    p.add_argument("--recent-path-exclude-current-m", type=float, default=1.8, help="Do not treat cells close to current pose as backtracking")
    p.add_argument("--recent-path-max-hits", type=int, default=0, help="Strict planner skips paths with more than this many recent cells")
    p.add_argument("--recent-path-penalty", type=float, default=10.0, help="Soft cost for paths that reuse recent cells when fallback is enabled")
    p.add_argument("--turnback-penalty", type=float, default=5.0, help="Penalty for frontier plans requiring a large yaw reversal")
    p.add_argument("--recent-heading-lookahead-m", type=float, default=4.0)
    p.add_argument("--recent-heading-max-hits", type=int, default=2)
    p.add_argument("--local-recent-penalty-block", type=float, default=2.2, help="Local centreline headings with visited penalty above this are considered backtracking")
    p.add_argument("--ghost-clearance-enabled", action=argparse.BooleanOptionalAction, default=True, help="Verify map-only blockages with a settled depth look and clear ghost cells if the route is open")
    p.add_argument("--ghost-clear-center-m", type=float, default=2.3)
    p.add_argument("--ghost-clear-min-side-m", type=float, default=0.55)
    p.add_argument("--ghost-clear-radius-m", type=float, default=1.05)
    p.add_argument("--ghost-clear-amount", type=int, default=10)
    p.add_argument("--ghost-verify-max-turn-deg", type=float, default=95.0)
    p.add_argument("--ghost-verify-yaw-timeout-s", type=float, default=0.9)
    p.add_argument("--ghost-verify-yaw-tolerance-deg", type=float, default=14.0)
    p.add_argument("--dead-end-escape-gap-s", type=float, default=2.0)
    p.add_argument("--history-record-every-m", type=float, default=0.70)
    p.add_argument("--history-record-gap-s", type=float, default=0.35)
    p.add_argument("--history-max-points", type=int, default=450)
    p.add_argument("--zone-loop-escape", action=argparse.BooleanOptionalAction, default=True, help="v38: detect repeated room loops and force a gateway/unknown-space scan")
    p.add_argument("--zone-loop-history-points", type=int, default=70)
    p.add_argument("--zone-loop-min-travel-m", type=float, default=36.0)
    p.add_argument("--zone-loop-max-span-m", type=float, default=12.0)
    p.add_argument("--zone-loop-close-m", type=float, default=2.0)
    p.add_argument("--zone-loop-min-current-visit", type=int, default=14)
    p.add_argument("--zone-loop-cooldown-s", type=float, default=26.0)
    p.add_argument("--progress-commitment", action=argparse.BooleanOptionalAction, default=True, help="After choosing a gateway/centreline/path, suppress non-essential rescans until the drone has moved a useful distance or the timer expires")
    p.add_argument("--global-path-priority", action=argparse.BooleanOptionalAction, default=True, help="v43: once an A* route to a high-information frontier exists, follow it before side-gateway/centreline scans. This prevents start-corner decision loops.")
    p.add_argument("--progress-commit-min-m", type=float, default=5.0, help="Distance to travel before non-essential gateway/zone-loop rescans are allowed again")
    p.add_argument("--progress-commit-s", type=float, default=10.0, help="Maximum time for a progress commitment before rescans are allowed again")
    p.add_argument("--local-corridor-hard-block-m", type=float, default=1.15, help="Only rescan a committed centreline immediately if forward clearance is below this or a supported front obstacle is present")

    # Photo collection policy.
    p.add_argument("--photo-interval-s", type=float, default=10.0, help="Minimum gap for sparse context photos")
    p.add_argument("--max-photo-gap-s", type=float, default=35.0, help="Force a context photo after this many seconds")
    p.add_argument("--sparse-photo-every-m", type=float, default=5.0)
    p.add_argument("--save-sparse-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--interest-min-pixels", type=int, default=180)
    p.add_argument("--interesting-save-gap-s", type=float, default=0.65)
    p.add_argument("--enable-interest-sweep", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--interest-sweep-gap-s", type=float, default=20.0)

    # YOLO / counting.
    p.add_argument("--detect-interval-s", type=float, default=0.60, help="Run YOLO at this interval in count mode")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold. Lower catches more barrels; higher reduces false positives")
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=50, help="Maximum YOLO detections per frame")
    p.add_argument("--yolo-imgsz", type=int, default=480, help="YOLO inference image size. Lower is faster; use 640 if small/distant barrels are missed.")
    p.add_argument("--merge-radius-m", type=float, default=0.70, help="Normal track merge radius in metres. Larger reduces double-counting; smaller separates close barrels")
    p.add_argument("--duplicate-suppression-radius-m", type=float, default=1.10, help="Extra radius used to suppress duplicate tracks near an already confirmed barrel")
    p.add_argument("--confirmed-merge-radius-m", type=float, default=0.45, help="Always merge same-colour tracks closer than this distance, even if both are confirmed")
    p.add_argument("--min-track-hits", type=int, default=2, help="Minimum detections required before a track contributes to final count. Use 2 to suppress one-frame duplicates")
    p.add_argument("--count-without-depth", action=argparse.BooleanOptionalAction, default=False, help="If depth is missing in a detection box, count using an approximate default depth. Useful as a last resort, but can duplicate tracks")
    p.add_argument("--default-detection-depth-m", type=float, default=4.0, help="Approximate depth used only when --count-without-depth is enabled")
    p.add_argument("--count-with-bbox-depth", action=argparse.BooleanOptionalAction, default=True, help="When the depth crop is invalid, estimate range from barrel bbox height so valid YOLO detections can still be tracked. This is safer than a fixed default depth and fixes missing yellow tracks near edges/floor seams.")
    p.add_argument("--barrel-physical-height-m", type=float, default=1.05, help="Approximate physical barrel height used by --count-with-bbox-depth")
    p.add_argument("--colour-override", action=argparse.BooleanOptionalAction, default=True, help="Use the detected crop colour as a safety net if YOLO class names/IDs disagree with the visible red/yellow colour")
    p.add_argument("--colour-override-min-pixels", type=int, default=80, help="Minimum crop pixels for red/yellow colour override")
    p.add_argument("--colour-override-ratio", type=float, default=1.20, help="Dominance ratio required before crop colour overrides YOLO class grouping")
    p.add_argument("--red-classes", nargs="*", default=["red", "red_barrel", "barrel_red", "red-barrel"])
    p.add_argument("--yellow-classes", nargs="*", default=["yellow", "yellow_barrel", "barrel_yellow", "yellow-barrel"])
    p.add_argument("--red-class-ids", nargs="*", type=int, default=[0], help="YOLO class IDs that should count as red barrels")
    p.add_argument("--yellow-class-ids", nargs="*", type=int, default=[1], help="YOLO class IDs that should count as yellow barrels")
    p.add_argument("--no-annotated", action="store_true")
    p.add_argument("--speedrun", action=argparse.BooleanOptionalAction, default=True, help="Use aggressive coverage-first timing defaults for a 5-minute run")

    return p


async def async_main() -> None:
    args = build_arg_parser().parse_args()
    mission = QualifierMission(args)
    await mission.run()


if __name__ == "__main__":
    asyncio.run(async_main())
