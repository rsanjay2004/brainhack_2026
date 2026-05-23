#!/usr/bin/env python3
"""Run barrel_best.pt YOLO on images and save annotated copies.

Examples:
    python3 detect_barrels.py img.jpg
    python3 detect_barrels.py path/to/folder
    python3 detect_barrels.py "path/*.png" --conf 0.25 --imgsz 640
    python3 detect_barrels.py img.jpg --model /other/model.pt --output-dir out
"""

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default class colours (BGR) for box drawing.
CLASS_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}
DEFAULT_BGR = (0, 255, 0)


def collect_images(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        # Expand glob patterns first.
        matches = glob.glob(raw)
        if matches:
            candidates = [Path(m) for m in matches]
        else:
            candidates = [Path(raw)]
        for c in candidates:
            c = c.expanduser().resolve()
            if c.is_dir():
                for ext in IMAGE_EXTS:
                    paths.extend(sorted(c.rglob(f"*{ext}")))
            elif c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                paths.append(c)
            else:
                print(f"[WARN] skipped (not an image / not found): {c}")
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def annotate(image, boxes, names: dict, conf_thresh: float) -> int:
    drawn = 0
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        colour = CLASS_BGR.get(name.lower().split("_")[0], DEFAULT_BGR)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(image, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Image files, folders, or glob patterns")
    ap.add_argument("--model", default="models/barrel_best.pt", help="YOLO weights path")
    ap.add_argument("--output-dir", default="detect_barrels_output", help="Where to write annotated images + CSV")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    ap.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    ap.add_argument("--device", default="", help="Inference device (e.g. cpu, 0, cuda:0). Empty = auto")
    ap.add_argument("--no-csv", action="store_true", help="Skip per-detection CSV output")
    ap.add_argument("--show-empty", action="store_true", help="Also save images where no barrels were detected")
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"[ERR] model not found: {model_path}")
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.inputs)
    if not images:
        print("[ERR] no images to process")
        return 1

    print(f"[INFO] model={model_path}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] images={len(images)} conf={args.conf} iou={args.iou} imgsz={args.imgsz}")

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    print(f"[INFO] classes={names}")

    csv_path = out_dir / "detections.csv"
    csv_file = None
    csv_writer = None
    if not args.no_csv:
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    totals = {name: 0 for name in names.values()}
    images_with_detections = 0

    try:
        results = model.predict(
            source=[str(p) for p in images],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device if args.device else None,
            stream=True,
            verbose=False,
        )
        for src_path, result in zip(images, results):
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"[WARN] could not read: {src_path}")
                continue
            boxes = result.boxes if result.boxes is not None else []
            drawn = annotate(img, boxes, names, args.conf)
            if drawn > 0:
                images_with_detections += 1
            if drawn == 0 and not args.show_empty:
                continue

            out_path = out_dir / f"{src_path.stem}_annotated{src_path.suffix}"
            counter = 1
            while out_path.exists():
                out_path = out_dir / f"{src_path.stem}_annotated_{counter}{src_path.suffix}"
                counter += 1
            cv2.imwrite(str(out_path), img)

            line_counts = {}
            for box in boxes:
                conf = float(box.conf[0])
                if conf < args.conf:
                    continue
                cls_id = int(box.cls[0])
                name = names.get(cls_id, str(cls_id))
                totals[name] = totals.get(name, 0) + 1
                line_counts[name] = line_counts.get(name, 0) + 1
                if csv_writer is not None:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    csv_writer.writerow([str(src_path), cls_id, name, f"{conf:.4f}", x1, y1, x2, y2])

            summary = ", ".join(f"{k}={v}" for k, v in sorted(line_counts.items())) or "no detections"
            print(f"[OK] {src_path.name} -> {out_path.name} ({summary})")
    finally:
        if csv_file is not None:
            csv_file.close()

    print("---")
    print(f"[DONE] processed={len(images)} with_detections={images_with_detections}")
    for name, count in sorted(totals.items()):
        print(f"[COUNT] {name}: {count}")
    if csv_writer is not None:
        print(f"[CSV] {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
