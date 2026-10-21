"""Run YOLO image classification on a video source and evaluate a classifier."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "yolo_classification"
DEFAULT_SOURCE = "0"
DEFAULT_OUTPUT = PROJECT_ROOT / "model_testing_random"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# OpenCV colors use BGR channel order.
HEALTH_COLORS = {
    "healthy": (0, 200, 0),
    "red beetle": (0, 0, 255),
    "yellowing": (0, 220, 255),
    "black beetle": (0, 220, 255),
}
DEFAULT_COLOR = (0, 220, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coconut-tree YOLO classification and test-set evaluation."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video file path, stream URL, or camera index (default: 0).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"YOLO weights file (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=(
            "YOLO classification dataset directory (default: "
            f"{DEFAULT_DATA}). Prepare it with src/prepare_yolo_dataset.py first."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder for annotated image tests (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def parse_video_source(source: str) -> Union[int, str]:
    """Interpret a numeric source as a camera index; preserve paths and URLs."""
    try:
        return int(source)
    except ValueError:
        return source


def color_for_label(label: str) -> Tuple[int, int, int]:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    for keyword, color in HEALTH_COLORS.items():
        if keyword in normalized:
            return color
    return DEFAULT_COLOR


def draw_classification(frame: Any, result: Any) -> None:
    """Overlay class scores and a color-coded border for whole-image health."""
    details = classification_details(result)
    height, width = frame.shape[:2]
    border_color = (
        HEALTH_COLORS["healthy"]
        if details["health_status"] == "HEALTHY"
        else HEALTH_COLORS["red beetle"]
    )
    border_thickness = max(2, min(height, width) // 180)
    cv2.rectangle(
        frame,
        (0, 0),
        (width - 1, height - 1),
        border_color,
        border_thickness,
    )

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, min(0.7, width / 1200))
    text_thickness = max(1, int(font_scale * 2))
    line_gap = max(22, int(30 * font_scale / 0.6))
    prediction_color = color_for_label(details["predicted_class"])
    lines = [
        ("COCONUT TREE HEALTH", (255, 255, 255)),
        (
            "Prediction: {} ({:.1f}%)".format(
                details["predicted_class"], details["confidence"] * 100
            ),
            prediction_color,
        ),
        (
            "Overall health: {} ({:.1f}%)".format(
                details["health_status"], details["health_probability"] * 100
            ),
            border_color,
        ),
        ("Damaged: {:.1f}%".format(details["damaged_probability"] * 100), (255, 255, 255)),
    ]
    lines.extend(
        (
            "{}: {:.1f}%".format(class_name, probability * 100),
            color_for_label(class_name),
        )
        for class_name, probability in details["class_probabilities"].items()
    )

    padding = 12
    text_sizes = [
        cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        for text, _ in lines
    ]
    panel_width = min(width - 2 * border_thickness, max(size[0] for size in text_sizes) + 2 * padding)
    panel_height = min(
        height - 2 * border_thickness,
        len(lines) * line_gap + 2 * padding,
    )
    if panel_width <= 0 or panel_height <= 0:
        return

    panel_x = border_thickness + 4
    panel_y = border_thickness + 4
    panel_region = frame[
        panel_y : panel_y + panel_height,
        panel_x : panel_x + panel_width,
    ]
    shade = panel_region.copy()
    cv2.rectangle(shade, (0, 0), (panel_width - 1, panel_height - 1), (25, 25, 25), -1)
    cv2.addWeighted(shade, 0.82, panel_region, 0.18, 0, panel_region)

    for line_number, ((text, color), (text_width, text_height)) in enumerate(
        zip(lines, text_sizes)
    ):
        text_y = panel_y + padding + line_number * line_gap + text_height
        if text_y >= panel_y + panel_height:
            break
        cv2.putText(
            frame,
            text,
            (panel_x + padding, text_y),
            font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )


def classification_details(result: Any) -> Dict[str, Any]:
    """Return class scores plus healthy-versus-damaged confidence."""
    probabilities = result.probs
    if probabilities is None:
        raise RuntimeError(
            "The loaded model did not return classification probabilities. "
            "Use YOLO classification weights for this dataset."
        )

    class_scores = probabilities.data.tolist()
    class_probabilities = {
        str(result.names[class_id]): float(score)
        for class_id, score in enumerate(class_scores)
    }
    healthy_probability = sum(
        probability
        for class_name, probability in class_probabilities.items()
        if "healthy" in class_name.lower()
    )
    damaged_probability = sum(
        probability
        for class_name, probability in class_probabilities.items()
        if "healthy" not in class_name.lower()
    )
    health_status = (
        "HEALTHY" if healthy_probability >= damaged_probability else "DAMAGED"
    )

    return {
        "predicted_class": str(result.names[int(probabilities.top1)]),
        "confidence": float(probabilities.top1conf.item()),
        "health_status": health_status,
        "health_probability": max(healthy_probability, damaged_probability),
        "healthy_probability": healthy_probability,
        "damaged_probability": damaged_probability,
        "class_probabilities": class_probabilities,
    }


def process_images(model: YOLO, source: Path, output_dir: Path) -> int:
    """Classify image files and export annotated copies plus a prediction CSV."""
    source = source.resolve()
    output_dir = output_dir.resolve()
    source_directory = source if source.is_dir() else source.parent
    if output_dir == source_directory:
        raise ValueError(
            "The output folder must be different from the input image folder "
            "so the original images are not overwritten."
        )

    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        results = model.predict(frame, verbose=False)
        if not results:
            raise RuntimeError(f"The model returned no prediction for: {image_path}")

        result = results[0]
        probabilities = result.probs
        if probabilities is None:
            raise RuntimeError(
                "The loaded model did not return classification probabilities. "
                "Use YOLO classification weights for image predictions."
            )

        details = classification_details(result)
        draw_classification(frame, result)
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Unable to save annotated image: {output_path}")

        record = {
            "image": image_path.name,
            "predicted_class": details["predicted_class"],
            "confidence": details["confidence"],
            "health_status": details["health_status"],
            "health_probability": details["health_probability"],
            "healthy_probability": details["healthy_probability"],
            "damaged_probability": details["damaged_probability"],
            "annotated_image": output_path.name,
        }
        record.update(
            {
                f"probability_{class_name}": score
                for class_name, score in details["class_probabilities"].items()
            }
        )
        records.append(record)
        print(
            f"{image_path.name}: {record['predicted_class']} "
            f"({record['confidence']:.1%})"
        )

    csv_path = output_dir / "predictions.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} annotated image(s) and predictions to: {output_dir}")
    print(f"Prediction details: {csv_path}")
    return len(records)


def select_tree_region(frame: Any, window_name: str) -> Tuple[int, int, int, int]:
    """Let the operator select one tree in the current video frame."""
    selection_frame = frame.copy()
    cv2.putText(
        selection_frame,
        "Drag a box around one tree, then press ENTER or SPACE",
        (12, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, selection_frame)
    x, y, width, height = cv2.selectROI(
        window_name, selection_frame, showCrosshair=True, fromCenter=False
    )
    if width <= 0 or height <= 0:
        raise RuntimeError(
            "No tree was selected. Drag a valid region and press Enter or Space; "
            "press Esc to cancel."
        )
    return int(x), int(y), int(width), int(height)


def features_in_region(gray_frame: Any, bbox: Tuple[int, int, int, int]) -> Any:
    """Find optical-flow features inside the selected tree region."""
    x, y, width, height = bbox
    mask = gray_frame.copy()
    mask[:] = 0
    cv2.rectangle(mask, (x, y), (x + width - 1, y + height - 1), 255, -1)
    return cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=7,
    )


def track_tree_region(
    previous_gray: Any,
    current_gray: Any,
    bbox: Tuple[int, int, int, int],
    previous_points: Any,
) -> Tuple[Optional[Tuple[int, int, int, int]], Any]:
    """Follow the selected region with pyramidal Lucas-Kanade optical flow."""
    if previous_points is None or len(previous_points) < 3:
        return None, None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if next_points is None or status is None:
        return None, None

    valid = status.reshape(-1) == 1
    old_valid = previous_points.reshape(-1, 2)[valid]
    new_valid = next_points.reshape(-1, 2)[valid]
    if len(new_valid) < 3:
        return None, None

    displacements = new_valid - old_valid
    dx, dy = displacements.mean(axis=0)
    x, y, width, height = bbox
    frame_height, frame_width = current_gray.shape[:2]
    next_x = max(0, min(frame_width - width, int(round(x + dx))))
    next_y = max(0, min(frame_height - height, int(round(y + dy))))
    if width > frame_width or height > frame_height:
        return None, None

    next_bbox = (next_x, next_y, width, height)
    tracked_points = new_valid.reshape(-1, 1, 2).astype("float32")
    tracked_points = tracked_points[:100]
    fresh_points = features_in_region(current_gray, next_bbox)
    remaining_slots = max(0, 100 - len(tracked_points))
    if fresh_points is not None and remaining_slots:
        fresh_points = fresh_points.astype("float32")
        fresh_points = fresh_points[:remaining_slots]
        next_points = cv2.vconcat((tracked_points, fresh_points))
    else:
        next_points = tracked_points

    return next_bbox, next_points


def process_video(model: YOLO, source: str) -> None:
    video_source = parse_video_source(source)
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {source}")

    window_name = "Coconut Tree Health - select one tree"
    try:
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Unable to read the first frame from: {source}")

        bbox = select_tree_region(frame, window_name)
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = features_in_region(previous_gray, bbox)
        if points is None or len(points) < 3:
            raise RuntimeError(
                "Could not find enough visual features in the selected region to track. "
                "Select a larger, well-lit tree area with visible texture."
            )

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                next_bbox, next_points = track_tree_region(
                    previous_gray, current_gray, bbox, points
                )
                if next_bbox is None:
                    cv2.putText(
                        frame,
                        "TRACK LOST - press R to select the tree again, Q to quit",
                        (12, 32),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("r"):
                        bbox = select_tree_region(frame, window_name)
                        points = features_in_region(current_gray, bbox)
                        if points is None or len(points) < 3:
                            raise RuntimeError(
                                "Could not find enough visual features in the new "
                                "selection. Choose a larger, textured tree region."
                            )
                    previous_gray = current_gray
                    continue

                bbox, points = next_bbox, next_points
                x, y, width, height = bbox
                tree_crop = frame[y : y + height, x : x + width]
                if tree_crop.size == 0:
                    raise RuntimeError("The tracked tree region is outside the frame.")

                started_at = time.perf_counter()
                results = model.predict(tree_crop, verbose=False)
                inference_ms = (time.perf_counter() - started_at) * 1000
                inference_fps = 1000 / inference_ms if inference_ms > 0 else 0.0
                if not results:
                    raise RuntimeError("The model returned no prediction for the tree.")

                details = classification_details(results[0])
                draw_classification(tree_crop, results[0])
                tree_color = (
                    HEALTH_COLORS["healthy"]
                    if details["health_status"] == "HEALTHY"
                    else HEALTH_COLORS["red beetle"]
                )
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    tree_color,
                    max(2, min(width, height) // 120),
                )
                tag = "{} {:.1f}%".format(
                    details["health_status"], details["health_probability"] * 100
                )
                cv2.putText(
                    frame,
                    tag,
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    tree_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{inference_fps:.1f} FPS | {inference_ms:.1f} ms/frame | R: reselect | Q: quit",
                    (12, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    bbox = select_tree_region(frame, window_name)
                    points = features_in_region(current_gray, bbox)
                    if points is None or len(points) < 3:
                        raise RuntimeError(
                            "Could not find enough visual features in the new "
                            "selection. Choose a larger, textured tree region."
                        )
                previous_gray = current_gray
        finally:
            capture.release()
            cv2.destroyAllWindows()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def evaluate_model(model: YOLO, dataset_dir: Path) -> Dict[str, float]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"YOLO classification dataset was not found: {dataset_dir}. "
            "Run src/prepare_yolo_dataset.py before evaluation."
        )

    test_dir = dataset_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split was not found: {test_dir}")

    metrics = model.val(data=str(dataset_dir), split="test")
    top1 = getattr(metrics, "top1", None)
    top5 = getattr(metrics, "top5", None)
    if top1 is None or top5 is None:
        raise RuntimeError(
            "Validation did not return classification accuracy metrics. "
            "Check that the weights and dataset use the classification task."
        )

    return {
        "Top-1 Accuracy": float(top1),
        "Top-5 Accuracy": float(top5),
    }


def print_summary(metrics: Optional[Dict[str, float]]) -> None:
    print("\nTest Dataset Classification Metrics")
    print("+----------------+---------+")
    print("| Metric         | Value   |")
    print("+----------------+---------+")
    for name in ("Top-1 Accuracy", "Top-5 Accuracy"):
        value = f"{metrics[name]:.4f}" if metrics is not None else "N/A"
        print(f"| {name:<14} | {value:>7} |")
    print("+----------------+---------+")


def main() -> int:
    args = parse_args()
    weights_path = args.weights.expanduser().resolve()
    dataset_dir = args.data.expanduser().resolve()
    source_path = Path(args.source).expanduser()
    output_dir = args.output.expanduser().resolve()

    if not weights_path.is_file():
        print(f"Error: model weights were not found: {weights_path}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(weights_path))
        if source_path.is_dir():
            process_images(model, source_path.resolve(), output_dir)
        elif source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            process_images(model, source_path.resolve(), output_dir)
        else:
            process_video(model, args.source)
        metrics = evaluate_model(model, dataset_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        print_summary(None)
        return 1
    finally:
        cv2.destroyAllWindows()

    print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
