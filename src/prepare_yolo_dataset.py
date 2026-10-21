"""Convert the project's one-hot CSV image labels to Ultralytics classification folders."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "dataset"
DEFAULT_DESTINATION = PROJECT_ROOT / "dataset" / "yolo_classification"
SPLIT_NAMES = ("train", "valid", "test")
SPLIT_DESTINATIONS = {"train": "train", "valid": "val", "test": "test"}


def read_labels(csv_path: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"Expected a 'filename' column in {csv_path}")

        class_names = [name for name in reader.fieldnames if name != "filename"]
        if not class_names:
            raise ValueError(f"No class columns were found in {csv_path}")

        labeled_files = []
        for row_number, row in enumerate(reader, start=2):
            selected = [
                class_name
                for class_name in class_names
                if (row.get(class_name) or "").strip() == "1"
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"{csv_path}:{row_number} must have exactly one class label set "
                    f"to 1; found {len(selected)}."
                )

            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"{csv_path}:{row_number} has an empty filename.")
            labeled_files.append((filename, selected[0]))

    return class_names, labeled_files


def convert_dataset(source: Path, destination: Path) -> Dict[str, int]:
    counts = {}
    expected_classes = None

    for split in SPLIT_NAMES:
        split_dir = source / split
        csv_path = split_dir / "_classes.csv"
        if not split_dir.is_dir() or not csv_path.is_file():
            raise FileNotFoundError(
                f"Expected split directory and labels file: {split_dir} "
                "with _classes.csv"
            )

        class_names, labeled_files = read_labels(csv_path)
        if expected_classes is None:
            expected_classes = class_names
        elif class_names != expected_classes:
            raise ValueError(
                f"Class columns differ between splits: {csv_path} "
                f"contains {class_names}, expected {expected_classes}."
            )

        for filename, class_name in labeled_files:
            source_image = split_dir / filename
            if not source_image.is_file():
                raise FileNotFoundError(f"Image listed in labels was not found: {source_image}")

            target_dir = destination / SPLIT_DESTINATIONS[split] / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_dir / filename)

        counts[split] = len(labeled_files)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy CSV-labeled images into the train/<class>, val/<class>, "
            "test/<class> layout used for Ultralytics YOLO classification."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Dataset directory containing train/valid/test (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Converted dataset directory (default: {DEFAULT_DESTINATION}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.output.expanduser().resolve()

    source_split_dirs = {source / split for split in SPLIT_NAMES}
    if destination == source or destination in source.parents or destination in source_split_dirs:
        print(
            "Error: output directory must not be the source dataset or one of its split directories.",
            file=sys.stderr,
        )
        return 1

    try:
        counts = convert_dataset(source, destination)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"YOLO classification dataset created at: {destination}")
    print("Ultralytics split folder mapping:")
    for split, count in counts.items():
        print(f"  {SPLIT_DESTINATIONS[split]}: {count} images")
    print("Each split contains one subfolder per class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
