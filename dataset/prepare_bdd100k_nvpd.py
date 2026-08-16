#!/usr/bin/env python3
"""Build and optionally publish the NVPD BDD100K dataset in YOLO format.

The source BDD100K train split is sampled into balanced NVPD train/validation
splits.  Every daytime/night frame from the source BDD100K validation split is
used as the NVPD test split.  The output is directly consumable by Ultralytics:

    output/
      data.yaml
      images/{train,val,test}/
      labels/{train,val,test}/

The ``train`` and ``traffic sign`` object categories are intentionally ignored,
and ``rider`` is merged into ``person``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        """Fallback iterator when optional tqdm progress bars are unavailable."""
        return iterable


DEFAULT_DATASET_SLUG = "solesensei/solesensei_bdd100k"
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DATASET_SLUG = "nvpdyf-bdd100k"
DEFAULT_OUTPUT_DATASET_TITLE = "nvpdyf_bdd100k"

# Final contiguous Ultralytics class ids used by the current notebook.
YOLO_CLASS_NAMES = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorcycle",
    4: "person",
    5: "traffic light",
    6: "truck",
}

YOLO_LABEL_TO_ID = {
    class_name: class_id for class_id, class_name in YOLO_CLASS_NAMES.items()
}

BDD_TO_YOLO_LABEL = {
    "bicycle": "bicycle",
    "bike": "bicycle",
    "bus": "bus",
    "car": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "person": "person",
    "pedestrian": "person",
    "rider": "person",
    "traffic light": "traffic light",
    "truck": "truck",
}

# These categories can have valid box2d annotations but are excluded by design.
IGNORED_BOX_CATEGORIES = {"traffic sign", "train"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_FRAME_COLUMNS = {
    "name",
    "labels",
    "scene",
    "weather",
    "timeofday",
}

LOGGER = logging.getLogger("bdd100k_nvpd")


@dataclass(frozen=True)
class DatasetLayout:
    """Paths to the BDD100K images and frame annotation files."""

    train_images_dir: Path
    val_images_dir: Path
    train_labels_path: Path
    val_labels_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download BDD100K, build the NVPD splits in Ultralytics YOLO "
            "format, and optionally publish them as a Kaggle Dataset."
        )
    )
    parser.add_argument(
        "--dataset-slug",
        default=DEFAULT_DATASET_SLUG,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET_SLUG}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("bdd100k_raw"),
        help="Directory for the downloaded source dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NVPD_UPDATE"),
        help="Directory for the YOLO dataset (default: NVPD_UPDATE).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed used for sampling (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads used for annotation processing and image copying.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an already downloaded dataset from --raw-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Pass --force to the Kaggle download command.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and rebuild a non-empty output directory.",
    )
    parser.add_argument(
        "--output-dataset-slug",
        default=DEFAULT_OUTPUT_DATASET_SLUG,
        help=(
            "Kaggle slug for the generated dataset "
            f"(default: {DEFAULT_OUTPUT_DATASET_SLUG})."
        ),
    )
    parser.add_argument(
        "--output-dataset-title",
        default=DEFAULT_OUTPUT_DATASET_TITLE,
        help=(
            "Kaggle title for the generated dataset "
            f"(default: {DEFAULT_OUTPUT_DATASET_TITLE})."
        ),
    )
    parser.add_argument(
        "--kaggle-owner",
        default=os.environ.get("KAGGLE_USERNAME"),
        help="Kaggle username/organization. Defaults to KAGGLE_USERNAME.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the prepared output with the Kaggle CLI.",
    )
    parser.add_argument(
        "--publish-mode",
        choices=("auto", "create", "version"),
        default="auto",
        help=(
            "Kaggle publication action. 'auto' creates a missing dataset and "
            "versions an existing one (default: auto)."
        ),
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make a newly created Kaggle Dataset public.",
    )
    parser.add_argument(
        "--version-message",
        default=(
            "Rebuilt Ultralytics YOLO dataset; test contains every daytime/night "
            "BDD100K validation image"
        ),
        help="Version message used when updating an existing Kaggle Dataset.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args(argv)

    if args.skip_download and args.force_download:
        parser.error("--skip-download and --force-download cannot be used together.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.random_state < 0:
        parser.error("--random-state must be non-negative.")
    if not args.output_dataset_slug or "/" in args.output_dataset_slug:
        parser.error("--output-dataset-slug must be one non-empty path segment.")
    if args.publish and not args.kaggle_owner:
        parser.error("--publish requires --kaggle-owner or KAGGLE_USERNAME.")

    return args


def configure_logging(level: str) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_working_paths(raw_dir: Path, output_dir: Path) -> None:
    """Reject path combinations that could mix source and output data."""
    raw_resolved = raw_dir.resolve()
    output_resolved = output_dir.resolve()

    if raw_resolved == output_resolved:
        raise ValueError("The raw and output directories must be different.")
    if raw_resolved in output_resolved.parents:
        raise ValueError("The output directory cannot be inside the raw directory.")
    if output_resolved in raw_resolved.parents:
        raise ValueError("The raw directory cannot be inside the output directory.")


def run_command(command: Sequence[str]) -> None:
    """Run an external command and raise an error if it fails."""
    LOGGER.debug("Running command: %s", " ".join(command))
    subprocess.run(list(command), check=True)


def download_kaggle_dataset(
    dataset_slug: str,
    destination_dir: Path,
    force: bool = False,
) -> None:
    """Download and extract a Kaggle dataset with the Kaggle CLI."""
    kaggle_executable = shutil.which("kaggle")
    if kaggle_executable is None:
        raise RuntimeError(
            "The Kaggle CLI was not found. Install it with 'python -m pip install kaggle'."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    command = [
        kaggle_executable,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(destination_dir),
        "--unzip",
    ]
    if force:
        command.append("--force")

    LOGGER.info("Downloading Kaggle dataset %s", dataset_slug)
    run_command(command)


def find_unique_file(root_dir: Path, filename: str) -> Path:
    """Find exactly one file with the requested name under a directory."""
    matches = sorted(path for path in root_dir.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root_dir}.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple files named {filename}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def find_image_directory(root_dir: Path, split_name: str) -> Path:
    """Find a BDD100K images/100k split directory."""
    candidates = sorted(
        path
        for path in root_dir.rglob(split_name)
        if path.is_dir()
        and path.parent.name == "100k"
        and path.parent.parent.name == "images"
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find an images/100k/{split_name} directory under {root_dir}."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple images/100k/{split_name} directories: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def resolve_dataset_layout(raw_dir: Path) -> DatasetLayout:
    """Resolve source paths independently of the outer archive layout."""
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw dataset directory does not exist: {raw_dir}")

    return DatasetLayout(
        train_images_dir=find_image_directory(raw_dir, "train"),
        val_images_dir=find_image_directory(raw_dir, "val"),
        train_labels_path=find_unique_file(
            raw_dir, "bdd100k_labels_images_train.json"
        ),
        val_labels_path=find_unique_file(raw_dir, "bdd100k_labels_images_val.json"),
    )


def dataset_is_ready(raw_dir: Path) -> bool:
    """Return whether all required BDD100K paths are available."""
    try:
        resolve_dataset_layout(raw_dir)
    except (FileNotFoundError, NotADirectoryError, RuntimeError):
        return False
    return True


def flatten_image_directory(image_dir: Path) -> int:
    """Move files from nested folders to the split root and remove empty folders."""
    nested_files = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.parent != image_dir
    )

    for source_path in nested_files:
        destination_path = image_dir / source_path.name
        if destination_path.exists():
            raise FileExistsError(
                f"Cannot flatten {image_dir}; duplicate filename: {source_path.name}"
            )
        shutil.move(str(source_path), str(destination_path))

    nested_directories = sorted(
        (path for path in image_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in nested_directories:
        try:
            directory.rmdir()
        except OSError:
            pass

    return len(nested_files)


def load_annotation_dataframe(annotation_path: Path) -> pd.DataFrame:
    """Load BDD100K records, expand frame attributes, and remove timestamps."""
    with annotation_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {annotation_path}.")

    normalized_records: list[dict[str, Any]] = []
    for record_index, source_record in enumerate(records):
        if not isinstance(source_record, dict):
            raise TypeError(
                f"Record {record_index} in {annotation_path} is not an object."
            )

        record = copy.deepcopy(source_record)
        attributes = record.pop("attributes", {})
        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise TypeError(
                f"Frame attributes at record {record_index} are not an object."
            )

        conflicting_keys = set(record).intersection(attributes)
        if conflicting_keys:
            raise ValueError(
                f"Frame attributes conflict with top-level keys at record "
                f"{record_index}: {sorted(conflicting_keys)}"
            )

        record.update(attributes)
        record.pop("timestamp", None)
        normalized_records.append(record)

    dataframe = pd.DataFrame(normalized_records)
    require_columns(dataframe, REQUIRED_FRAME_COLUMNS, annotation_path.name)
    return dataframe


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataframe_name: str,
) -> None:
    """Validate that a DataFrame contains every required column."""
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing columns: {sorted(missing_columns)}"
        )


def normalize_category(category: str) -> str:
    """Normalize a BDD100K category before a mapping lookup."""
    return category.strip().lower().replace("_", " ")


def box2d_to_yolo(
    box2d: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert an absolute BDD100K box to normalized YOLO coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {image_width}x{image_height}."
        )

    x1 = float(box2d["x1"])
    y1 = float(box2d["y1"])
    x2 = float(box2d["x2"])
    y2 = float(box2d["y2"])

    x1 = max(0.0, min(x1, float(image_width)))
    x2 = max(0.0, min(x2, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    y2 = max(0.0, min(y2, float(image_height)))

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Bounding box has a non-positive size: {box2d}")

    return [
        (x1 + x2) / 2.0 / image_width,
        (y1 + y2) / 2.0 / image_height,
        box_width / image_width,
        box_height / image_height,
    ]


def build_image_index(image_dir: Path) -> dict[str, Path]:
    """Index split images by basename and reject ambiguous filenames."""
    index: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in index:
            raise RuntimeError(
                f"Duplicate image basename {path.name}: {index[path.name]} and {path}"
            )
        index[path.name] = path

    if not index:
        raise FileNotFoundError(f"No image files found under {image_dir}.")
    return index


def validate_frame_name(filename: str) -> str:
    """Reject absolute paths and path traversal in annotation filenames."""
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Unsafe image filename in annotations: {filename!r}")
    return filename


def build_frame_yolo_label(
    record: dict[str, Any],
    image_index: dict[str, Path],
) -> tuple[str, list[str], Counter[str]]:
    """Convert one BDD100K frame record into Ultralytics label rows."""
    filename = validate_frame_name(str(record["name"]))

    try:
        image_path = image_index[filename]
    except KeyError as error:
        raise FileNotFoundError(f"Image not found for annotation: {filename}") from error

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    labels = record.get("labels", [])
    if not isinstance(labels, list):
        labels = []

    lines: list[str] = []
    stats: Counter[str] = Counter()

    for object_index, source_object in enumerate(labels):
        if not isinstance(source_object, dict):
            stats["invalid_annotation"] += 1
            continue

        box2d = source_object.get("box2d")
        if box2d is None:
            # lane and drivable area contain poly2d rather than detection boxes.
            stats["without_box"] += 1
            continue

        source_category = source_object.get("category")
        if not isinstance(source_category, str):
            raise ValueError(
                f"{filename} object {object_index} has no string category."
            )

        normalized_category = normalize_category(source_category)
        if normalized_category in IGNORED_BOX_CATEGORIES:
            stats[f"ignored:{normalized_category}"] += 1
            continue

        yolo_label = BDD_TO_YOLO_LABEL.get(normalized_category)
        if yolo_label is None:
            raise ValueError(
                f"Unsupported boxed BDD100K category {source_category!r} "
                f"in {filename}, object {object_index}."
            )

        try:
            x_center, y_center, width, height = box2d_to_yolo(
                box2d,
                image_width=image_width,
                image_height=image_height,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid box2d in {filename}, object {object_index}: {error}"
            ) from error

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Non-finite YOLO coordinates in {filename}, object {object_index}."
            )
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Out-of-range YOLO coordinates in {filename}, object "
                f"{object_index}: {coordinates}"
            )

        class_id = YOLO_LABEL_TO_ID[yolo_label]
        lines.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )
        stats["objects"] += 1
        stats[f"class:{yolo_label}"] += 1
        if normalized_category == "rider":
            stats["rider_merged_into_person"] += 1

    if not lines:
        stats["empty_images"] += 1
    stats["images"] += 1
    return filename, lines, stats


def write_dataframe_yolo_labels(
    dataframe: pd.DataFrame,
    image_index: dict[str, Path],
    destination_dir: Path,
    workers: int,
    split_name: str,
) -> Counter[str]:
    """Write one Ultralytics ``.txt`` annotation for every split image."""
    records = dataframe.to_dict(orient="records")
    destination_dir.mkdir(parents=True, exist_ok=True)

    stem_owners: dict[str, str] = {}
    for filename in dataframe["name"].astype(str):
        normalized_stem = Path(filename).stem.casefold()
        previous_owner = stem_owners.get(normalized_stem)
        if previous_owner is not None:
            raise ValueError(
                f"{split_name} contains label-stem collision: "
                f"{previous_owner} and {filename}."
            )
        stem_owners[normalized_stem] = filename

    def process(
        record: dict[str, Any],
    ) -> tuple[str, list[str], Counter[str]]:
        return build_frame_yolo_label(record, image_index)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        converted_records = list(
            tqdm(
                executor.map(process, records),
                total=len(records),
                desc=f"Writing {split_name} YOLO labels",
            )
        )

    total_stats: Counter[str] = Counter()
    for filename, lines, frame_stats in converted_records:
        label_path = destination_dir / f"{Path(filename).stem}.txt"
        text = "\n".join(lines)
        if text:
            text += "\n"
        label_path.write_text(text, encoding="utf-8")
        total_stats.update(frame_stats)

    LOGGER.info("%s YOLO annotation stats: %s", split_name, dict(total_stats))
    return total_stats


def extract_stratification_tokens(row: pd.Series) -> list[str]:
    """Build multilabel tokens for scene, weather, and object categories."""
    tokens: list[str] = []

    scene = row.get("scene")
    weather = row.get("weather")
    if pd.notna(scene):
        tokens.append(f"scene:{scene}")
    if pd.notna(weather):
        tokens.append(f"weather:{weather}")

    labels = row.get("labels", [])
    if isinstance(labels, list):
        object_categories = {
            obj.get("category")
            for obj in labels
            if isinstance(obj, dict) and obj.get("category") is not None
        }
        tokens.extend(
            f"object:{category}" for category in sorted(object_categories)
        )

    return tokens


def random_partition_indices(
    total_samples: int,
    selected_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic random partition when no stratification tokens exist."""
    generator = np.random.default_rng(random_state)
    permutation = generator.permutation(total_samples)
    return permutation[:selected_size], permutation[selected_size:]


def multilabel_partition_indices(
    dataframe: pd.DataFrame,
    selected_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions while preserving multilabel feature diversity."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as error:
        raise RuntimeError(
            "The iterative-stratification package is required. Install it with "
            "'python -m pip install iterative-stratification'."
        ) from error

    total_samples = len(dataframe)
    remainder_size = total_samples - selected_size

    if selected_size <= 0 or remainder_size <= 0:
        raise ValueError(
            f"Both partition sizes must be positive, got {selected_size} and "
            f"{remainder_size}."
        )

    tokens = dataframe.apply(extract_stratification_tokens, axis=1)
    encoder = MultiLabelBinarizer()
    stratification_matrix = encoder.fit_transform(tokens)

    if stratification_matrix.shape[1] == 0:
        return random_partition_indices(
            total_samples=total_samples,
            selected_size=selected_size,
            random_state=random_state,
        )

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=selected_size,
        test_size=remainder_size,
        random_state=random_state,
    )
    selected_positions, remainder_positions = next(
        splitter.split(
            np.zeros((total_samples, 1)),
            stratification_matrix,
        )
    )
    return selected_positions, remainder_positions


def diverse_sample(
    dataframe: pd.DataFrame,
    n_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """Select rows while preserving scene, weather, and object diversity."""
    total_samples = len(dataframe)
    if not 0 <= n_samples <= total_samples:
        raise ValueError(f"Requested {n_samples} rows, but only {total_samples} exist.")
    if n_samples == 0:
        return dataframe.iloc[0:0].copy()
    if n_samples == total_samples:
        return dataframe.copy()

    selected_positions, _ = multilabel_partition_indices(
        dataframe=dataframe,
        selected_size=n_samples,
        random_state=random_state,
    )
    return dataframe.iloc[selected_positions].copy()


def diverse_train_val_split(
    dataframe: pd.DataFrame,
    train_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one time-of-day group into diverse train and validation parts."""
    train_positions, val_positions = multilabel_partition_indices(
        dataframe=dataframe,
        selected_size=train_size,
        random_state=random_state,
    )
    return (
        dataframe.iloc[train_positions].copy(),
        dataframe.iloc[val_positions].copy(),
    )


def assert_unique_names(dataframe: pd.DataFrame, dataframe_name: str) -> None:
    """Reject repeated image names."""
    duplicated = dataframe["name"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = (
            dataframe.loc[duplicated, "name"].astype(str).drop_duplicates().head().tolist()
        )
        raise ValueError(
            f"{dataframe_name} contains duplicate image names. Examples: {examples}"
        )


def create_balanced_train_val(
    dataframe: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact 80/20 day-night and 80/20 train-validation splits."""
    require_columns(dataframe, REQUIRED_FRAME_COLUMNS, "training annotations")
    filtered = (
        dataframe[dataframe["timeofday"].isin(["daytime", "night"])]
        .copy()
        .reset_index(drop=True)
    )
    assert_unique_names(filtered, "training annotations")

    daytime = filtered[filtered["timeofday"] == "daytime"].copy()
    night = filtered[filtered["timeofday"] == "night"].copy()

    max_night_count = min(len(night), len(daytime) // 4)
    selected_night_count = (max_night_count // 5) * 5
    selected_daytime_count = selected_night_count * 4
    if selected_night_count < 5:
        raise ValueError(
            "At least 20 daytime and 5 night images are required for exact ratios."
        )

    LOGGER.info(
        "Training source distribution: daytime=%d, night=%d",
        len(daytime),
        len(night),
    )

    selected_daytime = diverse_sample(
        daytime, selected_daytime_count, random_state
    )
    selected_night = diverse_sample(
        night, selected_night_count, random_state + 1
    )

    train_daytime, val_daytime = diverse_train_val_split(
        selected_daytime,
        train_size=selected_daytime_count * 4 // 5,
        random_state=random_state + 2,
    )
    train_night, val_night = diverse_train_val_split(
        selected_night,
        train_size=selected_night_count * 4 // 5,
        random_state=random_state + 3,
    )

    train_dataframe = (
        pd.concat([train_daytime, train_night], ignore_index=True)
        .sample(frac=1, random_state=random_state + 4)
        .reset_index(drop=True)
    )
    val_dataframe = (
        pd.concat([val_daytime, val_night], ignore_index=True)
        .sample(frac=1, random_state=random_state + 5)
        .reset_index(drop=True)
    )

    validate_exact_ratio(train_dataframe, daytime_parts=4, night_parts=1, name="train")
    validate_exact_ratio(val_dataframe, daytime_parts=4, night_parts=1, name="val")
    validate_disjoint_splits({"train": train_dataframe, "val": val_dataframe})

    if len(train_dataframe) != 4 * len(val_dataframe):
        raise RuntimeError("The train-to-validation ratio is not exactly 4:1.")

    LOGGER.info(
        "Balanced training splits: train=%d, val=%d",
        len(train_dataframe),
        len(val_dataframe),
    )
    return train_dataframe, val_dataframe


def create_test_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use every daytime/night frame from the BDD100K validation split."""
    require_columns(dataframe, REQUIRED_FRAME_COLUMNS, "test annotations")
    test_dataframe = (
        dataframe[dataframe["timeofday"].isin(["daytime", "night"])]
        .copy()
        .sort_values(by="name")
        .reset_index(drop=True)
    )
    assert_unique_names(test_dataframe, "test annotations")
    if test_dataframe.empty:
        raise ValueError(
            "BDD100K validation contains no daytime/night frames for test."
        )

    counts = test_dataframe["timeofday"].value_counts()
    LOGGER.info(
        "BDD100K validation test split (no sampling): daytime=%d, night=%d, "
        "total=%d",
        int(counts.get("daytime", 0)),
        int(counts.get("night", 0)),
        len(test_dataframe),
    )
    return test_dataframe


def validate_exact_ratio(
    dataframe: pd.DataFrame,
    daytime_parts: int,
    night_parts: int,
    name: str,
) -> None:
    """Validate an exact daytime-to-night ratio."""
    counts = dataframe["timeofday"].value_counts()
    daytime_count = int(counts.get("daytime", 0))
    night_count = int(counts.get("night", 0))
    if daytime_count * night_parts != night_count * daytime_parts:
        raise RuntimeError(
            f"{name} does not have the required {daytime_parts}:{night_parts} "
            f"daytime-to-night ratio."
        )


def validate_disjoint_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Validate that no image name occurs in more than one split."""
    names_by_split = {
        split_name: set(dataframe["name"].astype(str))
        for split_name, dataframe in splits.items()
    }
    split_names = list(names_by_split)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = names_by_split[left_name].intersection(names_by_split[right_name])
            if overlap:
                examples = sorted(overlap)[:5]
                raise RuntimeError(
                    f"Splits {left_name} and {right_name} overlap by {len(overlap)} "
                    f"images. Examples: {examples}"
                )


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create a clean output directory."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite-output to rebuild it."
            )
        protected_paths = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output_dir.resolve() in protected_paths:
            raise ValueError(f"Refusing to delete protected path: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, dataset_slug: str) -> Path:
    """Write an Ultralytics configuration targeting the Kaggle mount path."""
    data_config = {
        "path": f"/kaggle/input/{dataset_slug}",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": YOLO_CLASS_NAMES,
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    return yaml_path


def write_kaggle_metadata(
    output_dir: Path,
    owner: str,
    dataset_slug: str,
    dataset_title: str,
) -> Path:
    """Write metadata required by ``kaggle datasets create/version``."""
    metadata = {
        "title": dataset_title,
        "id": f"{owner}/{dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    metadata_path = output_dir / "dataset-metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return metadata_path


def copy_one_image(source_path: Path, destination_path: Path) -> None:
    """Copy one image with metadata."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def copy_dataframe_images(
    dataframe: pd.DataFrame,
    image_index: dict[str, Path],
    destination_dir: Path,
    workers: int,
    split_name: str,
) -> int:
    """Copy all images referenced by a split into a flat output directory."""
    filenames = list(dict.fromkeys(dataframe["name"].astype(str)))
    copy_jobs: list[tuple[Path, Path]] = []

    for filename in filenames:
        validate_frame_name(filename)
        try:
            source_path = image_index[filename]
        except KeyError as error:
            raise FileNotFoundError(
                f"Source image for {split_name} was not found: {filename}"
            ) from error
        copy_jobs.append((source_path, destination_dir / filename))

    destination_dir.mkdir(parents=True, exist_ok=True)

    def copy_job(job: tuple[Path, Path]) -> None:
        copy_one_image(*job)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(copy_job, copy_jobs),
                total=len(copy_jobs),
                desc=f"Copying {split_name} images",
            )
        )

    return len(copy_jobs)


def log_split_summary(
    split_name: str,
    dataframe: pd.DataFrame,
    annotation_stats: Counter[str],
) -> None:
    """Log final frame, time-of-day, and retained YOLO object counts."""
    time_counts = dataframe["timeofday"].value_counts().to_dict()
    LOGGER.info(
        "%s summary: images=%d, objects=%d, daytime=%d, night=%d",
        split_name,
        len(dataframe),
        annotation_stats["objects"],
        int(time_counts.get("daytime", 0)),
        int(time_counts.get("night", 0)),
    )


def validate_yolo_dataset(
    output_dir: Path,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Validate image/label correspondence and every Ultralytics label row."""
    valid_class_ids = set(YOLO_CLASS_NAMES)

    for split_name, dataframe in splits.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name

        expected_image_names = set(dataframe["name"].astype(str))
        actual_image_names = {
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if actual_image_names != expected_image_names:
            missing = sorted(expected_image_names - actual_image_names)[:5]
            extra = sorted(actual_image_names - expected_image_names)[:5]
            raise RuntimeError(
                f"{split_name} image mismatch; missing={missing}, extra={extra}."
            )

        expected_label_stems = {Path(name).stem for name in expected_image_names}
        actual_label_stems = {path.stem for path in label_dir.glob("*.txt")}
        if actual_label_stems != expected_label_stems:
            missing = sorted(expected_label_stems - actual_label_stems)[:5]
            extra = sorted(actual_label_stems - expected_label_stems)[:5]
            raise RuntimeError(
                f"{split_name} label mismatch; missing={missing}, extra={extra}."
            )

        for label_path in label_dir.glob("*.txt"):
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number} must contain five fields."
                    )
                class_id = int(parts[0])
                if class_id not in valid_class_ids:
                    raise ValueError(
                        f"{label_path}:{line_number} has class_id={class_id}."
                    )
                x_center, y_center, width, height = map(float, parts[1:])
                if not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                ):
                    raise ValueError(
                        f"{label_path}:{line_number} has invalid coordinates."
                    )

    unexpected_json = [
        output_dir / f"{split_name}.json" for split_name in splits
        if (output_dir / f"{split_name}.json").exists()
    ]
    if unexpected_json:
        raise RuntimeError(
            "Split JSON files must not be published: "
            + ", ".join(str(path) for path in unexpected_json)
        )

    with (output_dir / "data.yaml").open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    configured_names = {
        int(class_id): class_name
        for class_id, class_name in data_config["names"].items()
    }
    if configured_names != YOLO_CLASS_NAMES:
        raise RuntimeError("data.yaml class names do not match YOLO_CLASS_NAMES.")


def find_kaggle_executable() -> str:
    """Return the Kaggle CLI executable or raise an actionable error."""
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "The Kaggle CLI was not found. Install it with "
            "'python -m pip install kaggle'."
        )
    return executable


def kaggle_dataset_exists(kaggle_executable: str, dataset_ref: str) -> bool:
    """Check dataset existence without masking authentication failures."""
    result = subprocess.run(
        [kaggle_executable, "datasets", "status", dataset_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True

    diagnostic = f"{result.stdout}\n{result.stderr}".strip()
    normalized_diagnostic = diagnostic.casefold()
    if "404" in normalized_diagnostic or "not found" in normalized_diagnostic:
        return False
    raise RuntimeError(
        f"Could not determine whether Kaggle Dataset {dataset_ref} exists: "
        f"{diagnostic or 'unknown Kaggle CLI error'}"
    )


def publish_kaggle_dataset(
    output_dir: Path,
    owner: str,
    dataset_slug: str,
    mode: str,
    public: bool,
    version_message: str,
) -> None:
    """Create a Kaggle Dataset or upload a new version of an existing one."""
    kaggle_executable = find_kaggle_executable()
    dataset_ref = f"{owner}/{dataset_slug}"

    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = (
            "version"
            if kaggle_dataset_exists(kaggle_executable, dataset_ref)
            else "create"
        )

    if resolved_mode == "create":
        command = [
            kaggle_executable,
            "datasets",
            "create",
            "-p",
            str(output_dir),
            "--dir-mode",
            "zip",
        ]
        if public:
            command.append("--public")
    elif resolved_mode == "version":
        command = [
            kaggle_executable,
            "datasets",
            "version",
            "-p",
            str(output_dir),
            "-m",
            version_message,
            "--dir-mode",
            "zip",
        ]
    else:
        raise ValueError(f"Unsupported publication mode: {resolved_mode}")

    LOGGER.info("Publishing %s with Kaggle mode=%s", dataset_ref, resolved_mode)
    run_command(command)


def build_dataset(
    layout: DatasetLayout,
    output_dir: Path,
    random_state: int,
    workers: int,
    overwrite_output: bool,
    output_dataset_slug: str,
    kaggle_owner: str | None,
    output_dataset_title: str,
) -> None:
    """Run the complete BDD100K-to-Ultralytics-NVPD conversion pipeline."""
    LOGGER.info("Flattening source image directories when needed")
    moved_train = flatten_image_directory(layout.train_images_dir)
    moved_val = flatten_image_directory(layout.val_images_dir)
    LOGGER.info("Flattened source files: train=%d, val=%d", moved_train, moved_val)

    train_image_index = build_image_index(layout.train_images_dir)
    val_image_index = build_image_index(layout.val_images_dir)

    LOGGER.info("Loading BDD100K frame annotations")
    source_train = load_annotation_dataframe(layout.train_labels_path)
    source_val = (
        load_annotation_dataframe(layout.val_labels_path)
        .sort_values(by="name")
        .reset_index(drop=True)
    )

    train_dataframe, val_dataframe = create_balanced_train_val(
        source_train,
        random_state=random_state,
    )
    test_dataframe = create_test_split(source_val)
    validate_disjoint_splits(
        {
            "train": train_dataframe,
            "val": val_dataframe,
            "test": test_dataframe,
        }
    )

    final_splits = {
        "train": train_dataframe,
        "val": val_dataframe,
        "test": test_dataframe,
    }

    prepare_output_directory(output_dir, overwrite=overwrite_output)

    image_indices = {
        "train": train_image_index,
        "val": train_image_index,
        "test": val_image_index,
    }
    annotation_stats: dict[str, Counter[str]] = {}

    for split_name, dataframe in final_splits.items():
        copy_dataframe_images(
            dataframe,
            image_indices[split_name],
            output_dir / "images" / split_name,
            workers,
            split_name,
        )
        annotation_stats[split_name] = write_dataframe_yolo_labels(
            dataframe,
            image_indices[split_name],
            output_dir / "labels" / split_name,
            workers,
            split_name,
        )

    write_data_yaml(output_dir, output_dataset_slug)
    if kaggle_owner:
        write_kaggle_metadata(
            output_dir,
            owner=kaggle_owner,
            dataset_slug=output_dataset_slug,
            dataset_title=output_dataset_title,
        )

    validate_yolo_dataset(output_dir, final_splits)
    for split_name, dataframe in final_splits.items():
        log_split_summary(split_name, dataframe, annotation_stats[split_name])
    LOGGER.info("Dataset created successfully at %s", output_dir.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    configure_logging(args.log_level)

    raw_dir = args.raw_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    validate_working_paths(raw_dir, output_dir)

    if args.skip_download:
        if not dataset_is_ready(raw_dir):
            raise FileNotFoundError(
                f"The required BDD100K files were not found under {raw_dir}."
            )
        LOGGER.info("Using the existing source dataset at %s", raw_dir.resolve())
    elif dataset_is_ready(raw_dir) and not args.force_download:
        LOGGER.info("Source dataset is already available at %s", raw_dir.resolve())
    else:
        download_kaggle_dataset(
            dataset_slug=args.dataset_slug,
            destination_dir=raw_dir,
            force=args.force_download,
        )

    layout = resolve_dataset_layout(raw_dir)
    build_dataset(
        layout=layout,
        output_dir=output_dir,
        random_state=args.random_state,
        workers=args.workers,
        overwrite_output=args.overwrite_output,
        output_dataset_slug=args.output_dataset_slug,
        kaggle_owner=args.kaggle_owner,
        output_dataset_title=args.output_dataset_title,
    )

    if args.publish:
        publish_kaggle_dataset(
            output_dir=output_dir,
            owner=args.kaggle_owner,
            dataset_slug=args.output_dataset_slug,
            mode=args.publish_mode,
            public=args.public,
            version_message=args.version_message,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
