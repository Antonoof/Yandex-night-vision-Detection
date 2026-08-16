"""Loading + EDA helpers for the nvpdyf-bdd100k dataset.

Unlike src/datasets/bdd100k.py (which *builds* a YOLO-format dataset out of
raw BDD100K json annotations), this dataset already ships pre-converted to
the standard ultralytics YOLO layout:

    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
    data.yaml

so there is no conversion step here - just finding the dataset, reading
data.yaml for the class names, and parsing the label .txt files for EDA
and for sampling scenes to draw.
"""

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_dataset_root(input_dir):
    """Find the folder that has data.yaml next to an images/ subfolder.

    Args:
        input_dir (str | Path): root to search under; the dataset can be
            nested arbitrarily deep inside it (e.g. Kaggle's /kaggle/input).
    Returns:
        data_root (Path): the folder containing data.yaml, images/, labels/.
    """
    input_dir = Path(input_dir)
    for candidate in input_dir.rglob("data.yaml"):
        if (candidate.parent / "images").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"Could not find the dataset under {input_dir}: need a folder that "
        "has both an images/ subfolder and a data.yaml next to it."
    )


def load_classes(data_root):
    """Read data.yaml and return the class id -> name mapping.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
    Returns:
        classes (dict[int, str]): class id -> class name, as declared in
            data.yaml (this is the source of truth - do not hardcode it
            elsewhere, unlike the fixed COCO-derived CLASSES list in
            src/datasets/bdd100k.py).
    """
    data_yaml = Path(data_root) / "data.yaml"
    names = yaml.safe_load(data_yaml.read_text())["names"]
    return {int(i): n for i, n in names.items()}


def parse_label_file(path):
    """Parse one YOLO label .txt file.

    Args:
        path (Path): path to a <stem>.txt label file.
    Returns:
        boxes (list[tuple]): (class_id, cx, cy, w, h), normalized [0, 1].
        errors (list[str]): human-readable messages for malformed lines;
            empty if the file is clean.
    """
    boxes, errors = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{i} - expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{i} - non-numeric field")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{i} - non-positive width/height")
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes, errors


def collect_stats(data_root, split):
    """Walk one split's images/labels and gather EDA statistics.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        split (str): "train", "val", or "test".
    Returns:
        stats (dict): counts and raw box list for the split - see the keys
            below; consumed by describe_dataset and by plotting code.
    """
    data_root = Path(data_root)
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split

    images = {
        p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    }
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}

    stats = {
        "n_images": len(images),
        "missing_labels": sorted(images.keys() - labels.keys()),
        "orphan_labels": sorted(labels.keys() - images.keys()),
        "empty_labels": [],
        "objects_per_image": [],
        "class_counts": Counter(),
        "images_per_class": defaultdict(set),
        "boxes": [],
        "errors": [],
    }

    for stem in images.keys() & labels.keys():
        boxes, errors = parse_label_file(labels[stem])
        stats["errors"].extend(errors)
        stats["objects_per_image"].append(len(boxes))
        if not boxes:
            stats["empty_labels"].append(stem)
        for cls, *_ in boxes:
            stats["class_counts"][cls] += 1
            stats["images_per_class"][cls].add(stem)
        stats["boxes"].extend(boxes)

    return stats


def describe_dataset(stats_by_split, classes):
    """Log a per-split + per-class summary, in the style of
    src/datasets/bdd100k.describe_split_balance.

    Args:
        stats_by_split (dict[str, dict]): split name -> collect_stats output.
        classes (dict[int, str]): class id -> name, from load_classes.
    """
    for split, s in stats_by_split.items():
        n_boxes = sum(s["class_counts"].values())
        logger.info(
            "%-5s: %6d images  missing_labels=%-4d empty_labels=%-4d orphan_labels=%-4d boxes=%7d",
            split,
            s["n_images"],
            len(s["missing_labels"]),
            len(s["empty_labels"]),
            len(s["orphan_labels"]),
            n_boxes,
        )
        if s["errors"]:
            logger.warning(
                "%-5s: %d malformed label lines, e.g. %s",
                split,
                len(s["errors"]),
                s["errors"][0],
            )

    total = Counter()
    for s in stats_by_split.values():
        total.update(s["class_counts"])

    logger.info("%-14s %8s  %s", "class", "boxes", "  ".join(stats_by_split))
    for cid, name in sorted(classes.items(), key=lambda kv: -total[kv[0]]):
        per_split = "  ".join(
            f"{s['class_counts'][cid]:>{len(split)}d}"
            for split, s in stats_by_split.items()
        )
        logger.info("%-14s %8d  %s", name, total[cid], per_split)


def sample_scenes(data_root, split, n, seed, with_boxes_only=True):
    """Pick a reproducible random sample of frames for drawing.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        split (str): "train", "val", or "test".
        n (int): number of frames to sample.
        seed (int): RNG seed, for a reproducible sample across notebook runs.
        with_boxes_only (bool): if True, only sample frames that have at
            least one labeled object (more informative to look at).
    Returns:
        records (list[dict]): each with "name", "path", "boxes" (list of
            (class_id, cx, cy, w, h)).
    """
    data_root = Path(data_root)
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split

    stems = sorted(
        p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    rng = random.Random(seed)
    rng.shuffle(stems)

    records = []
    for stem in stems:
        label_path = labels_dir / f"{stem}.txt"
        boxes, _ = parse_label_file(label_path) if label_path.exists() else ([], [])
        if with_boxes_only and not boxes:
            continue
        image_path = next(
            (
                images_dir / f"{stem}{ext}"
                for ext in IMG_EXTS
                if (images_dir / f"{stem}{ext}").exists()
            ),
            None,
        )
        records.append({"name": image_path.name, "path": image_path, "boxes": boxes})
        if len(records) == n:
            break
    return records
