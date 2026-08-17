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

Per-image day/night wasn't part of that original layout - it ships
separately as a sidecar timeofday.csv (see find_timeofday_csv/
load_timeofday), added after the fact once someone noticed the gap.

Known duplication: the origin/nvpdyf-dataset-loader branch (not yet merged
as of this writing) rewrote src/datasets/bdd100k.py itself to read this same
dataset, since train.py/inference.py/metrics already import from there -
that is the module to use for anything that feeds training or evaluation.
This module stays deliberately separate for now (EDA only, never imported
by train.py) rather than depending on an unmerged branch, but the two
disagree on one real semantic point, not just style: that branch's
find_dataset_root requires timeofday.csv to exist and load_records iterates
*its rows* (an image missing from timeofday.csv is invisible to training),
whereas this module iterates images/ and only falls back to UNKNOWN_TOD for
rows missing from timeofday.csv. Once that branch merges, revisit whether
this module should still exist, or whether dataset_baseline.ipynb should
just call into the merged src/datasets/bdd100k.py instead.
"""

import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# bucket used for images collect_stats/sample_scenes can't find in the
# timeofday map (no timeofday.csv given, or the image is missing a row) -
# never one of the real values from the csv, so it can't collide with them
UNKNOWN_TOD = "unknown"


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


def find_timeofday_csv(input_dir):
    """Find timeofday.csv under input_dir, if it was shipped at all.

    The base dataset didn't originally record day/night per image;
    timeofday.csv was added afterwards to fill that gap, and may land
    anywhere under the Kaggle input root - next to data.yaml, or as its
    own separate dataset input - so this searches the same way
    find_dataset_root does, rather than assuming a fixed location.

    Args:
        input_dir (str | Path): root to search under (e.g. /kaggle/input).
    Returns:
        csv_path (Path | None): path to timeofday.csv, or None if it
            isn't present under input_dir yet.
    """
    return next(Path(input_dir).rglob("timeofday.csv"), None)


def load_timeofday(csv_path):
    """Read timeofday.csv into an image stem -> timeofday mapping.

    Args:
        csv_path (str | Path): path to timeofday.csv, e.g. from
            find_timeofday_csv. Expected columns: "image" (a split-prefixed
            path, e.g. "train/0000f77c-6257be58.jpg") and "timeofday"
            ("daytime" or "night").
    Returns:
        tod_map (dict[str, str]): image stem -> timeofday value, taken
            as-is from the csv (not restricted to "daytime"/"night", in
            case another value shows up upstream later - callers that
            care should treat anything other than those two as unknown).
    """
    tod_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = (row.get("image") or "").strip()
            tod = (row.get("timeofday") or "").strip()
            if image and tod:
                tod_map[Path(image).stem] = tod
    return tod_map


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


def collect_stats(data_root, split, tod_map=None):
    """Walk one split's images/labels and gather EDA statistics.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        split (str): "train", "val", or "test".
        tod_map (dict[str, str] | None): image stem -> "daytime"/"night",
            from load_timeofday. Optional - without it, everything is
            counted under UNKNOWN_TOD and the day/night breakdown in
            describe_daynight has nothing to show.
    Returns:
        stats (dict): counts and raw box list for the split - see the keys
            below; consumed by describe_dataset/describe_daynight and by
            plotting code.
    """
    data_root = Path(data_root)
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    tod_map = tod_map or {}

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
        "n_images_by_tod": Counter(),
        "class_counts_by_tod": defaultdict(Counter),
    }

    for stem in images.keys() & labels.keys():
        tod = tod_map.get(stem, UNKNOWN_TOD)
        stats["n_images_by_tod"][tod] += 1

        boxes, errors = parse_label_file(labels[stem])
        stats["errors"].extend(errors)
        stats["objects_per_image"].append(len(boxes))
        if not boxes:
            stats["empty_labels"].append(stem)
        for cls, *_ in boxes:
            stats["class_counts"][cls] += 1
            stats["images_per_class"][cls].add(stem)
            stats["class_counts_by_tod"][tod][cls] += 1
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


def describe_daynight(stats_by_split, classes):
    """Log day/night frame counts per split and a per-class night share.

    Mirrors src/datasets/bdd100k.describe_split_balance, which computes
    the same night-share table straight from the (day/night-labeled) old
    dataset's json - here the split has to come from timeofday.csv
    instead, via collect_stats(..., tod_map=...).

    For context on what "balanced" should look like: per
    dataset/prepare_bdd100k_nvpd.py (the script that builds this dataset,
    see other branches), train/val are sampled to an exact 4:1
    daytime:night ratio, while test keeps BDD100K's validation split
    as-is (unbalanced).

    Args:
        stats_by_split (dict[str, dict]): split name -> collect_stats
            output, computed with a tod_map - without one, every image
            falls under UNKNOWN_TOD and this has nothing to show.
        classes (dict[int, str]): class id -> name, from load_classes.
    """
    if all(
        set(s["n_images_by_tod"]) <= {UNKNOWN_TOD} for s in stats_by_split.values()
    ):
        logger.warning(
            "no timeofday info for any split - pass tod_map=load_timeofday(...) "
            "to collect_stats to get a day/night breakdown"
        )
        return

    for split, s in stats_by_split.items():
        logger.info("%-5s: %s", split, dict(s["n_images_by_tod"]))

    total_by_tod = defaultdict(Counter)
    for s in stats_by_split.values():
        for tod, counts in s["class_counts_by_tod"].items():
            total_by_tod[tod].update(counts)

    def night_boxes(cid):
        return total_by_tod["night"][cid]

    logger.info("%-14s %8s %8s %11s   (all splits)", "class", "night", "day", "night share")
    for cid, name in sorted(classes.items(), key=lambda kv: -night_boxes(kv[0])):
        n, d = total_by_tod["night"][cid], total_by_tod["daytime"][cid]
        share = n / (n + d) if (n + d) else 0.0
        logger.info("%-14s %8d %8d %10.1f%%", name, n, d, share * 100)


def sample_scenes(data_root, splits, n, seed, with_boxes_only=True, tod=None, tod_map=None):
    """Pick a reproducible random sample of frames for drawing.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        splits (str | list[str]): one split name, or several to search
            across at once (e.g. all of SPLITS, when looking for a scene
            of a specific timeofday that might be rare in any one split).
        n (int): number of frames to sample.
        seed (int): RNG seed, for a reproducible sample across notebook runs.
        with_boxes_only (bool): if True, only sample frames that have at
            least one labeled object (more informative to look at).
        tod (str | None): if given (together with tod_map), only sample
            frames with this exact timeofday value (e.g. "night").
        tod_map (dict[str, str] | None): image stem -> timeofday, from
            load_timeofday. Required when tod is given.
    Returns:
        records (list[dict]): each with "name", "path", "split",
            "timeofday" (tod_map value, or None if no tod_map was given),
            and "boxes" (list of (class_id, cx, cy, w, h)).
    """
    if isinstance(splits, str):
        splits = [splits]
    data_root = Path(data_root)
    tod_map = tod_map or {}

    candidates = [
        (split, p.stem)
        for split in splits
        for p in (data_root / "images" / split).iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    records = []
    for split, stem in candidates:
        if tod is not None and tod_map.get(stem) != tod:
            continue
        images_dir = data_root / "images" / split
        label_path = data_root / "labels" / split / f"{stem}.txt"
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
        records.append(
            {
                "name": image_path.name,
                "path": image_path,
                "split": split,
                "timeofday": tod_map.get(stem),
                "boxes": boxes,
            }
        )
        if len(records) == n:
            break
    return records
