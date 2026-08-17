"""BDD100K -> YOLO dataset preparation for the night-vision detection baseline."""

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Our classes, in the order they appear in COCO.
CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# COCO-80 class index -> our class index. The gaps (4=airplane, 6=train, 8=boat, ...)
# are COCO classes that BDD100K does not have.
COCO80_TO_OURS = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 9: 6}

IMG_W, IMG_H = 1280, 720  # all BDD100K frames share this size


def find_dataset_root(input_dir):
    """Find the folder that has images/ and val.json next to each other.

    Args:
        input_dir (str | Path): root to search under; the dataset can be
            nested arbitrarily deep inside it (e.g. Kaggle's /kaggle/input).
    Returns:
        data_root (Path): the folder containing images/ and the *.json files.
    """
    input_dir = Path(input_dir)
    for candidate in input_dir.rglob("val.json"):
        if (candidate.parent / "images").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"Could not find the dataset under {input_dir}: need a folder that "
        "has both an images/ subfolder and a val.json next to it."
    )


def load_records(data_root, split):
    """Read <split>.json and convert the annotations to YOLO format.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        split (str): "train", "val", or "test".
    Returns:
        records (list[dict]): one dict per frame, with "name", "path",
            "timeofday", and "boxes" (list of (class_id, cx, cy, w, h)).
    """
    data_root = Path(data_root)
    images_dir = data_root / "images" / split
    split_json = data_root / f"{split}.json"

    records = []
    for r in json.loads(split_json.read_text()):
        boxes = []
        for lab in r.get("labels") or []:
            cid = CLASS_TO_ID.get(lab.get("coco_category"))
            if cid is None:  # class outside our list - skip
                continue
            # box_yolo is already [cx, cy, w, h], normalized to the frame size.
            cx, cy, w, h = lab["box_yolo"]
            boxes.append((cid, cx, cy, w, h))
        records.append(
            {
                "name": r["name"],
                "path": images_dir / r["name"],
                "timeofday": r.get("timeofday"),
                "boxes": boxes,
            }
        )

    if records:
        _check_frame_size(records[0]["path"])
    return records


def _check_frame_size(path):
    """Spot-check one frame against the fixed IMG_W/IMG_H assumption that
    targets_to_tensors relies on to denormalize ground-truth boxes -
    predictions are scaled to each image's real size by ultralytics, so a
    mismatch here would silently corrupt every mAP number."""
    actual = Image.open(path).size
    if actual != (IMG_W, IMG_H):
        raise ValueError(
            f"Expected {IMG_W}x{IMG_H} frames but {path} is "
            f"{actual[0]}x{actual[1]}. Update IMG_W/IMG_H in "
            "src/datasets/bdd100k.py if this dataset variant differs."
        )


def describe_split_balance(train_records, val_records):
    """Log frame counts and the per-class night/day imbalance (train+val).

    Args:
        train_records (list[dict]): records from load_records(..., "train").
        val_records (list[dict]): records from load_records(..., "val").
    Returns:
        stats (dict): the same numbers, for callers that want to plot or
            table them instead of reading the log:
            ``{"splits": {split: {"frames", "boxes", "timeofday"}},
               "per_class": {class_name: {"night", "day", "night_share"}}}``.
    """
    splits = {}
    for split, part in (("train", train_records), ("val", val_records)):
        tod = Counter(r["timeofday"] for r in part)
        splits[split] = {
            "frames": len(part),
            "boxes": sum(len(r["boxes"]) for r in part),
            "timeofday": dict(tod),
        }
        logger.info(
            "%-5s: %6d frames  %s  boxes=%7d",
            split,
            splits[split]["frames"],
            splits[split]["timeofday"],
            splits[split]["boxes"],
        )

    per_class = defaultdict(Counter)
    for r in train_records + val_records:
        for c, *_ in r["boxes"]:
            per_class[CLASSES[c]][r["timeofday"]] += 1

    logger.info(
        "%-14s %7s %7s %11s   (train+val)", "class", "night", "day", "night share"
    )
    balance = {}
    for name, c in sorted(per_class.items(), key=lambda kv: -sum(kv[1].values())):
        n, d = c["night"], c["daytime"]
        share = n / (n + d) if (n + d) else 0.0
        balance[name] = {"night": n, "day": d, "night_share": share}
        logger.info("%-14s %7d %7d %9.1f%%", name, n, d, share * 100)

    return {"splits": splits, "per_class": balance}


def subsample(records, limit, seed):
    """Shrink the set while keeping the night/day ratio.

    Args:
        records (list[dict]): records from load_records.
        limit (int | None): target size; None keeps all records unchanged.
        seed (int): shuffle seed, for a reproducible subsample.
    Returns:
        records (list[dict]): the (possibly shrunk) subset.
    """
    if limit is None or limit >= len(records):
        return records
    by_tod = defaultdict(list)
    for r in records:
        by_tod[r["timeofday"]].append(r)
    rng = random.Random(seed)
    out = []
    for tod in sorted(by_tod, key=str):
        items = sorted(by_tod[tod], key=lambda r: r["name"])  # sort => reproducible
        rng.shuffle(items)
        out += items[: round(limit * len(by_tod[tod]) / len(records))]
    return out


def _link_or_copy(src, dst):
    """Symlink src at dst; fall back to copying where symlinks aren't
    available (e.g. Windows without developer mode, some network mounts)."""
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_dataset(splits, out_dir):
    """Build the images/labels tree + data.yaml that ultralytics expects.

    Args:
        splits (dict[str, list[dict]]): e.g. {"train": [...], "val": [...]},
            each value a list of records from load_records/subsample.
        out_dir (str | Path): destination folder; wiped and rebuilt if it
            already exists.
    Returns:
        data_yaml (Path): path to the written data.yaml.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split_name, split_records in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for r in tqdm(split_records, desc=f"{split_name:5s}", leave=False):
            _link_or_copy(r["path"], img_dir / r["name"])
            lines = [
                f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for c, cx, cy, w, h in r["boxes"]
            ]
            # even frames with no objects get a (empty) label file
            (lbl_dir / f"{Path(r['name']).stem}.txt").write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {i: n for i, n in enumerate(CLASSES)},
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return data_yaml
