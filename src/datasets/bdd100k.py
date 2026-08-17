"""NVPDYF BDD100K dataset access for the night-vision detection baseline.

The dataset already ships in YOLO format (``images/``, ``labels/``,
``data.yaml``), so nothing is converted here - we only read it into records
that the rest of the project understands. Time of day, which every night/day
split in this project depends on, is not part of the YOLO format and lives in
a separate ``timeofday.csv``.
"""

import csv
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Class order is dictated by the dataset's own data.yaml (alphabetical), NOT by
# COCO order - the label files store these ids. Changing this list without
# rebuilding the dataset silently relabels everything.
CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "person",
    "traffic light",
    "truck",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# COCO-80 class index -> our class index, used to read zero-shot predictions
# from a COCO-pretrained model. Hardcoded on purpose (COCO order is fixed), so
# it does NOT follow CLASSES automatically - see _check_class_names.
COCO80_TO_OURS = {0: 4, 1: 0, 2: 2, 3: 3, 5: 1, 7: 6, 9: 5}

IMG_W, IMG_H = 1280, 720  # all BDD100K frames share this size

TIMEOFDAY_CSV = "timeofday.csv"


def find_dataset_root(input_dir):
    """Find the folder that has data.yaml, images/ and timeofday.csv together.

    Args:
        input_dir (str | Path): root to search under; the dataset can be
            nested arbitrarily deep inside it (e.g. Kaggle's /kaggle/input).
    Returns:
        data_root (Path): the folder containing images/, labels/, data.yaml.
    """
    input_dir = Path(input_dir)
    for candidate in input_dir.rglob("data.yaml"):
        root = candidate.parent
        if (root / "images").is_dir() and (root / TIMEOFDAY_CSV).is_file():
            _check_class_names(root)
            return root
    raise FileNotFoundError(
        f"Could not find the dataset under {input_dir}: need a folder that has "
        f"an images/ subfolder, a data.yaml and a {TIMEOFDAY_CSV} next to it."
    )


def _check_class_names(data_root):
    """Fail loudly if the dataset's class order differs from CLASSES.

    The label files store bare integer ids, so a reordered data.yaml does not
    break anything visibly - it just relabels every box (trucks counted as
    traffic lights, and so on) and quietly corrupts every metric. This is the
    one failure mode that has already bitten this project once, so it is
    checked on every run rather than trusted.
    """
    names = yaml.safe_load((data_root / "data.yaml").read_text()).get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if list(names or []) != CLASSES:
        raise ValueError(
            f"{data_root / 'data.yaml'} lists classes {names}, but "
            f"src/datasets/bdd100k.py expects {CLASSES}. Fix CLASSES *and* "
            "COCO80_TO_OURS together, then re-measure the baseline - metrics "
            "from different class orders are not comparable."
        )


def _load_timeofday(data_root):
    """Read timeofday.csv into {"<split>/<file name>": timeofday}."""
    with open(data_root / TIMEOFDAY_CSV, newline="") as f:
        return {row["image"]: row["timeofday"] for row in csv.DictReader(f)}


def load_records(data_root, split):
    """Read one split's YOLO labels and attach each frame's time of day.

    Args:
        data_root (str | Path): folder returned by find_dataset_root.
        split (str): "train", "val", or "test".
    Returns:
        records (list[dict]): one dict per frame, with "name", "path",
            "timeofday", and "boxes" (list of (class_id, cx, cy, w, h),
            normalized to the frame size).
    """
    data_root = Path(data_root)
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split

    # timeofday.csv is the source of truth for a split's contents: it covers
    # every frame, while a label file is absent for frames with no objects.
    timeofday = _load_timeofday(data_root)
    prefix = f"{split}/"

    records = []
    for image_rel in sorted(k for k in timeofday if k.startswith(prefix)):
        name = image_rel[len(prefix) :]
        label_path = labels_dir / f"{Path(name).stem}.txt"

        boxes = []
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:  # trailing blank line
                    continue
                cid, *coords = parts
                cx, cy, w, h = (float(v) for v in coords[:4])
                boxes.append((int(cid), cx, cy, w, h))

        records.append(
            {
                "name": name,
                "path": images_dir / name,
                "timeofday": timeofday[image_rel],
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

    The source dataset is already in this format, but it is rebuilt anyway:
    subsample() may have dropped frames, and the source tree also holds the
    test split, which must never be visible to training.

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
