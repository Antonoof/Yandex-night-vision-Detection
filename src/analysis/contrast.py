"""Object-vs-background contrast, measured per ground-truth box.

The project's working hypothesis is that night detection suffers because
objects blend into a dark background. That is measurable, not just visible:
for every labelled box we compare its brightness with the brightness of the
ring around it, and record how many of the 255 grey levels the object
actually spans.

The second number is the one that explains Zero-DCE's failure. Stretching a
tonal range that spans five levels cannot invent detail - it only amplifies
sensor noise and JPEG artefacts along with the signal.

Nothing here needs a GPU. Matching against a checkpoint's predictions does,
but only for inference, and it runs perfectly well on CPU.
"""

import logging

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.datasets.bdd100k import CLASSES, IMG_H, IMG_W

logger = logging.getLogger(__name__)

# COCO's own size buckets, in pixels of the original frame, so that these rows
# can be read next to map_small / map_medium / map_large.
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96

CSV_COLUMNS = [
    "name",
    "timeofday",
    "class_id",
    "class_name",
    "area_px",
    "size_bucket",
    "l_object",
    "l_background",
    "weber",
    "dyn_range",
    "rms_contrast",
    "matched",
    "best_iou",
]


def size_bucket(area_px):
    """COCO bucket for a box area given in pixels."""
    if area_px < SMALL_MAX:
        return "small"
    return "medium" if area_px < MEDIUM_MAX else "large"


def to_pixels(box, width=IMG_W, height=IMG_H):
    """(cls, cx, cy, w, h) normalized -> (cls, x1, y1, x2, y2) in pixels."""
    cls, cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2) * width))
    y1 = int(round((cy - bh / 2) * height))
    x2 = int(round((cx + bw / 2) * width))
    y2 = int(round((cy + bh / 2) * height))
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    return cls, x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def frame_contrast(path, boxes, ring_frac=0.5):
    """Per-box brightness statistics for one frame.

    The background ring deliberately excludes every other labelled box: in
    dense traffic the neighbourhood of a car is mostly other cars, and
    measuring "contrast against other objects" would answer a different
    question than the one we are asking.

    Args:
        path (str | Path): frame to read.
        boxes (list[tuple]): (cls, cx, cy, w, h), normalized.
        ring_frac (float): ring width, as a fraction of the box's own size.
    Returns:
        rows (list[dict]): one dict per box; boxes whose ring came out empty
            are skipped rather than reported with a fabricated background.
    """
    grey = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    height, width = grey.shape

    pixel_boxes = [to_pixels(b, width, height) for b in boxes]
    occupied = np.zeros((height, width), dtype=bool)
    for _, x1, y1, x2, y2 in pixel_boxes:
        occupied[y1:y2, x1:x2] = True

    rows = []
    for (cls, x1, y1, x2, y2), _ in zip(pixel_boxes, boxes):
        patch = grey[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        pad_x = max(1, int(round(ring_frac * (x2 - x1))))
        pad_y = max(1, int(round(ring_frac * (y2 - y1))))
        ox1, oy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        ox2, oy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)

        neighbourhood = grey[oy1:oy2, ox1:ox2]
        ring_mask = ~occupied[oy1:oy2, ox1:ox2]
        ring = neighbourhood[ring_mask]
        if ring.size < 16:  # fully enclosed by other objects - no usable ring
            continue

        l_object = float(patch.mean())
        l_background = float(ring.mean())
        low, high = np.percentile(patch, [5, 95])

        rows.append(
            {
                "class_id": int(cls),
                "class_name": CLASSES[int(cls)],
                "area_px": int((x2 - x1) * (y2 - y1)),
                "size_bucket": size_bucket((x2 - x1) * (y2 - y1)),
                "l_object": l_object,
                "l_background": l_background,
                # Weber contrast: how far the object stands out from its
                # surroundings, relative to those surroundings.
                "weber": (l_object - l_background) / max(l_background, 1.0),
                # How many grey levels the object spans at all. This is the
                # ceiling on what any tone-mapping method could recover.
                "dyn_range": float(high - low),
                "rms_contrast": float(patch.std()),
                "_pixel_box": (x1, y1, x2, y2),
            }
        )
    return rows


def collect(records, ring_frac=0.5, desc="contrast"):
    """Run frame_contrast over records, carrying frame-level fields through."""
    rows = []
    for record in tqdm(records, desc=desc):
        try:
            for row in frame_contrast(record["path"], record["boxes"], ring_frac):
                row["name"] = record["name"]
                row["timeofday"] = record["timeofday"]
                row["matched"] = None
                row["best_iou"] = None
                rows.append(row)
        except Exception as e:  # a single unreadable frame must not stop the run
            logger.warning("кадр %s пропущен: %s", record["name"], e)
    return rows


def _iou_matrix(gt_boxes, pred_boxes):
    """IoU between two sets of xyxy boxes -> [len(gt), len(pred)]."""
    if not len(gt_boxes) or not len(pred_boxes):
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    gt = np.asarray(gt_boxes, dtype=np.float32)[:, None, :]
    pred = np.asarray(pred_boxes, dtype=np.float32)[None, :, :]

    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

    area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    area_pred = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    union = area_gt + area_pred - inter
    return np.where(union > 0, inter / union, 0.0)


def match_predictions(model, records, rows, imgsz, conf, iou_thr, device, batch_size):
    """Mark which boxes the model actually found.

    Turns the contrast numbers into an answer rather than a description: if
    recall as a function of contrast is the same curve for night and day, the
    whole night-time deficit is explained by contrast. If the night curve sits
    lower at equal contrast, something else is at work - noise, motion blur,
    headlight glare - and tone mapping was never going to fix it.

    Matching is greedy by IoU within the same class, which is what a
    per-object "was it found" question needs; it is not the COCO protocol and
    the numbers here are not mAP.
    """
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["name"], []).append(row)

    for start in tqdm(range(0, len(records), batch_size), desc="predict"):
        chunk = records[start : start + batch_size]
        results = model.predict(
            [str(r["path"]) for r in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        for record, result in zip(chunk, results):
            frame_rows = by_frame.get(record["name"])
            if not frame_rows:
                continue
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)

            for row in frame_rows:
                same_class = pred_cls == row["class_id"]
                if not same_class.any():
                    row["matched"], row["best_iou"] = False, 0.0
                    continue
                ious = _iou_matrix([row["_pixel_box"]], pred_boxes[same_class])[0]
                best = float(ious.max())
                row["best_iou"] = best
                row["matched"] = best >= iou_thr
    return rows
