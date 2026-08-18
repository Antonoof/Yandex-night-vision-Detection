"""COCO-style detection metrics, computed separately for night and day."""

import logging

import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

from src.datasets.bdd100k import CLASSES, IMG_H, IMG_W

logger = logging.getLogger(__name__)

RESULT_KEYS = [
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_100",
]


def targets_to_tensors(record):
    """Ground truth for one frame -> torchmetrics format (xyxy in pixels).

    Args:
        record (dict): a record from src.datasets.bdd100k.load_records.
    Returns:
        target (dict): {"boxes": Tensor[N, 4], "labels": Tensor[N]}.
    """
    if not record["boxes"]:
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    t = torch.tensor(
        [[cx, cy, w, h] for _, cx, cy, w, h in record["boxes"]], dtype=torch.float32
    )
    boxes = torch.stack(
        [
            (t[:, 0] - t[:, 2] / 2) * IMG_W,  # x1
            (t[:, 1] - t[:, 3] / 2) * IMG_H,  # y1
            (t[:, 0] + t[:, 2] / 2) * IMG_W,  # x2
            (t[:, 1] + t[:, 3] / 2) * IMG_H,  # y2
        ],
        dim=1,
    )
    labels = torch.tensor([c for c, *_ in record["boxes"]], dtype=torch.long)
    return {"boxes": boxes, "labels": labels}


@torch.no_grad()
def evaluate_detector(
    model,
    records,
    imgsz,
    conf,
    iou,
    max_det,
    device,
    batch_size,
    class_map=None,
    desc="eval",
):
    """Run the model over frames and compute COCO metrics.

    Args:
        model (ultralytics.YOLO): model to evaluate.
        records (list[dict]): records from src.datasets.bdd100k.load_records.
        imgsz (int): inference image size.
        conf (float): confidence threshold (use a low value, e.g. 0.001, for
            a fair mAP sweep - see src/configs/metrics/detection.yaml).
        iou (float): NMS IoU threshold.
        max_det (int): max detections per image.
        device (int | str): device to run inference on.
        batch_size (int): frames per predict() call.
        class_map (dict[int, int] | None): if given, keep only predictions
            with these class indices and remap them to ours. Needed for
            zero-shot eval (COCO-80 -> our 7 classes).
        desc (str): tqdm progress bar label.
    Returns:
        metrics (dict): flat dict of overall + per-class mAP/mAR values.
    """
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )

    for i in tqdm(range(0, len(records), batch_size), desc=desc):
        chunk = records[i : i + batch_size]
        results = model.predict(
            [str(r["path"]) for r in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        preds = []
        for res in results:
            b = res.boxes
            boxes = b.xyxy.cpu()
            scores = b.conf.cpu()
            labels = b.cls.cpu().long()
            if class_map is not None:
                keep = torch.tensor(
                    [int(l) in class_map for l in labels], dtype=torch.bool
                )
                boxes, scores = boxes[keep], scores[keep]
                labels = torch.tensor(
                    [class_map[int(l)] for l in labels[keep]], dtype=torch.long
                )
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})

        metric.update(preds, [targets_to_tensors(r) for r in chunk])

    raw = metric.compute()

    out = {k: float(raw[k]) for k in RESULT_KEYS}
    per_class = torch.atleast_1d(raw["map_per_class"])
    for class_id, value in zip(torch.atleast_1d(raw["classes"]).tolist(), per_class):
        out[f"map_{CLASSES[class_id]}"] = float(value)
    metric.reset()
    return out


def print_results(title, night, day):
    """Log night/day side by side - shows the degradation, not raw numbers.

    Args:
        title (str): header line.
        night (dict): output of evaluate_detector on the night subset.
        day (dict): output of evaluate_detector on the day subset.
    """
    logger.info("%s", "=" * 74)
    logger.info("%s", title)
    logger.info("%s", "=" * 74)
    logger.info("%-22s %9s %9s %10s", "metric", "night", "day", "diff")
    for key in night:
        n, d = night[key], day.get(key, float("nan"))
        # -1 = metric undefined (class absent from the subset), COCO convention
        mark = "  (no data)" if min(n, d) < 0 else ""
        logger.info("%-22s %9.4f %9.4f %+10.4f%s", key, n, d, n - d, mark)
