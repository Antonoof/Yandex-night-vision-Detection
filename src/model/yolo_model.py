"""Thin wrapper around ultralytics YOLO for the night-vision baseline."""

import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def build_model(weights: str) -> YOLO:
    """Load a YOLO model.

    Args:
        weights (str): a COCO-pretrained name (e.g. "yolov8n.pt") or a path
            to a local checkpoint (e.g. ".../weights/best.pt").
    Returns:
        model (YOLO): the loaded model.
    """
    return YOLO(weights)


def log_head_info(model: YOLO) -> None:
    """Log the detection head layout - the part fine-tuning replaces."""
    detect = model.model.model[-1]  # last module of the network = Detect
    logger.info("head module type: %s", type(detect).__name__)
    logger.info("nc (classes): %s", detect.nc)
    logger.info(
        "DFL reg_max: %s -> %s channels per box", detect.reg_max, 4 * detect.reg_max
    )
    logger.info("scales (P3/P4/P5): %s", detect.nl)

    logger.info("classification branch cv3 (replaced during fine-tuning):")
    for i, branch in enumerate(detect.cv3):
        last = branch[-1]  # final 1x1 conv of the branch
        out = getattr(last, "out_channels", "?")
        logger.info(
            "  P%d: last layer %s, out_channels=%s", i + 3, type(last).__name__, out
        )

    logger.info("coordinate branch cv2 (does not depend on class count):")
    for i, branch in enumerate(detect.cv2):
        last = branch[-1]
        logger.info("  P%d: out_channels=%s", i + 3, getattr(last, "out_channels", "?"))

    total = sum(p.numel() for p in model.model.parameters())
    head = sum(p.numel() for p in detect.parameters())
    cls_head = sum(p.numel() for p in detect.cv3.parameters())
    logger.info("total parameters: %s", f"{total:,}")
    logger.info("  head: %s (%.1f%%)", f"{head:,}", 100 * head / total)
    logger.info(
        "  classification branch: %s (%.1f%%)", f"{cls_head:,}", 100 * cls_head / total
    )
