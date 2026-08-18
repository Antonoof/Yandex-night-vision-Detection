"""Version-pinned NMS runtime configuration for complete validation output."""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def configure_nms_time_limit(max_time_img: float) -> None:
    """Increase Ultralytics' NMS budget without changing prediction scores.

    Ultralytics 8.4.120 does not expose ``max_time_img`` through train/val or
    predict arguments. Its NMS function does expose it, so a small pinned
    wrapper is used by native validation, periodic evaluation and inference.
    On timeout Ultralytics stops processing the remaining images in a batch,
    which can otherwise make mAP artificially low.
    """
    max_time_img = float(max_time_img)
    if max_time_img <= 0:
        raise ValueError("metrics.nms_max_time_img must be positive.")

    from ultralytics.utils import nms

    current = nms.non_max_suppression
    original = getattr(current, "_night_vision_original", current)

    @wraps(original)
    def configured(*args, **kwargs):
        kwargs.setdefault("max_time_img", max_time_img)
        return original(*args, **kwargs)

    configured._night_vision_original = original
    nms.non_max_suppression = configured
    logger.info("NMS max_time_img configured to %.3fs", max_time_img)
