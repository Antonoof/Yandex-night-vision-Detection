"""Ground truth and prediction drawn on the same crop, with the IoU.

The prediction figures we already log to Comet put ground truth and
predictions in separate panels, on the full 1280x720 frame. At that scale a
box that is off by twenty pixels looks correct - which is exactly why the
localization gap went unnoticed for so long.

This draws both boxes on one crop around the object, and prints the IoU. The
gap stops being a statistic and becomes something you can point at.
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

logger = logging.getLogger(__name__)

GT_COLOR = "#2ecc71"
PRED_COLOR = "#e74c3c"


def pick_examples(
    rows,
    timeofday="night",
    iou_range=(0.4, 0.8),
    min_area=2000,
    n=8,
    seed=42,
    require_matched=True,
):
    """Choose comparable examples: objects in a given IoU band.

    Only reasonably large objects are eligible - on a 20x20 box the drawn
    line width itself would hide the effect being shown.

    ``require_matched`` keeps only boxes counted as found, i.e. IoU >= the
    matching threshold (0.5 by default), which silently raises the floor of
    ``iou_range``. Pass False to reach the badly-placed boxes below it: the
    prediction and its IoU are recorded there too, and those are the cases
    where it is visible at all whether the model or the label is wrong.
    """
    pool = [
        r
        for r in rows
        if r["timeofday"] == timeofday
        and (r["matched"] if require_matched else True)
        and r.get("best_iou") is not None
        and iou_range[0] <= r["best_iou"] <= iou_range[1]
        and r["area_px"] >= min_area
        and r.get("pred_x1") is not None
    ]
    if not pool:
        return []
    pool.sort(key=lambda r: (r["name"], r["gt_x1"]))  # deterministic order
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def draw_localization_grid(
    rows, images_dir, out_path, timeofday="night", n=8, ncols=4, pad_frac=0.8, **kwargs
):
    """A grid of crops, each showing ground truth vs prediction and the IoU.

    Args:
        rows (list[dict]): rows from boxes.csv (needs the gt_*/pred_* columns).
        images_dir (str | Path): folder holding the frames named in "name".
        out_path (str | Path): where to write the figure.
        timeofday (str): "night" or "daytime".
        n (int): how many examples.
        ncols (int): columns in the grid.
        pad_frac (float): crop padding, as a fraction of the box size.
    Returns:
        out_path: the figure, or None if no example matched the filters.
    """
    from pathlib import Path

    examples = pick_examples(rows, timeofday=timeofday, n=n, **kwargs)
    if not examples:
        logger.warning("нет подходящих примеров для '%s'", timeofday)
        return None

    images_dir = Path(images_dir)
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, examples):
        frame = images_dir / row["name"]
        try:
            image = Image.open(frame).convert("RGB")
        except Exception as e:
            logger.warning("кадр %s не открылся: %s", frame, e)
            continue

        gt = (row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"])
        pr = (row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"])

        # crop around both boxes together: cropping around ground truth alone
        # would push a badly-placed prediction out of frame
        x1 = min(gt[0], pr[0])
        y1 = min(gt[1], pr[1])
        x2 = max(gt[2], pr[2])
        y2 = max(gt[3], pr[3])
        pad_x = max(12, int(pad_frac * (x2 - x1)))
        pad_y = max(12, int(pad_frac * (y2 - y1)))
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2 = min(image.width, x2 + pad_x)
        cy2 = min(image.height, y2 + pad_y)

        ax.imshow(np.asarray(image.crop((cx1, cy1, cx2, cy2))))
        for (bx1, by1, bx2, by2), color, style in (
            (gt, GT_COLOR, "-"),
            (pr, PRED_COLOR, "--"),
        ):
            ax.add_patch(
                patches.Rectangle(
                    (bx1 - cx1, by1 - cy1),
                    bx2 - bx1,
                    by2 - by1,
                    linewidth=2,
                    edgecolor=color,
                    linestyle=style,
                    facecolor="none",
                )
            )
        # Name the worst edge in pixels. Two boxes with the same IoU can be
        # wrong in opposite ways, and the eye is bad at telling which edge
        # moved - saying it out loud is what makes a crop answerable: is the
        # prediction wrong here, or is the ground truth?
        edges = {
            "лево": gt[0] - pr[0],
            "право": pr[2] - gt[2],
            "верх": gt[1] - pr[1],
            "низ": pr[3] - gt[3],
        }
        worst, delta = max(edges.items(), key=lambda kv: abs(kv[1]))
        ax.set_title(
            f"{row['class_name']}  IoU={row['best_iou']:.2f}  "
            f"{worst} {delta:+d}px",
            fontsize=10,
        )
        ax.axis("off")

    label = "ночь" if timeofday == "night" else "день"
    fig.suptitle(
        f"{label}: разметка (сплошная зелёная) против предсказания "
        f"(пунктирная красная)",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("график: %s", out_path)
    return out_path
