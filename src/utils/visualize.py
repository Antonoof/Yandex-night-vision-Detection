"""Drawing helpers: ground-truth scenes and model predictions."""

import sys
from pathlib import Path

import matplotlib

# train.py/inference.py run headless (no display server, e.g. on a cloud
# VM or Kaggle) and need the non-interactive Agg backend to save figures at
# all. A notebook is the opposite: forcing Agg there would silently break
# inline plt.show() output. ipykernel is only ever imported when running
# under a Jupyter kernel, so use that to tell the two apart.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.bdd100k import CLASSES, IMG_H, IMG_W

# One fixed color per class index, cycled from matplotlib's tab10 - classes
# keep the same color across figures instead of getting reassigned whenever
# the set of classes present in a given plot changes.
_PALETTE = plt.get_cmap("tab10").colors


def draw_ground_truth(records, classes, out_path=None, cols=1):
    """Save/return a figure with ground-truth boxes drawn on each record.

    Unlike draw_predictions, this needs no model - it draws the labels the
    dataset already has, for a quick "does this look right" sanity check
    (e.g. in an EDA notebook, before any training happens).

    Args:
        records (list[dict]): records from
            src.datasets.nvpdyf_bdd100k.sample_scenes, i.e. dicts with
            "name", "path", "boxes" (list of (class_id, cx, cy, w, h),
            normalized [0, 1]), and optionally "split"/"timeofday" (shown
            in the title when present).
        classes (dict[int, str]): class id -> name, e.g. from
            src.datasets.nvpdyf_bdd100k.load_classes.
        out_path (str | Path | None): if given, save the figure here
            (parent dirs are created); the figure is returned either way.
        cols (int): number of scenes per row.
    Returns:
        fig (matplotlib.figure.Figure): the composed figure.
    """
    rows = -(-len(records) // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, records):
        img = Image.open(r["path"])
        w, h = img.size
        ax.imshow(img)
        for cls, cx, cy, bw, bh in r["boxes"]:
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            color = _PALETTE[cls % len(_PALETTE)]
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), bw * w, bh * h, fill=False, color=color, linewidth=1.5
                )
            )
            ax.text(x1, y1 - 4, classes[cls], color=color, fontsize=8, weight="bold")
        # split/timeofday are optional (only present when records came from
        # sample_scenes with a tod_map) - fall back to just name + count
        tag = " ".join(str(r[k]) for k in ("split", "timeofday") if r.get(k))
        title = f"{r['name']}" + (f"  [{tag}]" if tag else "")
        ax.set_title(f"{title}  ({len(r['boxes'])} objects)")
        ax.axis("off")
    for ax in axes[len(records) :]:
        ax.axis("off")

    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    return fig


def draw_ground_truth(records, out_path=None, box_color="lime"):
    """Draw the annotated boxes of a few frames.

    Worth doing before any training: a conversion bug (xyxy vs. cxcywh, a
    forgotten normalization, the wrong frame size) raises no exception -
    training runs, the loss falls, and the metric just stays near zero. If
    the boxes sit on the objects, the conversion is right.

    Args:
        records (list[dict]): records from src.datasets.bdd100k.load_records.
        out_path (str | Path | None): where to save the figure. If None, the
            figure is returned instead of saved (for notebooks).
        box_color (str): colour of the rectangles.
    Returns:
        fig (matplotlib.figure.Figure): the drawn figure.
    """
    if not records:
        raise ValueError("nothing to draw: 'records' is empty")

    fig, axes = plt.subplots(len(records), 1, figsize=(13, 7 * len(records)))
    axes = [axes] if len(records) == 1 else axes

    for ax, r in zip(axes, records):
        ax.imshow(Image.open(r["path"]))
        for c, cx, cy, w, h in r["boxes"]:
            # back from normalized centre+size to pixel corner+size
            x, y = (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w * IMG_W,
                    h * IMG_H,
                    fill=False,
                    color=box_color,
                    linewidth=1.5,
                )
            )
            ax.text(x, y - 4, CLASSES[c], color=box_color, fontsize=8)
        ax.set_title(
            f"{r['name']} - {r['timeofday']}, {len(r['boxes'])} labeled objects"
        )
        ax.axis("off")

    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    return fig


def draw_comparison(model, records, out_path, imgsz, conf, device, title=""):
    """Ground truth next to the model's predictions, one row per frame.

    The ground-truth panel is what makes this useful: a missed object is
    invisible in a predictions-only figure — an empty frame looks the same
    whether the model failed or there was nothing to find.

    Args:
        model (ultralytics.YOLO): model to run predictions with.
        records (list[dict]): records from src.datasets.bdd100k.load_records.
            Use a *fixed* set (e.g. sorted by name) so figures stay
            comparable between runs.
        out_path (str | Path): where to save the figure.
        imgsz (int): inference image size.
        conf (float): confidence threshold — a human-facing value (e.g.
            0.25), unlike the low threshold used for mAP evaluation.
        device (int | str): device to run inference on.
        title (str): prefix for each row's titles, e.g. "night".
    Returns:
        out_path (Path): where the figure was saved.
    """
    if not records:
        raise ValueError("nothing to draw: 'records' is empty")

    fig, axes = plt.subplots(
        len(records), 2, figsize=(18, 5.2 * len(records)), squeeze=False
    )

    for row, r in enumerate(records):
        image = Image.open(r["path"])

        ax = axes[row][0]
        ax.imshow(image)
        for c, cx, cy, w, h in r["boxes"]:
            x, y = (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w * IMG_W,
                    h * IMG_H,
                    fill=False,
                    color="lime",
                    linewidth=1.5,
                )
            )
            ax.text(x, y - 4, CLASSES[c], color="lime", fontsize=7)
        ax.set_title(f"{title} {r['name']} — разметка: {len(r['boxes'])}", fontsize=10)
        ax.axis("off")

        res = model.predict(
            str(r["path"]), imgsz=imgsz, conf=conf, device=device, verbose=False
        )[0]
        ax = axes[row][1]
        ax.imshow(image)
        kept = 0
        for box, cls, score in zip(
            res.boxes.xyxy.cpu(), res.boxes.cls.cpu().long(), res.boxes.conf.cpu()
        ):
            cid = int(cls)
            if cid >= len(CLASSES):  # model trained on a different class set
                continue
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=1.5
                )
            )
            ax.text(
                x1,
                y1 - 4,
                f"{CLASSES[cid]} {float(score):.2f}",
                color="red",
                fontsize=7,
            )
            kept += 1
        ax.set_title(f"предсказание (conf ≥ {conf}) — найдено: {kept}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=90)
    plt.close(fig)
    return out_path


def draw_predictions(model, records, out_path, imgsz, conf, device, class_map=None):
    """Save a stacked figure: one row per record, predictions drawn in red.

    Args:
        model (ultralytics.YOLO): model to run predictions with.
        records (list[dict]): records from src.datasets.bdd100k.load_records.
        out_path (str | Path): where to save the figure (e.g. a .png path).
        imgsz (int): inference image size.
        conf (float): confidence threshold - a human-facing value (e.g.
            0.25), unlike the low threshold used for mAP evaluation.
        device (int | str): device to run inference on.
        class_map (dict[int, int] | None): same meaning as in
            evaluate_detector - keep only these prediction classes and remap
            them to ours. Required for COCO weights, whose class indices run
            to 79 while CLASSES has 7 entries; without it this function
            raises IndexError on the first prediction outside our set, and
            the picture would be labelled with the wrong names anyway.
    """
    fig, axes = plt.subplots(len(records), 1, figsize=(9, 5 * len(records)))
    axes = [axes] if len(records) == 1 else axes

    for ax, r in zip(axes, records):
        res = model.predict(
            str(r["path"]), imgsz=imgsz, conf=conf, device=device, verbose=False
        )[0]
        ax.imshow(Image.open(r["path"]))
        kept = 0
        for box, cls in zip(res.boxes.xyxy.cpu(), res.boxes.cls.cpu().long()):
            if class_map is not None:
                if int(cls) not in class_map:
                    continue
                cls = class_map[int(cls)]
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=1.5
                )
            )
            ax.text(x1, y1 - 4, CLASSES[int(cls)], color="red", fontsize=8)
            kept += 1
        ax.set_title(f"{r['name']} - predicted {kept} (labeled {len(r['boxes'])})")
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
