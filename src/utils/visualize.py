"""Figures comparing model predictions / ground truth to images."""

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

from src.datasets.bdd100k import CLASSES

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


def draw_predictions(model, records, out_path, imgsz, conf, device):
    """Save a stacked figure: one row per record, predictions drawn in red.

    Args:
        model (ultralytics.YOLO): model to run predictions with.
        records (list[dict]): records from src.datasets.bdd100k.load_records.
        out_path (str | Path): where to save the figure (e.g. a .png path).
        imgsz (int): inference image size.
        conf (float): confidence threshold - a human-facing value (e.g.
            0.25), unlike the low threshold used for mAP evaluation.
        device (int | str): device to run inference on.
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
