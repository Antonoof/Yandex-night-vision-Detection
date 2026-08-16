"""Save a figure comparing model predictions to ground-truth box counts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.bdd100k import CLASSES


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
