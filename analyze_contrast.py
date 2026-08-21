"""Measure object-vs-background contrast on the val split, night vs day.

Answers the project's standing hypothesis - "we miss objects at night because
they blend into the dark background" - with numbers instead of impressions,
and explains what any tone-mapping method could have achieved at best.

Runs on CPU. Without `analysis.weights` it needs neither torch nor ultralytics.
"""

import logging
import warnings
from pathlib import Path

import hydra

from src.analysis import (
    collect,
    contrast_control_summary,
    draw_localization_grid,
    localization_decomposition,
    localization_summary,
    match_predictions,
    plot_decomposition,
    plot_localization,
    plot_overview,
    recall_by_contrast,
    summarize,
    write_csv,
)
from src.datasets import bdd100k
from src.logger import setup_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="analysis")
def main(config):
    """
    Per-box contrast statistics for one split, split by time of day.

    Args:
        config (DictConfig): hydra config, see src/configs/analysis.yaml.
    """
    save_path = ROOT_PATH / config.analysis.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path)
    logger = logging.getLogger("analyze_contrast")

    if config.analysis.split == "test":
        raise ValueError(
            "the test split stays closed until the end of the project - "
            "analyse 'val'."
        )

    data_root = bdd100k.find_dataset_root(ROOT_PATH / config.datasets.input_dir)
    records = bdd100k.load_records(data_root, config.analysis.split)
    if config.analysis.max_frames:
        records = records[: config.analysis.max_frames]

    night = sum(r["timeofday"] == "night" for r in records)
    logger.info(
        "%s: %d кадров (ночь=%d, день=%d)",
        config.analysis.split,
        len(records),
        night,
        len(records) - night,
    )

    rows = collect(records, ring_frac=config.analysis.ring_frac)
    logger.info("боксов измерено: %d", len(rows))

    curves = None
    if config.analysis.weights:
        # imported here so the contrast half runs without ultralytics installed
        from ultralytics import YOLO

        weights = Path(config.analysis.weights)
        if not weights.is_absolute():
            weights = ROOT_PATH / weights
        logger.info("сопоставляю с предсказаниями: %s", weights)

        match_predictions(
            YOLO(str(weights)),
            records,
            rows,
            imgsz=config.analysis.imgsz,
            conf=config.analysis.conf,
            iou_thr=config.analysis.iou,
            device=config.analysis.device,
            batch_size=config.analysis.batch,
        )
        curves = recall_by_contrast(rows)
    else:
        logger.info(
            "analysis.weights не задан: считаю только распределения контраста, "
            "без recall"
        )

    summarize(rows)
    write_csv(rows, save_path / "boxes.csv")
    plot_overview(rows, save_path / "contrast.png", recall_curves=curves)

    if config.analysis.weights:
        # The localization half only exists once boxes have been matched.
        localization_summary(rows)
        plot_localization(rows, save_path / "localization.png")
        # "Which boxes are wrong" (above) and "how they are wrong" (below).
        # The second question is what tells a fix apart from a workaround:
        # a displaced box and an unsteady one need different treatment.
        localization_decomposition(rows)
        contrast_control_summary(rows)
        plot_decomposition(rows, save_path / "decomposition.png")
        images_dir = data_root / "images" / config.analysis.split
        for tod, suffix in (("night", "night"), ("daytime", "day")):
            draw_localization_grid(
                rows,
                images_dir,
                save_path / f"boxes_{suffix}.png",
                timeofday=tod,
                n=config.analysis.num_examples,
                iou_range=tuple(config.analysis.example_iou_range),
            )

    logger.info("готово: %s", save_path)


if __name__ == "__main__":
    main()
