import json
import logging
import warnings

import hydra
from hydra.utils import instantiate

from src.datasets import bdd100k
from src.logger import setup_logging
from src.metrics import evaluate_detector, print_results
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.visualize import draw_predictions

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Loads a trained checkpoint and evaluates it
    on one BDD100K split, separately for night and day frames.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    save_path = ROOT_PATH / config.inferencer.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path)
    logger = logging.getLogger("inference")

    device = resolve_device(config.inferencer.device)

    split = config.inferencer.split
    data_root = bdd100k.find_dataset_root(ROOT_PATH / config.datasets.input_dir)
    records = bdd100k.load_records(data_root, split)
    night_records = [r for r in records if r["timeofday"] == "night"]
    day_records = [r for r in records if r["timeofday"] == "daytime"]
    logger.info(
        "%s split: night=%d frames, day=%d frames",
        split,
        len(night_records),
        len(day_records),
    )
    if split == "test":
        # Held out for the whole project, so every number measured on it is a
        # one-shot claim. Say so in the log: a reader six months from now needs
        # to know which rows may have been tuned against and which may not.
        logger.info(
            "ВНИМАНИЕ: это test. На нём ничего не подбиралось; "
            "test намеренно обогащён ночью (42.8%% против 20.2%% в val), "
            "поэтому его числа не сравнимы с val напрямую."
        )

    model = instantiate(config.model, weights=config.inferencer.weights)

    eval_kwargs = dict(
        imgsz=config.inferencer.imgsz,
        conf=config.metrics.conf,
        iou=config.metrics.iou,
        max_det=config.metrics.max_det,
        device=device,
        batch_size=config.inferencer.batch,
    )
    night_metrics = evaluate_detector(model, night_records, desc="night", **eval_kwargs)
    day_metrics = evaluate_detector(model, day_records, desc="day", **eval_kwargs)
    print_results(
        f"Evaluation [{split}]: {config.inferencer.weights}",
        night_metrics,
        day_metrics,
    )

    results_path = save_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "weights": config.inferencer.weights,
                "split": split,
                "night": night_metrics,
                "day": day_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("results saved to %s", results_path)

    if config.inferencer.save_visualizations:
        fixed_samples = sorted(night_records, key=lambda r: r["name"])[
            : config.inferencer.num_visualization_samples
        ]
        if not fixed_samples:
            logger.info("no night frames to visualize, skipping predictions.png")
        else:
            predictions_path = save_path / "predictions.png"
            draw_predictions(
                model,
                fixed_samples,
                predictions_path,
                imgsz=config.inferencer.imgsz,
                conf=config.inferencer.visualization_conf,
                device=device,
            )
            logger.info("visualizations saved to %s", predictions_path)


if __name__ == "__main__":
    main()
