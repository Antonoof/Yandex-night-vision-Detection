import os

# comet_ml must be imported before torch/ultralytics, otherwise part of its
# autologging does not get patched in - this is a requirement of the library.
USE_COMET = bool(os.environ.get("COMET_API_KEY"))
if USE_COMET:
    import comet_ml  # noqa: F401

import json
import logging
import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate

from src.datasets import bdd100k
from src.logger import CometRunLogger, log_evaluation_run, setup_logging
from src.metrics import PeriodicNightDayEval, evaluate_detector, print_results
from src.model import log_head_info
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.visualize import draw_comparison

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Builds the YOLO-format dataset from BDD100K,
    fine-tunes a YOLO detection head on it, and evaluates the result
    separately on night and day frames.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    if USE_COMET:
        os.environ["COMET_PROJECT_NAME"] = config.writer.project_name

    save_dir = ROOT_PATH / config.trainer.save_dir / config.trainer.run_name
    if (save_dir / "weights").exists() and not config.trainer.override:
        raise ValueError(
            f"'{save_dir}' already has a finished run in it. Change "
            "trainer.run_name or set trainer.override=true to overwrite it."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    logger = logging.getLogger("train")

    device = resolve_device(config.trainer.device)
    logger.info(
        "device: %s | comet: %s", device, "enabled" if USE_COMET else "disabled"
    )

    data_root = bdd100k.find_dataset_root(ROOT_PATH / config.datasets.input_dir)
    train_records = bdd100k.load_records(data_root, "train")
    val_records = bdd100k.load_records(data_root, "val")
    bdd100k.describe_split_balance(train_records, val_records)

    train_used = bdd100k.subsample(
        train_records, config.datasets.max_train_images, config.trainer.seed
    )
    # splits are independent by construction, but the check costs milliseconds
    # while an unnoticed leak would invalidate every metric in the project
    assert not (
        {r["name"] for r in train_used} & {r["name"] for r in val_records}
    ), "the same frame is present in both train and val!"

    if config.datasets.night_oversample > 1:
        before = len(train_used)
        train_used = bdd100k.oversample_night(
            train_used, config.datasets.night_oversample
        )
        logger.info(
            "night_oversample=%d: train %d -> %d frames",
            config.datasets.night_oversample,
            before,
            len(train_used),
        )

    data_yaml = bdd100k.build_yolo_dataset(
        {"train": train_used, "val": val_records}, ROOT_PATH / config.datasets.work_dir
    )

    night_records = [r for r in val_records if r["timeofday"] == "night"]
    day_records = [r for r in val_records if r["timeofday"] == "daytime"]
    logger.info(
        "train: %d frames | val: night=%d, day=%d",
        len(train_used),
        len(night_records),
        len(day_records),
    )

    eval_kwargs = dict(
        imgsz=config.trainer.imgsz,
        conf=config.metrics.conf,
        iou=config.metrics.iou,
        max_det=config.metrics.max_det,
        device=device,
        batch_size=config.trainer.batch,
    )

    zs_night = zs_day = None
    if config.trainer.eval_zero_shot:
        pretrained = instantiate(config.model)
        log_head_info(pretrained)

        zs_night = evaluate_detector(
            pretrained,
            night_records,
            class_map=bdd100k.COCO80_TO_OURS,
            desc="zero-shot night",
            **eval_kwargs,
        )
        zs_day = evaluate_detector(
            pretrained,
            day_records,
            class_map=bdd100k.COCO80_TO_OURS,
            desc="zero-shot day",
            **eval_kwargs,
        )
        print_results(
            "A. ZERO-SHOT: COCO-pretrained YOLO, no fine-tuning", zs_night, zs_day
        )
        log_evaluation_run(
            "00_baseline_zeroshot",
            ["stage:baseline", "method:zero-shot"],
            zs_night,
            zs_day,
            {"model": config.model.weights, "epochs": 0},
            project_name=config.writer.project_name,
            dataset_version=config.writer.dataset_version,
            enabled=USE_COMET,
        )

    # One experiment for the whole fine-tuning run: the per-epoch curves
    # (losses, lr, grad_norm, per-class AP/R) and the final night/day
    # metrics belong together, otherwise neither half can be read alone.
    with CometRunLogger(
        config.trainer.run_name,
        ["stage:baseline", "method:finetune-head"],
        {
            "model": config.model.weights,
            "epochs": config.trainer.epochs,
            "batch": config.trainer.batch,
            "imgsz": config.trainer.imgsz,
            "freeze": config.trainer.freeze,
            "seed": config.trainer.seed,
            "night_oversample": config.datasets.night_oversample,
            "n_train": len(train_used),
            "n_val_night": len(night_records),
            "n_val_day": len(day_records),
            **{f"augment.{k}": v for k, v in config.augment.items()},
        },
        project_name=config.writer.project_name,
        dataset_version=config.writer.dataset_version,
        enabled=USE_COMET,
    ) as comet_run:
        # fresh COCO weights: if eval_zero_shot ran, "pretrained" was already
        # touched by inference, so we reload the model before training
        model = instantiate(config.model)
        comet_run.attach(model)

        # Night/day mAP while training: ultralytics logs one aggregate mAP per
        # epoch, which cannot show the gap this project is about. Failures in
        # here are swallowed - a side metric must not kill a multi-hour run.
        periodic = PeriodicNightDayEval(
            night_records,
            day_records,
            every_k=config.trainer.eval_every_k_epochs,
            eval_kwargs=eval_kwargs,
            comet_run=comet_run,
            save_dir=save_dir,
            n_samples=config.trainer.num_visualization_samples,
            viz_conf=config.trainer.visualization_conf,
        )
        periodic.attach(model)

        results = model.train(
            data=str(data_yaml),
            epochs=config.trainer.epochs,
            imgsz=config.trainer.imgsz,
            batch=config.trainer.batch,
            freeze=config.trainer.freeze or None,
            seed=config.trainer.seed,
            device=device,
            project=str(ROOT_PATH / config.trainer.save_dir),
            name=config.trainer.run_name,
            exist_ok=True,
            patience=config.trainer.patience,
            workers=config.trainer.workers,
            verbose=True,
            **config.augment,
        )

        weights_save_dir = getattr(results, "save_dir", None) or model.trainer.save_dir
        best_weights = Path(weights_save_dir) / "weights" / "best.pt"
        logger.info("best checkpoint: %s", best_weights)

        best = instantiate(config.model, weights=str(best_weights))
        # proof that the head was rebuilt: nc goes 80 (COCO) -> 8 (ours), and
        # the cv3 branch's output channels follow. Compare with the head dump
        # logged above for the pretrained model.
        log_head_info(best)

        ft_night = evaluate_detector(
            best, night_records, desc="fine-tuned night", **eval_kwargs
        )
        ft_day = evaluate_detector(
            best, day_records, desc="fine-tuned day", **eval_kwargs
        )
        print_results(
            "B. FINE-TUNED: YOLO with a new head, fine-tuned on BDD100K",
            ft_night,
            ft_day,
        )

        comet_run.log_eval("night", ft_night)
        comet_run.log_eval("day", ft_day)

        # Eyeballing the errors: metrics say WHAT dropped, figures say WHY
        # (missed object / box off target / false positive on a glare).
        # Frames are fixed (first by name), so they stay comparable between
        # runs; a random sample would not allow that.
        n_samples = config.trainer.num_visualization_samples
        for split, split_records in (("night", night_records), ("day", day_records)):
            samples = sorted(split_records, key=lambda r: r["name"])[:n_samples]
            if not samples:
                logger.info("no '%s' frames to draw, skipping", split)
                continue
            figure = Path(weights_save_dir) / f"predictions_{split}.png"
            draw_comparison(
                best,
                samples,
                figure,
                imgsz=config.trainer.imgsz,
                conf=config.trainer.visualization_conf,
                device=device,
                title=split,
            )
            comet_run.log_image(figure, f"predictions_{split}")
            logger.info("saved: %s", figure)

    metrics = {"fine-tuned/night": ft_night, "fine-tuned/day": ft_day}
    if config.trainer.eval_zero_shot:
        metrics = {"zero-shot/night": zs_night, "zero-shot/day": zs_day, **metrics}

    summary = {
        "dataset_version": config.writer.dataset_version,
        "model": config.model.weights,
        "imgsz": config.trainer.imgsz,
        "epochs": config.trainer.epochs,
        "batch": config.trainer.batch,
        "seed": config.trainer.seed,
        "freeze": config.trainer.freeze,
        "n_images": {
            "train": len(train_used),
            "val_night": len(night_records),
            "val_day": len(day_records),
        },
        "metrics": metrics,
        "per_epoch_night_day": periodic.history,
    }
    results_path = Path(weights_save_dir) / "results.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("summary saved to %s", results_path)


if __name__ == "__main__":
    main()
