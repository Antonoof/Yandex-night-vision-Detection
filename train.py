import os

# comet_ml must be imported before torch/ultralytics, otherwise part of its
# autologging does not get patched in - this is a requirement of the library.
USE_COMET = bool(os.environ.get("COMET_API_KEY"))
if USE_COMET:
    import comet_ml  # noqa: F401

import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets import bdd100k
from src.logger import CometRunLogger, log_evaluation_run, setup_logging
from src.metrics import PeriodicNightDayEval, evaluate_night_day, print_results
from src.model import configure_nms_time_limit, log_head_info
from src.training import AdaptiveLossController
from src.transforms import build_dataset_preprocessor
from src.utils.init_utils import resolve_device, set_random_seed, split_devices
from src.utils.io_utils import ROOT_PATH
from src.utils.parallel import run_paired
from src.utils.visualize import draw_comparison

warnings.filterwarnings("ignore", category=UserWarning)


def _flatten_config(prefix, value):
    """Flatten a Hydra subtree into Comet-friendly scalar parameters."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            out.update(_flatten_config(f"{prefix}.{key}", item))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {prefix: ",".join(str(item) for item in value)}
    return {prefix: value}


@hydra.main(version_base=None, config_path="src/configs", config_name="adaptive")
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
        os.environ["COMET_WORKSPACE"] = str(config.writer.workspace)
        os.environ["COMET_PROJECT_NAME"] = str(config.writer.project_name)

    save_dir = ROOT_PATH / config.trainer.save_dir / config.trainer.run_name
    if (save_dir / "weights").exists() and not config.trainer.override:
        raise ValueError(
            f"'{save_dir}' already has a finished run in it. Change "
            "trainer.run_name or set trainer.override=true to overwrite it."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    OmegaConf.save(config=config, f=save_dir / "resolved_config.yaml", resolve=True)
    logger = logging.getLogger("train")

    device = resolve_device(config.trainer.device)
    # "0,1" is also the ultralytics convention for launching DDP training
    # across two GPUs; eval_devices reuses the same value to decide whether
    # the *other* independent halves of the pipeline (Zero-DCE on
    # train/val, night/day evaluation) can also run two single-GPU jobs at
    # once instead of one after another. See src/utils/parallel.py.
    eval_devices = split_devices(device)
    configure_nms_time_limit(config.metrics.nms_max_time_img)
    logger.info(
        "device: %s | eval_devices: %s | comet: %s",
        device,
        eval_devices,
        "enabled" if USE_COMET else "disabled",
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

    # Zero-DCE is deterministic deployment preprocessing, not a stochastic
    # augmentation. Materialize/cache it before building the YOLO view so the
    # exact same night pixels reach train, native val, periodic eval and plots.
    # train and val are independent frame sets, so with 2 devices they are
    # enhanced concurrently, one per GPU, instead of one after another - the
    # cache namespace is derived from the Zero-DCE weights/settings, not the
    # device, so both preprocessors write into the same cache directory
    # without colliding (they touch disjoint files).
    preprocessors = [
        build_dataset_preprocessor(config.transforms, ROOT_PATH, d)
        for d in eval_devices
    ]
    if len(preprocessors) > 1:
        jobs = [
            partial(preprocessors[0].prepare_split, train_used, "train"),
            partial(preprocessors[1].prepare_split, val_records, "val"),
        ]
        train_used, val_records = run_paired(jobs, eval_devices)
    else:
        prepared = preprocessors[0].prepare_splits(
            {"train": train_used, "val": val_records}
        )
        train_used = prepared["train"]
        val_records = prepared["val"]
    preprocessor = preprocessors[0]
    transform_info = {
        "transforms.zero_dce.enabled": preprocessor.active,
        "transforms.zero_dce.cache_namespace": preprocessor.cache_namespace,
        "transforms.zero_dce.apply_to": ",".join(sorted(preprocessor.apply_to)),
        "transforms.zero_dce.splits": ",".join(sorted(preprocessor.splits)),
    }
    del preprocessors, preprocessor

    data_yaml = bdd100k.build_yolo_dataset(
        {"train": train_used, "val": val_records}, ROOT_PATH / config.datasets.work_dir
    )

    adaptive_loss = AdaptiveLossController(
        train_used,
        bdd100k.CLASSES,
        config.loss,
    )
    adaptive_run = (
        bool(transform_info["transforms.zero_dce.enabled"])
        or adaptive_loss.enabled
    )
    run_tags = (
        ["stage:adaptive", "method:zero-dce-balanced-loss"]
        if adaptive_run
        else ["stage:baseline", "method:finetune-head"]
    )

    night_records = [r for r in val_records if r["timeofday"] == "night"]
    day_records = [r for r in val_records if r["timeofday"] == "daytime"]
    logger.info(
        "train: %d frames | val: night=%d, day=%d",
        len(train_used),
        len(night_records),
        len(day_records),
    )

    # eval_kwargs["device"] is deliberately a single device (eval_devices[0]),
    # never the raw possibly-multi-GPU `device`: predict()/draw_comparison()
    # move one model to one device and aren't safe to point at "0,1". Places
    # that can actually use a second GPU take `eval_devices` separately.
    eval_kwargs = dict(
        imgsz=config.trainer.imgsz,
        conf=config.metrics.conf,
        iou=config.metrics.iou,
        max_det=config.metrics.max_det,
        device=eval_devices[0],
        batch_size=config.trainer.batch,
    )

    zs_night = zs_day = None
    if config.trainer.eval_zero_shot:
        log_head_info(instantiate(config.model))

        zs_night, zs_day = evaluate_night_day(
            lambda: instantiate(config.model),
            night_records,
            day_records,
            eval_kwargs,
            eval_devices,
            class_map=bdd100k.COCO80_TO_OURS,
            desc_prefix="zero-shot ",
        )
        print_results(
            "A. ZERO-SHOT: COCO-pretrained YOLO, no fine-tuning", zs_night, zs_day
        )
        log_evaluation_run(
            "00_zero_shot_zero_dce" if adaptive_run else "00_baseline_zeroshot",
            [*run_tags, "method:zero-shot"],
            zs_night,
            zs_day,
            {"model": config.model.weights, "epochs": 0, **transform_info},
            project_name=config.writer.project_name,
            dataset_version=config.writer.dataset_version,
            enabled=USE_COMET,
        )

    # One experiment for the whole fine-tuning run: the per-epoch curves
    # (losses, lr, grad_norm, per-class AP/R) and the final night/day
    # metrics belong together, otherwise neither half can be read alone.
    with CometRunLogger(
        config.trainer.run_name,
        run_tags,
        {
            "model": config.model.weights,
            "epochs": config.trainer.epochs,
            "batch": config.trainer.batch,
            "imgsz": config.trainer.imgsz,
            "freeze": config.trainer.freeze,
            "seed": config.trainer.seed,
            "n_train": len(train_used),
            "n_val_night": len(night_records),
            "n_val_day": len(day_records),
            **_flatten_config("config.augment", config.augment),
            **_flatten_config("config.transforms", config.transforms),
            **_flatten_config("config.loss", config.loss),
            **transform_info,
            **adaptive_loss.parameters(),
        },
        project_name=config.writer.project_name,
        dataset_version=config.writer.dataset_version,
        enabled=USE_COMET,
    ) as comet_run:
        # fresh COCO weights: zero-shot eval (if it ran) already built and
        # touched its own throwaway instances, so training starts from its
        # own clean checkpoint reload instead of reusing one of those.
        model = instantiate(config.model)
        adaptive_loss.attach(model)
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
            devices=eval_devices,
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
        # proof that the head was rebuilt: nc goes 80 (COCO) -> 7 (ours), and
        # the cv3 branch's output channels follow. Compare with the head dump
        # logged above for the pretrained model.
        log_head_info(best)

        ft_night, ft_day = evaluate_night_day(
            lambda: instantiate(config.model, weights=str(best_weights)),
            night_records,
            day_records,
            eval_kwargs,
            eval_devices,
            desc_prefix="fine-tuned ",
        )
        print_results(
            "B. FINE-TUNED: YOLO with a new head, fine-tuned on BDD100K",
            ft_night,
            ft_day,
        )

        comet_run.log_eval("night", ft_night)
        comet_run.log_eval("day", ft_day)

        # Разбор ошибок глазами: метрики говорят, ЧТО просело, а картинки —
        # почему именно (пропуск / рамка мимо / ложное срабатывание на блике).
        # Кадры фиксированные (первые по имени), поэтому их можно сравнивать
        # между запусками; случайная выборка такого не позволяет.
        n_samples = config.trainer.num_visualization_samples
        for split, split_records in (("night", night_records), ("day", day_records)):
            samples = sorted(split_records, key=lambda r: r["name"])[:n_samples]
            if not samples:
                logger.info("нет кадров '%s' для отрисовки, пропускаю", split)
                continue
            figure = Path(weights_save_dir) / f"predictions_{split}.png"
            draw_comparison(
                best,
                samples,
                figure,
                imgsz=config.trainer.imgsz,
                conf=config.trainer.visualization_conf,
                device=eval_devices[0],
                title=split,
            )
            comet_run.log_image(figure, f"predictions_{split}")
            logger.info("сохранено: %s", figure)

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
        "config": OmegaConf.to_container(config, resolve=True),
        "transforms": transform_info,
        "adaptive_loss": adaptive_loss.parameters(),
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
