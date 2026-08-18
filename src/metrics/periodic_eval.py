"""Night/day evaluation every K epochs, during training.

ultralytics already validates on the whole val split every epoch, but it
reports one aggregate mAP. The numbers this project exists to move - night
mAP and the night-day gap - are not in it, and by the time the run ends it is
too late to see *when* the gap started growing.

This runs our own night/day evaluation every K epochs and feeds it into the
same Comet experiment as the training curves, so both live on one x-axis.
It costs a real inference pass (~11 min for both splits on a T4), which is
why it is periodic rather than per-epoch.

Every failure here is swallowed on purpose: a broken side-metric must never
kill a multi-hour training run.
"""

import logging
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO

from src.metrics.detection import evaluate_detector
from src.utils.visualize import draw_comparison

logger = logging.getLogger(__name__)


class PeriodicNightDayEval:
    """Callback: measure night/day mAP and draw predictions every K epochs.

    Args:
        night_records (list[dict]): val frames with timeofday == "night".
        day_records (list[dict]): val frames with timeofday == "daytime".
        every_k (int): run every K epochs; 0 disables the callback entirely.
        eval_kwargs (dict): passed straight to evaluate_detector (imgsz, conf,
            iou, max_det, device, batch_size).
        comet_run (CometRunLogger | None): where to send the metrics.
        save_dir (str | Path): where per-epoch figures are written.
        n_samples (int): frames per figure; 0 turns figures off.
        viz_conf (float): human-facing confidence threshold for the figures.
    """

    def __init__(
        self,
        night_records,
        day_records,
        every_k,
        eval_kwargs,
        comet_run=None,
        save_dir=".",
        n_samples=6,
        viz_conf=0.25,
    ):
        self.night_records = night_records
        self.day_records = day_records
        self.every_k = int(every_k or 0)
        self.eval_kwargs = dict(eval_kwargs)
        self.comet_run = comet_run
        self.save_dir = Path(save_dir)
        self.n_samples = int(n_samples or 0)
        self.viz_conf = viz_conf
        # fixed frames, so figures from different epochs are comparable
        self._samples = {
            "night": sorted(night_records, key=lambda r: r["name"])[: self.n_samples],
            "day": sorted(day_records, key=lambda r: r["name"])[: self.n_samples],
        }
        self.history = []

    def attach(self, model):
        """Register on an ultralytics model whose train() is about to run."""
        if self.every_k <= 0:
            logger.info("периодическая оценка выключена (eval_every_k_epochs=0)")
            return
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        logger.info(
            "периодическая оценка: каждые %d эпох, night=%d day=%d кадров",
            self.every_k,
            len(self.night_records),
            len(self.day_records),
        )

    def _snapshot(self, trainer):
        """An inference-ready YOLO holding the weights as they are right now.

        Prefers the EMA weights: those are what ultralytics validates and what
        ends up in best.pt, so measuring anything else would describe a model
        we do not keep. Falls back to last.pt, which may lag by one epoch
        depending on the ultralytics version's callback order - hence the
        warning.
        """
        try:
            ema = getattr(getattr(trainer, "ema", None), "ema", None)
            net = ema if ema is not None else trainer.model
            probe = self.save_dir / "_periodic_probe.pt"
            probe.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": deepcopy(net).float().eval()}, probe)
            return YOLO(str(probe)), "ema"
        except Exception as e:
            last = getattr(trainer, "last", None)
            if last is not None and Path(last).is_file():
                logger.warning(
                    "снимок EMA не удался (%s), беру last.pt — он может "
                    "отставать на эпоху",
                    e,
                )
                return YOLO(str(last)), "last.pt"
            raise

    def _on_fit_epoch_end(self, trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if self.every_k <= 0 or epoch % self.every_k:
            return

        model = None
        try:
            model, source = self._snapshot(trainer)
            logger.info("эпоха %d: замеряю ночь/день (веса: %s)", epoch, source)

            night = evaluate_detector(
                model, self.night_records, desc=f"ep{epoch} night", **self.eval_kwargs
            )
            day = evaluate_detector(
                model, self.day_records, desc=f"ep{epoch} day", **self.eval_kwargs
            )

            gap = (day["map"] - night["map"]) / day["map"] * 100 if day["map"] else 0.0
            logger.info(
                "эпоха %d: night mAP=%.4f  day mAP=%.4f  разрыв=%.2f%%",
                epoch,
                night["map"],
                day["map"],
                gap,
            )
            self.history.append(
                {"epoch": epoch, "night": night, "day": day, "gap_pct": gap}
            )

            if self.comet_run is not None:
                self.comet_run.log_eval("night", night, epoch=epoch)
                self.comet_run.log_eval("day", day, epoch=epoch)
                self.comet_run.log_metrics({"gap/map_pct": gap}, epoch=epoch)

            self._draw(model, epoch)
        except Exception as e:
            logger.warning("периодическая оценка на эпохе %d не удалась: %s", epoch, e)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _draw(self, model, epoch):
        """Predictions on a fixed set of frames, one figure per split."""
        if self.n_samples <= 0:
            return
        for split, samples in self._samples.items():
            if not samples:
                continue
            try:
                figure = self.save_dir / f"predictions_{split}_ep{epoch:03d}.png"
                draw_comparison(
                    model,
                    samples,
                    figure,
                    imgsz=self.eval_kwargs["imgsz"],
                    conf=self.viz_conf,
                    device=self.eval_kwargs["device"],
                    title=f"{split} ep{epoch}",
                )
                if self.comet_run is not None:
                    self.comet_run.log_image(
                        figure, f"predictions_{split}", epoch=epoch
                    )
                logger.info("сохранено: %s", figure)
            except Exception as e:
                logger.warning(
                    "не удалось нарисовать %s на эпохе %d: %s", split, epoch, e
                )
