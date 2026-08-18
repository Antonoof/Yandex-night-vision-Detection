"""Comet ML logging.

Two entry points, for two different shapes of run:

* ``log_evaluation_run`` - a one-shot evaluation with no training behind it
  (the zero-shot baseline). Opens an experiment, writes the night/day
  metrics, closes it.
* ``CometRunLogger`` - a full fine-tuning run. Owns a single experiment for
  the whole run, so the training curves (losses, lr, grad_norm, per-class
  AP/R) and the final night/day evaluation end up in the *same* place.

Both are separate from ``src.logger.logger``, which only handles the
console/file logging.
"""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ultralytics' own metric names -> the names used in this project.
# Renaming is a one-way door (see docs/EXPERIMENTS.md): once runs are logged
# under these names, changing them breaks comparison with every earlier run.
_KEY_RENAMES = {
    "metrics/precision(B)": "val/precision",
    "metrics/recall(B)": "val/recall",
    "metrics/mAP50(B)": "val/mAP50",
    "metrics/mAP50-95(B)": "val/mAP50-95",
    "fitness": "val/fitness",
}


def _normalize_key(key):
    """Map an ultralytics metric name onto this project's naming.

    Args:
        key (str): metric name as ultralytics reports it.
    Returns:
        key (str): name to log under.
    """
    if key in _KEY_RENAMES:
        return _KEY_RENAMES[key]
    if key.startswith("lr/"):  # lr/pg0 -> train/lr_pg0
        return f"train/lr_{key.split('/', 1)[1]}"
    return key


def _clean(payload):
    """Drop non-numeric entries and normalize names before logging.

    Args:
        payload (dict): raw {name: value} metrics.
    Returns:
        payload (dict): {normalized name: float}.
    """
    out = {}
    for key, value in payload.items():
        try:
            out[_normalize_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@contextmanager
def _ultralytics_comet_disabled():
    """Temporarily switch off ultralytics' built-in Comet integration.

    It opens a *second* experiment for the same run and closes it when
    training ends, which would (a) split one run across two experiments and
    (b) leave the final night/day metrics with nowhere to go, since they are
    only computed after training finishes. The previous value is restored on
    the way out.
    """
    try:
        from ultralytics.utils import SETTINGS
    except Exception:  # ultralytics too old/new to have SETTINGS - skip
        yield
        return

    previous = SETTINGS.get("comet", True)
    try:
        SETTINGS["comet"] = False
        yield
    finally:
        SETTINGS["comet"] = previous


def _grad_norm(model, scaler):
    """Total L2 norm of the gradients, undoing AMP loss scaling.

    Called just before the optimizer step, while the gradients still exist:
    ultralytics zeroes them inside ``optimizer_step``.

    Args:
        model (nn.Module): model being trained.
        scaler (GradScaler): the trainer's AMP scaler. Gradients are scaled
            by ``scaler.get_scale()`` at this point, so the raw norm has to
            be divided by it to be comparable across steps.
    Returns:
        norm (float): gradient L2 norm, or 0.0 if there are no gradients.
    """
    import torch

    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    scale = 1.0
    try:
        scale = float(scaler.get_scale())
    except Exception:
        pass

    total = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    return float(total) / max(scale, 1e-12)


class CometRunLogger:
    """One Comet experiment for one fine-tuning run.

    Usage::

        with CometRunLogger(...) as run:
            run.attach(model)
            model.train(...)
            run.log_eval("night", night_metrics)

    Every callback body is wrapped in try/except on purpose: a bug in
    logging must never take down a training run that has been going for
    hours.

    Args:
        run_name (str): experiment name shown in the Comet UI.
        tags (list[str]): experiment tags.
        params (dict): hyperparameters to log.
        project_name (str): Comet project to log into.
        dataset_version (str): logged as a parameter, for comparability.
        enabled (bool): if False, every method is a no-op.
        grad_norm_every (int): compute the gradient norm once per this many
            optimizer steps. Each computation forces a GPU sync, so doing it
            on every step would slow training down for no extra insight.
    """

    def __init__(
        self,
        run_name,
        tags,
        params,
        *,
        project_name,
        dataset_version,
        enabled,
        grad_norm_every=50,
    ):
        self.run_name = run_name
        self.enabled = enabled
        self.grad_norm_every = grad_norm_every
        self.exp = None
        self._grad = {"sum": 0.0, "n": 0, "step": 0}
        self._last_epoch = 0  # so final metrics land at the end of the x-axis

        if not enabled:
            logger.info("[Comet disabled] not logging %s", run_name)
            return

        import comet_ml

        self.exp = comet_ml.Experiment(
            project_name=project_name,
            auto_metric_logging=False,
            auto_param_logging=False,
        )
        self.exp.set_name(run_name)
        self.exp.add_tags(tags)
        self.exp.log_parameters({"dataset_version": dataset_version, **params})

    # -- context manager -------------------------------------------------

    def __enter__(self):
        self._disable_ctx = _ultralytics_comet_disabled()
        self._disable_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._disable_ctx.__exit__(*exc)
        self.end()
        return False

    # -- wiring into ultralytics -----------------------------------------

    def attach(self, model):
        """Register the training callbacks on an ultralytics model.

        Args:
            model (ultralytics.YOLO): model whose ``train()`` will be called.
        """
        if not self.enabled:
            return
        model.add_callback("on_train_start", self._on_train_start)
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        model.add_callback("on_train_end", self._on_train_end)

    def _on_train_start(self, trainer):
        """Wrap ``optimizer_step`` so gradients can be measured before they
        are zeroed. The original method is still what does the work - this
        only observes."""
        try:
            original = trainer.optimizer_step

            def observed():
                self._grad["step"] += 1
                if self._grad["step"] % self.grad_norm_every == 0:
                    self._grad["sum"] += _grad_norm(trainer.model, trainer.scaler)
                    self._grad["n"] += 1
                original()

            trainer.optimizer_step = observed
        except Exception as e:
            logger.warning("could not install the grad-norm probe: %s", e)

    def _on_fit_epoch_end(self, trainer):
        """Log everything available at the end of a train+val epoch."""
        if self.exp is None:
            return
        try:
            payload = {}

            # box = regression loss (where the object is, CIoU + DFL),
            # cls = classification loss (what the object is, BCE)
            payload.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
            # val losses + aggregate precision/recall/mAP
            payload.update(trainer.metrics)
            payload.update(trainer.lr)

            if self._grad["n"]:
                payload["train/grad_norm"] = self._grad["sum"] / self._grad["n"]
                self._grad["sum"], self._grad["n"] = 0.0, 0

            payload.update(self._per_class_metrics(trainer))

            epoch = int(getattr(trainer, "epoch", 0)) + 1
            self._last_epoch = epoch
            self.exp.log_metrics(_clean(payload), step=epoch, epoch=epoch)
        except Exception as e:
            logger.warning("comet epoch logging failed: %s", e)

    def _on_train_end(self, trainer):
        """Log the confusion matrix of the final validation pass.

        ultralytics re-validates with best.pt right before this callback, so
        the matrix belongs to the checkpoint we actually keep. It answers in
        one glance the question the scalar metrics cannot: are classes being
        confused with each other, or with the background?
        """
        if self.exp is None:
            return
        try:
            cm = trainer.validator.confusion_matrix.matrix
            names = getattr(trainer.validator, "names", None) or trainer.data["names"]
            labels = [str(names[i]) for i in sorted(names)] + ["background"]
            self.exp.log_confusion_matrix(
                matrix=[[int(v) for v in row] for row in cm],
                labels=labels,
                title="confusion matrix (best.pt, весь val)",
                file_name="confusion-matrix.json",
            )
        except Exception as e:
            logger.warning("could not log the confusion matrix: %s", e)

    @staticmethod
    def _per_class_metrics(trainer):
        """Per-class AP and recall from the validator's own metrics.

        ultralytics only reports the aggregate mAP in ``trainer.metrics``;
        the per-class breakdown lives on the validator.

        Args:
            trainer (BaseTrainer): the ultralytics trainer.
        Returns:
            payload (dict): {metric name: value}, empty if unavailable.
        """
        payload = {}
        try:
            box = trainer.validator.metrics.box
            names = getattr(trainer.validator, "names", None) or trainer.data["names"]
            for i, class_id in enumerate(box.ap_class_index):
                name = str(names[int(class_id)]).replace(" ", "_")
                payload[f"val/AP50-95_{name}"] = box.ap[i]
                payload[f"val/AP50_{name}"] = box.ap50[i]
                payload[f"val/R_{name}"] = box.r[i]
                payload[f"val/P_{name}"] = box.p[i]
        except Exception as e:
            logger.warning("per-class metrics unavailable: %s", e)
        return payload

    # -- final evaluation -------------------------------------------------

    def log_eval(self, split, metrics, epoch=None):
        """Log a night/day evaluation into this same experiment.

        Each call is one point on the night/day curves. Without ``epoch`` the
        point is stamped with the last training step, so a final evaluation
        lines up with the end of the training curves instead of landing at
        step 0; ``PeriodicNightDayEval`` passes the epoch it measured to build
        an actual curve.

        Args:
            split (str): "night" or "day".
            metrics (dict): output of ``evaluate_detector``.
            epoch (int | None): epoch to stamp; defaults to the last logged one.
        """
        self.log_metrics({f"{split}/{k}": v for k, v in metrics.items()}, epoch=epoch)

    def log_metrics(self, payload, epoch=None):
        """Log an arbitrary flat dict of scalars at a given epoch.

        Args:
            payload (dict): metric name -> value.
            epoch (int | None): epoch to stamp; defaults to the last logged one.
        """
        if self.exp is None:
            return
        step = epoch if epoch is not None else self._last_epoch
        try:
            self.exp.log_metrics(_clean(payload), step=step or None, epoch=step or None)
        except Exception as e:
            logger.warning("comet eval logging failed: %s", e)

    def log_image(self, path, name, epoch=None):
        """Attach a figure to the experiment.

        Args:
            path (str | Path): image file to upload.
            name (str): name shown in the Comet UI's Graphics tab.
            epoch (int | None): step to stamp; defaults to the last logged one.
                Figures logged under the same name at different steps become
                a scrubbable sequence in Comet.
        """
        if self.exp is None:
            return
        step = epoch if epoch is not None else self._last_epoch
        try:
            self.exp.log_image(str(path), name=name, step=step or None)
            logger.info("comet: изображение '%s' отправлено", name)
        except Exception as e:
            logger.warning("could not log image '%s': %s", name, e)

    def end(self):
        """Close the experiment."""
        if self.exp is None:
            return
        try:
            url = self.exp.url
            self.exp.end()
            logger.info("sent to comet: %s -> %s", self.run_name, url)
        except Exception as e:
            logger.warning("could not close the comet experiment: %s", e)
        finally:
            self.exp = None


def log_evaluation_run(
    run_name, tags, night, day, params, *, project_name, dataset_version, enabled
):
    """Log night/day metrics for a training-free evaluation run.

    Used for the zero-shot baseline, which has no training curves to attach
    - for fine-tuning runs use CometRunLogger instead.

    Args:
        run_name (str): experiment name shown in the Comet UI.
        tags (list[str]): experiment tags.
        night (dict): output of evaluate_detector on the night subset.
        day (dict): output of evaluate_detector on the day subset.
        params (dict): extra hyperparameters to log (model, epochs, ...).
        project_name (str): Comet project to log into.
        dataset_version (str): logged as a parameter, for run comparability.
        enabled (bool): if False, this is a no-op (e.g. no COMET_API_KEY).
    """
    if not enabled:
        logger.info("[Comet disabled] skipping %s", run_name)
        return

    import comet_ml

    exp = comet_ml.Experiment(
        project_name=project_name, auto_metric_logging=False, auto_param_logging=False
    )
    exp.set_name(run_name)
    exp.add_tags(tags)
    exp.log_parameters({"dataset_version": dataset_version, **params})
    for split, metrics in (("night", night), ("day", day)):
        exp.log_metrics({f"{split}/{k}": v for k, v in metrics.items()})
    exp.end()
    logger.info("sent to comet: %s -> %s", run_name, exp.url)
