"""Comet ML logging for baseline evaluation runs (zero-shot vs. fine-tuned).

Separate from the per-step src.logger.logger console/file logging: this logs
one summary experiment per evaluation run (not per training step).
"""

import logging

logger = logging.getLogger(__name__)


def log_evaluation_run(
    run_name, tags, night, day, params, *, project_name, dataset_version, enabled
):
    """Log night/day metrics for one evaluation run as a Comet experiment.

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
