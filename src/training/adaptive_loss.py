"""Ultralytics callback implementing per-image adaptive detection loss."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .balance import BalanceSpec, build_balance_spec

logger = logging.getLogger(__name__)


def _slice_prediction_tree(value: Any, indices: Tensor, batch_size: int) -> Any:
    """Keep selected images in an arbitrary raw-prediction structure."""
    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, list):
        return [_slice_prediction_tree(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _slice_prediction_tree(item, indices, batch_size) for item in value
        )
    if isinstance(value, Mapping):
        return {
            key: _slice_prediction_tree(item, indices, batch_size)
            for key, item in value.items()
        }
    return value


def _scale_items(items: Any, factor: float | Tensor) -> Any:
    if isinstance(items, Mapping):
        return {key: _scale_items(value, factor) for key, value in items.items()}
    return items * factor


def _add_items(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return {key: _add_items(left[key], right[key]) for key in left}
    return left + right


class AdaptiveDetectionLoss:
    """Apply exact class-BCE weights and a night/day detection-loss weight.

    Ultralytics 8.4.120 natively reads ``model.class_weights`` inside its BCE
    classification component, so inverse-frequency weights are applied to
    every class/anchor rather than approximated at image level. The batch is
    split into daytime/night groups for the time-of-day multiplier, requiring
    at most two native loss calls. Dataset-mean normalization keeps the usual
    global scale without cancelling x5 on all-night batches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        spec: BalanceSpec,
        timeofday_by_name: Mapping[str, str],
    ) -> None:
        self.spec = spec
        self.timeofday_by_name = dict(timeofday_by_name)
        device = next(model.parameters()).device
        self.class_weights = torch.tensor(
            spec.class_weights,
            dtype=torch.float32,
            device=device,
        )
        model.class_weights = self.class_weights
        self.base = model.init_criterion()
        self.use_timeofday_weights = any(
            abs(weight - 1.0) > 1e-12
            for weight in spec.timeofday_weights.values()
        )

    def _sample_batch(self, batch: Mapping[str, Any], indices: Tensor) -> dict[str, Tensor]:
        batch_size = int(batch["img"].shape[0])
        old_batch_idx = batch["batch_idx"].long().flatten()
        remap = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=old_batch_idx.device,
        )
        remap[indices] = torch.arange(len(indices), device=old_batch_idx.device)
        selected = remap[old_batch_idx] >= 0
        sample = {
            "img": batch["img"].index_select(0, indices),
            "batch_idx": remap[old_batch_idx[selected]].to(batch["batch_idx"].dtype),
            "cls": batch["cls"][selected],
            "bboxes": batch["bboxes"][selected],
        }
        return sample

    def _groups(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        batch_size = int(batch["img"].shape[0])
        device = batch["img"].device
        if not self.use_timeofday_weights:
            return {"all": torch.arange(batch_size, device=device)}

        paths = batch.get("im_file")
        if paths is None or len(paths) != batch_size:
            raise KeyError(
                "Adaptive time-of-day loss requires batch['im_file']; "
                "the installed Ultralytics dataset did not provide it."
            )
        grouped: dict[str, list[int]] = {}
        for index, path in enumerate(paths):
            name = Path(str(path)).name
            timeofday = self.timeofday_by_name.get(name)
            if timeofday is None:
                if self.spec.strict_timeofday:
                    raise KeyError(f"No timeofday entry for training image {name!r}.")
                timeofday = "unknown"
            grouped.setdefault(timeofday, []).append(index)
        return {
            name: torch.tensor(indices, dtype=torch.long, device=device)
            for name, indices in grouped.items()
        }

    def __call__(self, predictions: Any, batch: Mapping[str, Any]):
        batch_size = int(batch["img"].shape[0])
        weighted_total = None
        weighted_items = None

        for timeofday, indices in self._groups(batch).items():
            sample_batch = self._sample_batch(batch, indices)
            sample_predictions = _slice_prediction_tree(
                predictions, indices, batch_size
            )
            sample_total, sample_items = self.base(
                sample_predictions, sample_batch
            )
            time_weight = (
                1.0
                if timeofday == "all"
                else self.spec.timeofday_weights.get(timeofday, 1.0)
            )
            term = sample_total.sum() * time_weight
            weighted_total = term if weighted_total is None else weighted_total + term
            weighted_items = _add_items(
                weighted_items,
                _scale_items(sample_items, len(indices) * time_weight),
            )

        if weighted_total is None or weighted_items is None:
            raise RuntimeError("Adaptive loss received an empty image batch.")

        normalizer = float(self.spec.normalization_factor)
        total = weighted_total / normalizer
        items = _scale_items(weighted_items, 1.0 / (batch_size * normalizer))
        return total, items


class AdaptiveLossController:
    """Resolve data weights and install the criterion on training start."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        class_names: Sequence[str],
        config: Mapping[str, Any],
    ) -> None:
        self.spec = build_balance_spec(records, class_names, config)
        self.timeofday_by_name: dict[str, str] = {}
        for record in records:
            name = Path(record["name"]).name
            value = str(record["timeofday"])
            previous = self.timeofday_by_name.setdefault(name, value)
            if previous != value:
                raise ValueError(
                    f"Duplicate training filename {name!r} has two timeofday values."
                )

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    def parameters(self) -> dict[str, float | int | str | bool]:
        return self.spec.as_parameters()

    def attach(self, yolo_model: Any) -> None:
        if not self.enabled:
            logger.info("adaptive loss: disabled; using native Ultralytics criterion")
            return

        def install(trainer: Any) -> None:
            try:
                from ultralytics.utils.torch_utils import unwrap_model
            except ImportError:
                from ultralytics.utils.torch_utils import de_parallel as unwrap_model
            model = de_parallel(trainer.model)
            model.criterion = AdaptiveDetectionLoss(
                model,
                self.spec,
                self.timeofday_by_name,
            )
            logger.info(
                "adaptive loss installed: time=%s class=%s normalization=%.4f",
                self.spec.timeofday_weights,
                {
                    name: round(weight, 4)
                    for name, weight in zip(
                        self.spec.class_names, self.spec.class_weights
                    )
                },
                self.spec.normalization_factor,
            )

        yolo_model.add_callback("on_train_start", install)
