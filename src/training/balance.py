"""Pure-Python construction of time-of-day and class balancing weights."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _timeofday(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {"day": "daytime", "daylight": "daytime", "nighttime": "night"}
    return aliases.get(value, value)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_positive(values: Sequence[float], name: str) -> None:
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} weights must be finite and positive: {values}")


@dataclass(frozen=True)
class BalanceSpec:
    """Resolved weights used by the adaptive criterion."""

    enabled: bool
    class_names: tuple[str, ...]
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]
    timeofday_counts: dict[str, int]
    timeofday_weights: dict[str, float]
    normalization: str
    normalization_factor: float
    strict_timeofday: bool

    def record_weight(self, record: Mapping[str, Any]) -> float:
        return self.timeofday_weights.get(_timeofday(record["timeofday"]), 1.0)

    def as_parameters(self) -> dict[str, float | int | str | bool]:
        out: dict[str, float | int | str | bool] = {
            "loss.adaptive.enabled": self.enabled,
            "loss.adaptive.normalization": self.normalization,
            "loss.adaptive.normalization_factor": self.normalization_factor,
        }
        for name, count, weight in zip(
            self.class_names, self.class_counts, self.class_weights
        ):
            safe = name.replace(" ", "_")
            out[f"loss.class_count.{safe}"] = count
            out[f"loss.class_weight.{safe}"] = weight
        for name, count in sorted(self.timeofday_counts.items()):
            out[f"loss.timeofday_count.{name}"] = count
        for name, weight in sorted(self.timeofday_weights.items()):
            out[f"loss.timeofday_weight.{name}"] = weight
        return out


def build_balance_spec(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    config: Mapping[str, Any],
) -> BalanceSpec:
    """Resolve configured weights from the actual selected training records."""
    adaptive = config.get("adaptive") or config
    enabled = bool(adaptive.get("enabled", False))
    class_cfg = adaptive.get("classes") or {}
    time_cfg = adaptive.get("timeofday") or {}

    class_counts_counter = Counter(
        int(class_id)
        for record in records
        for class_id, *_ in record.get("boxes", [])
    )
    class_counts = tuple(
        class_counts_counter.get(index, 0) for index in range(len(class_names))
    )
    time_counts = Counter(_timeofday(record["timeofday"]) for record in records)

    class_weights = [1.0] * len(class_names)
    if enabled and bool(class_cfg.get("enabled", True)):
        mode = str(class_cfg.get("mode", "inverse_frequency")).lower()
        if mode == "manual":
            manual = class_cfg.get("weights") or {}
            class_weights = [
                float(manual.get(name, manual.get(str(index), 1.0)))
                for index, name in enumerate(class_names)
            ]
        elif mode == "inverse_frequency":
            power = float(class_cfg.get("power", 1.0))
            smoothing = float(class_cfg.get("smoothing", 1.0))
            lower = float(class_cfg.get("min_weight", 0.25))
            upper = float(class_cfg.get("max_weight", 5.0))
            total = sum(class_counts) + smoothing * len(class_counts)
            denominator_classes = max(1, len(class_counts))
            raw = [
                (total / (denominator_classes * (count + smoothing))) ** power
                for count in class_counts
            ]
            if bool(class_cfg.get("normalize", True)) and sum(class_counts):
                mean = sum(
                    count * weight for count, weight in zip(class_counts, raw)
                ) / sum(class_counts)
                raw = [weight / max(mean, 1e-12) for weight in raw]
            class_weights = [_clip(weight, lower, upper) for weight in raw]
        else:
            raise ValueError(
                "loss.adaptive.classes.mode must be inverse_frequency or manual."
            )

    time_weights = {name: 1.0 for name in time_counts}
    if enabled and bool(time_cfg.get("enabled", True)):
        mode = str(time_cfg.get("mode", "manual")).lower()
        lower = float(time_cfg.get("min_weight", 0.25))
        upper = float(time_cfg.get("max_weight", 10.0))
        if mode == "manual":
            manual = {
                _timeofday(name): float(weight)
                for name, weight in (time_cfg.get("weights") or {}).items()
            }
            time_weights = {
                name: _clip(manual.get(name, 1.0), lower, upper)
                for name in time_counts
            }
            for name, weight in manual.items():
                time_weights.setdefault(name, _clip(weight, lower, upper))
        elif mode == "inverse_frequency":
            power = float(time_cfg.get("power", 1.0))
            smoothing = float(time_cfg.get("smoothing", 1.0))
            reference = max(time_counts.values(), default=1)
            time_weights = {
                name: _clip(
                    ((reference + smoothing) / (count + smoothing)) ** power,
                    lower,
                    upper,
                )
                for name, count in time_counts.items()
            }
        else:
            raise ValueError(
                "loss.adaptive.timeofday.mode must be inverse_frequency or manual."
            )

    _validate_positive(class_weights, "class")
    _validate_positive(list(time_weights.values()), "timeofday")

    normalization = str(adaptive.get("normalization", "dataset_mean")).lower()
    if normalization not in {"dataset_mean", "none"}:
        raise ValueError("loss.adaptive.normalization must be dataset_mean or none.")

    provisional = BalanceSpec(
        enabled=enabled,
        class_names=tuple(str(name) for name in class_names),
        class_counts=class_counts,
        class_weights=tuple(float(value) for value in class_weights),
        timeofday_counts=dict(time_counts),
        timeofday_weights={key: float(value) for key, value in time_weights.items()},
        normalization=normalization,
        normalization_factor=1.0,
        strict_timeofday=bool(adaptive.get("strict_timeofday", True)),
    )
    if enabled and normalization == "dataset_mean" and records:
        normalization_factor = sum(
            provisional.record_weight(record) for record in records
        ) / len(records)
    else:
        normalization_factor = 1.0
    if not math.isfinite(normalization_factor) or normalization_factor <= 0:
        raise ValueError(
            f"Adaptive-loss normalization is invalid: {normalization_factor}."
        )

    return BalanceSpec(
        **{
            **provisional.__dict__,
            "normalization_factor": float(normalization_factor),
        }
    )
