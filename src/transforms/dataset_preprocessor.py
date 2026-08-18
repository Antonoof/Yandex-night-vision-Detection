"""Deterministic, time-of-day-aware preprocessing for YOLO dataset views.

Zero-DCE is a deployment preprocessor rather than a random training
augmentation: the detector must see the same enhanced night domain during
training, validation, periodic evaluation and inference.  This module writes
enhanced night frames to a content-addressed cache once, while daytime frames
continue to reference the original JPEGs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .zero_dce import ZeroDCETransform

logger = logging.getLogger(__name__)


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _torch_device(value: Any, trainer_device: Any) -> torch.device:
    """Translate Ultralytics device notation (``0``/``"0,1"``) to torch."""
    selected = trainer_device if str(value) == "same_as_trainer" else value
    if isinstance(selected, int):
        return torch.device(f"cuda:{selected}")
    text = str(selected)
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    if "," in text and all(part.strip().isdigit() for part in text.split(",")):
        return torch.device(f"cuda:{text.split(',', 1)[0].strip()}")
    return torch.device(text)


def _normalized_timeofday(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {"day": "daytime", "daylight": "daytime", "nighttime": "night"}
    return aliases.get(value, value)


class ConditionalDatasetPreprocessor:
    """Cache Zero-DCE outputs for configured time-of-day values and splits."""

    def __init__(
        self,
        *,
        enabled: bool,
        weights_path: str | Path,
        cache_dir: str | Path,
        apply_to: Sequence[str],
        splits: Sequence[str],
        device: torch.device,
        use_amp: bool,
        batch_size: int,
        overwrite_cache: bool,
        jpeg_quality: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.apply_to = {_normalized_timeofday(value) for value in apply_to}
        self.splits = {str(value) for value in splits}
        self.cache_dir = Path(cache_dir)
        self.batch_size = int(batch_size)
        self.overwrite_cache = bool(overwrite_cache)
        self.jpeg_quality = int(jpeg_quality)

        if self.batch_size < 1:
            raise ValueError("transforms.zero_dce.batch_size must be positive.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("transforms.cache.jpeg_quality must be in [1, 100].")

        self.transform = None
        self.cache_namespace = "disabled"
        if self.enabled:
            weights_path = Path(weights_path)
            signature = hashlib.sha256()
            signature.update(weights_path.read_bytes())
            signature.update(
                json.dumps(
                    {
                        "transform": "zero_dce",
                        "use_amp": bool(use_amp),
                        "jpeg_quality": self.jpeg_quality,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            self.cache_namespace = f"zero_dce_{signature.hexdigest()[:12]}"
            self.transform = ZeroDCETransform(
                weights_path=weights_path,
                device=device,
                probability=1.0,
                use_amp=use_amp,
            )

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.apply_to) and bool(self.splits)

    def _applies(self, record: Mapping[str, Any], split: str) -> bool:
        return split in self.splits and _normalized_timeofday(
            record["timeofday"]
        ) in self.apply_to

    def _cache_path(self, record: Mapping[str, Any], split: str) -> Path:
        return self.cache_dir / self.cache_namespace / split / Path(record["name"])

    def prepare_splits(
        self, splits: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Return record copies whose enhanced frames point into the cache."""
        return {
            split: self.prepare_split(records, split)
            for split, records in splits.items()
        }

    def prepare_split(
        self, records: Sequence[Mapping[str, Any]], split: str
    ) -> list[dict[str, Any]]:
        prepared = [dict(record) for record in records]
        if not self.active or split not in self.splits:
            return prepared

        pending: list[tuple[int, Mapping[str, Any], Path]] = []
        cached = 0
        for index, record in enumerate(records):
            if not self._applies(record, split):
                continue
            destination = self._cache_path(record, split)
            prepared[index]["path"] = destination
            if destination.is_file() and not self.overwrite_cache:
                cached += 1
            else:
                pending.append((index, record, destination))

        logger.info(
            "Zero-DCE %s: target=%d cached=%d pending=%d device=%s",
            split,
            cached + len(pending),
            cached,
            len(pending),
            getattr(self.transform, "device", "disabled"),
        )
        if not pending:
            return prepared

        assert self.transform is not None
        progress = tqdm(
            range(0, len(pending), self.batch_size),
            desc=f"zero-dce {split}",
            leave=False,
        )
        for start in progress:
            chunk = pending[start : start + self.batch_size]
            images = []
            for _, record, _ in chunk:
                with Image.open(record["path"]) as source:
                    images.append(source.convert("RGB").copy())

            enhanced = self.transform.transform_batch(images)
            for output, (_, _, destination) in zip(enhanced, chunk):
                destination.parent.mkdir(parents=True, exist_ok=True)
                array = (
                    output.mul(255.0)
                    .round()
                    .clamp(0, 255)
                    .to(dtype=torch.uint8)
                    .permute(1, 2, 0)
                    .numpy()
                )
                image = Image.fromarray(np.ascontiguousarray(array))
                temporary = destination.with_name(f"{destination.name}.tmp")
                if destination.suffix.lower() in {".jpg", ".jpeg"}:
                    image.save(
                        temporary,
                        format="JPEG",
                        quality=self.jpeg_quality,
                        subsampling=0,
                    )
                else:
                    image.save(
                        temporary,
                        format=(destination.suffix.lstrip(".") or "PNG").upper(),
                    )
                temporary.replace(destination)

        return prepared


def build_dataset_preprocessor(
    config: Mapping[str, Any], root: str | Path, trainer_device: Any
) -> ConditionalDatasetPreprocessor:
    """Build the configured dataset preprocessor with resolved local paths."""
    root = Path(root)
    zero_dce = config.get("zero_dce") or {}
    cache = config.get("cache") or {}
    enabled = bool(zero_dce.get("enabled", False))
    weights_path = _resolve_path(
        zero_dce.get("weights_path", "weights/zero_dce_Epoch99.pth"), root
    )
    cache_dir = _resolve_path(cache.get("dir", "data/preprocessed_cache"), root)
    device = _torch_device(
        zero_dce.get("device", "same_as_trainer"), trainer_device
    )
    return ConditionalDatasetPreprocessor(
        enabled=enabled,
        weights_path=weights_path,
        cache_dir=cache_dir,
        apply_to=zero_dce.get("apply_to", ["night"]),
        splits=zero_dce.get("splits", ["train", "val"]),
        device=device,
        use_amp=bool(zero_dce.get("use_amp", True)),
        batch_size=int(zero_dce.get("batch_size", 8)),
        overwrite_cache=bool(cache.get("overwrite", False)),
        jpeg_quality=int(cache.get("jpeg_quality", 95)),
    )
