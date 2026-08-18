"""Zero-DCE image enhancement transform.

Run scripts from the repository root, like the rest of the project: the
network lives in ``src.model.zero_dce_net`` and is imported the same way as
every other ``src.*`` module.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping, TypeAlias

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.model.zero_dce_net import enhance_net_nopool


ImageInput: TypeAlias = Image.Image | np.ndarray | Tensor


class ZeroDCETransform:
    """Enhance a low-light RGB image with a pretrained Zero-DCE model.

    Accepted inputs:

    - ``PIL.Image.Image``;
    - ``numpy.ndarray`` with shape ``[H, W, C]`` and 3 or 4 channels;
    - ``torch.Tensor`` with shape ``[3, H, W]``.

    ``uint8`` input is converted to ``float32`` and divided by 255. Floating
    input whose maximum value is greater than 1 is also interpreted as the
    0..255 range. The returned tensor has shape ``[3, H, W]``, dtype
    ``float32``, values in ``[0, 1]``, RGB channel order, and resides on CPU.

    Args:
        weights_path: Path to the pretrained Zero-DCE state dictionary.
        device: PyTorch inference device, for example ``"cpu"`` or ``"cuda"``.
        probability: Probability of applying enhancement. When skipped, the
            normalized input tensor is returned.
        use_amp: Enable CUDA automatic mixed precision. Ignored on non-CUDA
            devices.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device = "cpu",
        probability: float = 1.0,
        use_amp: bool = True,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability must be between 0 and 1, got {probability}."
            )

        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Zero-DCE weights were not found: {self.weights_path}"
            )

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested, but CUDA is not available."
            )

        self.probability = float(probability)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.model = enhance_net_nopool().to(self.device)
        state_dict = self._load_state_dict(self.weights_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _load_state_dict(weights_path: Path) -> dict[str, Tensor]:
        """Load a plain state dict or a checkpoint containing ``state_dict``."""
        load_kwargs: dict[str, Any] = {
            "map_location": "cpu",
        }

        try:
            checkpoint = torch.load(
                weights_path,
                weights_only=True,
                **load_kwargs,
            )
        except TypeError:
            # Compatibility with older PyTorch versions without weights_only.
            checkpoint = torch.load(weights_path, **load_kwargs)

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "The Zero-DCE checkpoint must be a state dictionary or contain "
                "a 'state_dict' mapping."
            )

        nested_state_dict = checkpoint.get("state_dict")
        if isinstance(nested_state_dict, Mapping):
            checkpoint = nested_state_dict

        state_dict: dict[str, Tensor] = {}
        for key, value in checkpoint.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise TypeError(
                    "The Zero-DCE state dictionary must map string keys to "
                    "torch.Tensor values."
                )
            state_dict[key.removeprefix("module.")] = value

        if not state_dict:
            raise ValueError("The Zero-DCE state dictionary is empty.")

        return state_dict

    @staticmethod
    def _to_tensor(image: ImageInput) -> Tensor:
        """Convert a supported image input into a normalized CHW RGB tensor."""
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB")).copy()

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(
                    "Expected a NumPy HWC image with 3 or 4 channels, "
                    f"got shape={image.shape}."
                )

            # Discard the alpha channel when present.
            image = np.ascontiguousarray(image[..., :3])
            image = torch.from_numpy(image).permute(2, 0, 1)

        if not isinstance(image, Tensor):
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if image.ndim != 3:
            raise ValueError(
                f"Expected a CHW tensor, got shape={tuple(image.shape)}."
            )
        if image.shape[0] != 3:
            raise ValueError(
                "Zero-DCE expects three RGB channels, "
                f"got channels={image.shape[0]}."
            )
        if image.numel() == 0:
            raise ValueError("The input image is empty.")

        image = image.detach()
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)
            if image.max().item() > 1.0:
                image = image / 255.0

        return image.clamp(0.0, 1.0)

    @torch.inference_mode()
    def transform_batch(self, images: Sequence[ImageInput]) -> Tensor:
        """Normalize and enhance a batch of equally-sized RGB images.

        BDD100K frames all have the same resolution, so batching makes the
        one-off dataset preprocessing substantially faster on a GPU.  The
        result is always a CPU tensor with shape ``[B, 3, H, W]``.
        """
        if not images:
            return torch.empty((0, 3, 0, 0), dtype=torch.float32)

        normalized = [self._to_tensor(image) for image in images]
        shapes = {tuple(image.shape) for image in normalized}
        if len(shapes) != 1:
            raise ValueError(
                "Zero-DCE batch inputs must have one common shape, "
                f"got {sorted(shapes)}."
            )

        source = torch.stack(normalized, dim=0)
        if self.probability <= 0.0:
            return source.cpu()

        if self.probability >= 1.0:
            selected = torch.arange(len(source), dtype=torch.long)
        else:
            selected = torch.nonzero(
                torch.rand(len(source)) < self.probability,
                as_tuple=False,
            ).flatten()
        if selected.numel() == 0:
            return source.cpu()

        batch = source[selected].to(self.device, non_blocking=True)

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if self.use_amp
            else nullcontext()
        )

        with amp_context:
            _, enhanced, _ = self.model(batch)

        output = source.clone()
        output[selected] = enhanced.to(dtype=torch.float32).clamp(0.0, 1.0).cpu()
        return output

    @torch.inference_mode()
    def __call__(self, image: ImageInput) -> Tensor:
        """Normalize and optionally enhance one RGB image."""
        return self.transform_batch([image])[0]
