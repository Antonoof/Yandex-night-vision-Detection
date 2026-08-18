"""A transform that changes nothing - the control for preprocessing runs."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class IdentityTransform:
    """Return the frame unchanged, in ZeroDCETransform's output format.

    Writing a frame through the preprocessing path re-encodes it as JPEG, and
    that alone shifts pixels by a level or two. Comparing ``preprocess=none``
    against ``preprocess=zero_dce`` therefore measures enhancement *plus*
    re-encoding, and a small effect could be either.

    Running this instead isolates it: ``none`` vs ``reencode`` is the price of
    re-encoding, ``reencode`` vs ``zero_dce`` is what Zero-DCE actually did.

    Accepts the same inputs as ZeroDCETransform and returns ``[3, H, W]``
    float32 in ``[0, 1]``, RGB, on CPU.
    """

    def __call__(self, image: Image.Image | np.ndarray | Tensor) -> Tensor:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB")).copy()
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image[..., :3])).permute(
                2, 0, 1
            )
        if not isinstance(image, Tensor):
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        image = image.detach()
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)
            if image.max().item() > 1.0:
                image = image / 255.0
        return image.clamp(0.0, 1.0).cpu()
