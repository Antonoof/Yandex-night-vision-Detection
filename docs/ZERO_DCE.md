# `ZeroDCETransform`

`ZeroDCETransform` — a callable transform that brightens RGB images using a
pretrained Zero-DCE model.

## Repository layout

```text
Yandex-night-vision-Detection/
├── src/
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── zero_dce.py      # ZeroDCETransform
│   └── model/
│       └── zero_dce_net.py  # contains enhance_net_nopool
├── weights/
│   └── zero_dce_Epoch99.pth
├── docs/ZERO_DCE.md         # this file
└── ...
```

The transform originally lived as two separate top-level packages
(`transforms/` and `zero_dce/`). It was moved into `src/` so the project has
exactly one way to import things — the same `src.*` the rest of the code
uses — and so it needs neither `pip install -e .` nor a `sys.path` edit.

`zero_dce.py` uses an absolute import:

```python
from src.model.zero_dce_net import enhance_net_nopool
```

So the repository root must be on the Python import path. Do not add a
`sys.path` change inside `src/transforms/zero_dce.py`: the module must stay
independent of where the repository happens to be cloned.

## Recommended import

Thanks to the re-export in `src/transforms/__init__.py`, use:

```python
from src.transforms import ZeroDCETransform
```

This is more correct than:

```python
import src.transforms.ZeroDCETransform
```

In the second form Python expects a submodule named `ZeroDCETransform`, not
a class. In the actual layout the module is called
`src.transforms.zero_dce`, and the class is re-exported at the
`src.transforms` package level.

A direct import also works if you need it:

```python
from src.transforms.zero_dce import ZeroDCETransform
```

## Running it

No special install is required. Run it from the repository root, the same
way as `train.py` and `inference.py`:

```bash
cd Yandex-night-vision-Detection
python3 your_script.py
```

The current working directory ends up on the Python import path, so every
project module is available:

```python
from src.transforms import ZeroDCETransform
from src.model.zero_dce_net import enhance_net_nopool
```

In a Kaggle Notebook, if the working directory is not the repository root,
add it to `sys.path` **at the entry point** (not inside the transform
itself):

```python
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/Yandex-night-vision-Detection").resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.transforms import ZeroDCETransform
```

## Dependencies

```bash
python -m pip install numpy Pillow torch
```

The repository must also contain:

- `src/model/zero_dce_net.py` with the `enhance_net_nopool` function;
- a Zero-DCE weights file, e.g. `weights/zero_dce_Epoch99.pth`.

## Basic usage

```python
from pathlib import Path

import torch
from PIL import Image

from src.transforms import ZeroDCETransform


REPO_DIR = Path("/kaggle/working/Yandex-night-vision-Detection")

transform = ZeroDCETransform(
    weights_path=REPO_DIR / "weights/zero_dce_Epoch99.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
    probability=1.0,
    use_amp=True,
)

image = Image.open("example.jpg").convert("RGB")
enhanced = transform(image)

print(enhanced.shape)  # torch.Size([3, H, W])
print(enhanced.dtype)  # torch.float32
print(enhanced.min().item(), enhanced.max().item())
```

The result is always returned on the CPU as an RGB tensor `[3, H, W]`,
`float32`, range `[0, 1]`.

## Probabilistic application

The `probability` parameter lets you use the brightening as a probabilistic
augmentation:

```python
transform = ZeroDCETransform(
    weights_path="weights/zero_dce_Epoch99.pth",
    device="cuda",
    probability=0.5,
)
```

When Zero-DCE is skipped, the transform still normalizes the input and
returns an RGB tensor `[3, H, W]` in range `[0, 1]`.

The valid range for `probability` is `0.0` to `1.0`.

## Supported inputs

### PIL

```python
from PIL import Image

image = Image.open("example.jpg")
enhanced = transform(image)
```

The image is automatically converted to RGB.

### NumPy

```python
import numpy as np

image = np.zeros((720, 1280, 3), dtype=np.uint8)
enhanced = transform(image)
```

Arrays of shape `[H, W, 3]` and `[H, W, 4]` are supported. The alpha channel
is dropped.

### PyTorch

```python
import torch

image = torch.rand(3, 720, 1280)
enhanced = transform(image)
```

The tensor must be in `[3, H, W]` format with RGB channel order.

## Usage inside a Dataset

```python
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class NightImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image
```

```python
dataset = NightImageDataset(
    image_paths=["night_1.jpg", "night_2.jpg"],
    transform=transform,
)
```

## AMP and device

- `use_amp=True` enables `float16` autocast only when `device="cuda"`;
- AMP is automatically disabled on CPU;
- if a CUDA device is requested but CUDA is unavailable, the constructor
  raises a clear error immediately;
- the model is switched to `eval()`;
- gradients are disabled on the model's parameters;
- the call runs under `torch.inference_mode()`.

## Checkpoint format

Two common variants are supported:

1. a plain state dict:

```python
torch.save(model.state_dict(), "Epoch99.pth")
```

2. a checkpoint with a nested `state_dict`:

```python
torch.save(
    {
        "state_dict": model.state_dict(),
        "epoch": 99,
    },
    "Epoch99.pth",
)
```

The `module.` prefix added by `torch.nn.DataParallel` is stripped
automatically.

## Quick import check

From the repository root:

```bash
python -c "from src.transforms import ZeroDCETransform; print(ZeroDCETransform)"
```

If you get `ModuleNotFoundError: No module named 'src'`, the repository
root is not on the Python import path: run the code from the repository
root, or add the root to `sys.path` at the entry point.
