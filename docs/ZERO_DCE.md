# `ZeroDCETransform`

`ZeroDCETransform` — вызываемая transform для осветления RGB-изображений с помощью предобученной модели Zero-DCE.

## Структура репозитория

```text
Yandex-night-vision-Detection/
├── src/
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── zero_dce.py      # ZeroDCETransform
│   └── model/
│       └── zero_dce_net.py  # содержит enhance_net_nopool
├── weights/
│   └── zero_dce_Epoch99.pth
├── docs/ZERO_DCE.md         # этот файл
└── ...
```

Изначально трансформ жил двумя отдельными пакетами в корне репозитория
(`transforms/` и `zero_dce/`). Он перенесён в `src/`, чтобы в проекте был
ровно один способ импорта — тот же `src.*`, которым пользуется весь
остальной код, — и чтобы не требовался ни `pip install -e .`, ни правка
`sys.path`.

В `zero_dce.py` используется абсолютный импорт:

```python
from src.model.zero_dce_net import enhance_net_nopool
```

Поэтому корень репозитория должен находиться в Python import path. Не добавляйте изменение `sys.path` внутрь `src/transforms/zero_dce.py`: модуль должен оставаться независимым от конкретного места клонирования репозитория.

## Рекомендуемый импорт

Благодаря реэкспорту в `src/transforms/__init__.py` используйте:

```python
from src.transforms import ZeroDCETransform
```

Это правильнее, чем:

```python
import src.transforms.ZeroDCETransform
```

Во втором варианте Python ожидает подмодуль с именем `ZeroDCETransform`, а не класс. В предоставленной структуре модуль называется `src.transforms.zero_dce`, а класс реэкспортируется на уровень пакета `src.transforms`.

При необходимости прямой импорт тоже доступен:

```python
from src.transforms.zero_dce import ZeroDCETransform
```

## Запуск

Специальной установки не требуется. Запускайте из корня репозитория — так же,
как `train.py` и `inference.py`:

```bash
cd Yandex-night-vision-Detection
python3 your_script.py
```

Текущая рабочая директория попадает в Python import path, поэтому доступны
все модули проекта:

```python
from src.transforms import ZeroDCETransform
from src.model.zero_dce_net import enhance_net_nopool
```

В Kaggle Notebook, если рабочая директория не корень репозитория, добавьте
его в `sys.path` **в точке входа** (не внутри самого трансформа):

```python
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/Yandex-night-vision-Detection").resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.transforms import ZeroDCETransform
```

## Зависимости

```bash
python -m pip install numpy Pillow torch
```

Также в репозитории должны присутствовать:

- `src/model/zero_dce_net.py` с функцией `enhance_net_nopool`;
- файл весов Zero-DCE, например `weights/zero_dce_Epoch99.pth`.

## Базовое использование

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

Результат всегда возвращается на CPU в формате RGB-тензора `[3, H, W]`, `float32`, диапазон `[0, 1]`.

Для эффективной обработки BDD100K доступен батчевый вызов:

```python
enhanced_batch = transform.transform_batch([image_1, image_2])
print(enhanced_batch.shape)  # [2, 3, H, W]
```

В `train.py` вручную вызывать его не нужно. Конфиг
`transforms=night_zero_dce` использует `timeofday.csv`, обрабатывает только
night-кадры и сохраняет результат в content-addressed cache. Поэтому train,
val, периодические метрики и inference получают одинаковые изображения.

## Вероятностное применение

Параметр `probability` позволяет использовать осветление как вероятностную аугментацию:

```python
transform = ZeroDCETransform(
    weights_path="weights/zero_dce_Epoch99.pth",
    device="cuda",
    probability=0.5,
)
```

При пропуске Zero-DCE transform всё равно нормализует вход и возвращает RGB-тензор `[3, H, W]` в диапазоне `[0, 1]`.

Допустимый диапазон `probability` — от `0.0` до `1.0`.

## Поддерживаемые входы

### PIL

```python
from PIL import Image

image = Image.open("example.jpg")
enhanced = transform(image)
```

Изображение автоматически преобразуется в RGB.

### NumPy

```python
import numpy as np

image = np.zeros((720, 1280, 3), dtype=np.uint8)
enhanced = transform(image)
```

Поддерживаются массивы `[H, W, 3]` и `[H, W, 4]`. Alpha-канал отбрасывается.

### PyTorch

```python
import torch

image = torch.rand(3, 720, 1280)
enhanced = transform(image)
```

Tensor должен иметь формат `[3, H, W]` и порядок каналов RGB.

## Использование в Dataset

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

## AMP и устройство

- `use_amp=True` включает `float16` autocast только при `device="cuda"`;
- на CPU AMP автоматически отключается;
- если запрошен CUDA device, но CUDA недоступна, конструктор сразу поднимет понятную ошибку;
- модель переводится в `eval()`;
- градиенты параметров модели отключаются;
- вызов выполняется под `torch.inference_mode()`.

## Формат checkpoint

Поддерживаются два распространённых варианта:

1. обычный state dict:

```python
torch.save(model.state_dict(), "Epoch99.pth")
```

2. checkpoint со вложенным `state_dict`:

```python
torch.save(
    {
        "state_dict": model.state_dict(),
        "epoch": 99,
    },
    "Epoch99.pth",
)
```

Префикс `module.`, добавленный `torch.nn.DataParallel`, удаляется автоматически.

## Быстрая проверка импорта

Из корня репозитория:

```bash
python -c "from src.transforms import ZeroDCETransform; print(ZeroDCETransform)"
```

Если возникает `ModuleNotFoundError: No module named 'src'`, значит корень репозитория не находится в Python import path: запускайте код из корня репозитория или добавьте корень в `sys.path` в точке входа.
