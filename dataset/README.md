# Сборка NVPDYF BDD100K в формате Ultralytics YOLO

Инструкция описывает запуск `prepare_bdd100k_nvpd.py` в Kaggle Notebook. Скрипт:

- скачивает или использует уже скачанный BDD100K;
- формирует сбалансированные `train` и `val` из исходной BDD100K `train`;
- переносит в `test` **все** изображения исходной BDD100K `val`, у которых `timeofday` равен `daytime` или `night`;
- создаёт разметку непосредственно в формате Ultralytics YOLO;
- при необходимости создаёт новый Kaggle Dataset или публикует новую версию существующего.

Split-JSON-файлы `train.json`, `val.json` и `test.json` в итоговый датасет не записываются.

## 1. Итоговые классы

Используется семь классов с непрерывной нумерацией YOLO:

| ID | Класс |
|---:|---|
| 0 | `bicycle` |
| 1 | `bus` |
| 2 | `car` |
| 3 | `motorcycle` |
| 4 | `person` |
| 5 | `traffic light` |
| 6 | `truck` |

Дополнительные правила:

- `rider` объединяется с `person`;
- `bike` объединяется с `bicycle`;
- `motor` и `motorbike` объединяются с `motorcycle`;
- классы объектов `train` и `traffic sign` полностью исключаются из разметки.

Важно: удаление класса объектов `train` не связано с директорией split `images/train`. Обучающая выборка сохраняется.

## 2. Итоговая структура

```text
NVPD_UPDATE/
├── data.yaml
├── dataset-metadata.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Каждому изображению соответствует одноимённый `.txt` в `labels/<split>`:

```text
class_id x_center y_center width height
```

Координаты нормализованы в диапазон от `0` до `1`.

## 3. Подготовка Kaggle Notebook

Загрузите `prepare_bdd100k_nvpd.py` в текущий Notebook или добавьте файл как входной ресурс.

Установите зависимости:

```python
!pip install -q \
    kaggle \
    iterative-stratification \
    PyYAML \
    pandas \
    numpy \
    scikit-learn \
    Pillow \
    tqdm
```

Убедитесь, что скрипт доступен:

```python
from pathlib import Path

SCRIPT_PATH = Path("prepare_bdd100k_nvpd.py")
assert SCRIPT_PATH.is_file(), f"Скрипт не найден: {SCRIPT_PATH}"
```

Если файл находится в другой директории, укажите соответствующий путь в командах ниже.

## 4. Настройка Kaggle credentials

В настройках Kaggle Notebook добавьте два секрета:

- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`.

Затем экспортируйте их в переменные окружения:

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
```

Проверьте авторизацию:

```python
!kaggle datasets list --mine
```

Не выводите значение `KAGGLE_KEY` в лог Notebook.

## 5. Вариант A: BDD100K ещё не скачан

Скрипт самостоятельно скачает исходный датасет `solesensei/solesensei_bdd100k`:

```python
!python "prepare_bdd100k_nvpd.py" \
    --raw-dir "bdd100k_raw" \
    --output-dir "NVPD_UPDATE" \
    --overwrite-output \
    --workers 8 \
    --random-state 42 \
    --kaggle-owner "$KAGGLE_USERNAME" \
    --output-dataset-slug "nvpdyf-bdd100k" \
    --output-dataset-title "nvpdyf_bdd100k"
```

На этом шаге датасет собирается локально в `/kaggle/working/NVPD_UPDATE`, но ещё не публикуется.

## 6. Вариант B: BDD100K уже скачан

Если исходные изображения и аннотации уже находятся внутри `bdd100k_raw`, добавьте `--skip-download`:

```python
!python "prepare_bdd100k_nvpd.py" \
    --raw-dir "bdd100k_raw" \
    --output-dir "NVPD_UPDATE" \
    --skip-download \
    --overwrite-output \
    --workers 8 \
    --random-state 42 \
    --kaggle-owner "$KAGGLE_USERNAME" \
    --output-dataset-slug "nvpdyf-bdd100k" \
    --output-dataset-title "nvpdyf_bdd100k"
```

Скрипт самостоятельно ищет:

- `images/100k/train`;
- `images/100k/val`;
- `bdd100k_labels_images_train.json`;
- `bdd100k_labels_images_val.json`.

Поэтому внешняя структура архива BDD100K может отличаться.

## 7. Сборка и публикация одной командой

Для автоматического выбора между созданием и обновлением используйте `--publish --publish-mode auto`:

```python
!python "prepare_bdd100k_nvpd.py" \
    --raw-dir "bdd100k_raw" \
    --output-dir "NVPD_UPDATE" \
    --skip-download \
    --overwrite-output \
    --workers 8 \
    --random-state 42 \
    --kaggle-owner "$KAGGLE_USERNAME" \
    --output-dataset-slug "nvpdyf-bdd100k" \
    --output-dataset-title "nvpdyf_bdd100k" \
    --publish \
    --publish-mode auto \
    --public
```

Режим `auto`:

- создаёт новый датасет, если `KAGGLE_USERNAME/nvpdyf-bdd100k` ещё не существует;
- создаёт новую версию, если датасет уже существует.

Флаг `--public` применяется только при создании нового датасета. Без него новый датасет будет приватным.

## 8. Явное создание нового датасета

Используйте этот режим только один раз, когда датасет ещё не существует:

```python
!python "prepare_bdd100k_nvpd.py" \
    --raw-dir "bdd100k_raw" \
    --output-dir "NVPD_UPDATE" \
    --skip-download \
    --overwrite-output \
    --kaggle-owner "$KAGGLE_USERNAME" \
    --output-dataset-slug "nvpdyf-bdd100k" \
    --output-dataset-title "nvpdyf_bdd100k" \
    --publish \
    --publish-mode create \
    --public
```

## 9. Обновление существующего датасета

Для публикации новой версии используйте:

```python
!python "prepare_bdd100k_nvpd.py" \
    --raw-dir "bdd100k_raw" \
    --output-dir "NVPD_UPDATE" \
    --skip-download \
    --overwrite-output \
    --kaggle-owner "$KAGGLE_USERNAME" \
    --output-dataset-slug "nvpdyf-bdd100k" \
    --output-dataset-title "nvpdyf_bdd100k" \
    --publish \
    --publish-mode version \
    --version-message "Rebuilt YOLO dataset with all daytime/night BDD100K val images in test"
```

## 10. Проверка локального результата

Проверьте основные файлы и директории:

```python
from pathlib import Path

DATASET_ROOT = Path("NVPD_UPDATE")

required_paths = [
    DATASET_ROOT / "data.yaml",
    DATASET_ROOT / "dataset-metadata.json",
    DATASET_ROOT / "images/train",
    DATASET_ROOT / "images/val",
    DATASET_ROOT / "images/test",
    DATASET_ROOT / "labels/train",
    DATASET_ROOT / "labels/val",
    DATASET_ROOT / "labels/test",
]

for path in required_paths:
    assert path.exists(), f"Не найдено: {path}"

for split in ("train", "val", "test"):
    image_count = len(list((DATASET_ROOT / "images" / split).glob("*.jpg")))
    label_count = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))

    assert image_count == label_count
    print(f"{split}: images={image_count:,}, labels={label_count:,}")

for filename in ("train.json", "val.json", "test.json"):
    assert not (DATASET_ROOT / filename).exists(), filename

print("Локальная структура корректна")
```

Проверка ID классов:

```python
valid_class_ids = set(range(7))
found_class_ids = set()

for label_path in (DATASET_ROOT / "labels").rglob("*.txt"):
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        assert len(parts) == 5

        class_id = int(parts[0])
        assert class_id in valid_class_ids
        found_class_ids.add(class_id)

print("Найденные class_id:", sorted(found_class_ids))
```

## 11. Проверка публикации

```python
DATASET_REF = f"{os.environ['KAGGLE_USERNAME']}/nvpdyf-bdd100k"
print(DATASET_REF)
```

```python
!kaggle datasets status "$KAGGLE_USERNAME/nvpdyf-bdd100k"
```

```python
!kaggle datasets files "$KAGGLE_USERNAME/nvpdyf-bdd100k" --page-size 200
```

## 12. Полезные параметры скрипта

Посмотреть полный список параметров:

```python
!python "/kaggle/working/prepare_bdd100k_nvpd.py" --help
```

Основные параметры:

| Параметр | Назначение |
|---|---|
| `--raw-dir` | Директория исходного BDD100K |
| `--output-dir` | Директория итогового YOLO-датасета |
| `--skip-download` | Не скачивать BDD100K повторно |
| `--force-download` | Принудительно скачать BDD100K заново |
| `--overwrite-output` | Полностью пересоздать непустую выходную директорию |
| `--workers` | Количество потоков обработки и копирования |
| `--random-state` | Seed для воспроизводимого формирования train/val |
| `--kaggle-owner` | Kaggle username или организация |
| `--output-dataset-slug` | Slug итогового Kaggle Dataset |
| `--output-dataset-title` | Отображаемое название Kaggle Dataset |
| `--publish` | Опубликовать результат после сборки |
| `--publish-mode auto` | Создать датасет или обновить существующий автоматически |
| `--publish-mode create` | Явно создать новый датасет |
| `--publish-mode version` | Явно создать новую версию датасета |
| `--public` | Сделать новый датасет публичным |
| `--version-message` | Комментарий к новой версии |

## 13. Частые ошибки

### Выходная директория не пустая

```text
Output directory is not empty
```

Добавьте `--overwrite-output`. Скрипт удалит и заново создаст только явно указанную выходную директорию.

### Исходный BDD100K не найден

Если используется `--skip-download`, проверьте значение `--raw-dir`. Внутри него должны находиться исходные изображения и JSON-аннотации BDD100K.

### Не найден Kaggle CLI

```text
The Kaggle CLI was not found
```

Установите пакет:

```python
!pip install -q --upgrade kaggle
```

### Не установлен пакет стратификации

```text
The iterative-stratification package is required
```

Установите его:

```python
!pip install -q iterative-stratification
```

### Ошибка авторизации Kaggle

Повторно проверьте секреты `KAGGLE_USERNAME` и `KAGGLE_KEY`, затем выполните:

```python
!kaggle datasets list --mine
```

### Датасет уже существует

Вместо `--publish-mode create` используйте `--publish-mode version` или `--publish-mode auto`.
