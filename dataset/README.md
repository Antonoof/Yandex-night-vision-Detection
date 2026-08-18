# Building NVPDYF BDD100K in Ultralytics YOLO format

This guide covers running `prepare_bdd100k_nvpd.py` in a Kaggle Notebook.
The script:

- downloads BDD100K, or uses an already-downloaded copy;
- builds balanced `train` and `val` splits out of the source BDD100K `train`;
- moves **all** images from the source BDD100K `val` whose `timeofday` is
  `daytime` or `night` into `test`;
- writes labels directly in Ultralytics YOLO format;
- optionally creates a new Kaggle Dataset or publishes a new version of an
  existing one.

The split JSON files `train.json`, `val.json` and `test.json` are not
written into the final dataset.

## 1. Final classes

Seven classes with contiguous YOLO numbering:

| ID | Class |
|---:|---|
| 0 | `bicycle` |
| 1 | `bus` |
| 2 | `car` |
| 3 | `motorcycle` |
| 4 | `person` |
| 5 | `traffic light` |
| 6 | `truck` |

Additional rules:

- `rider` is merged into `person`;
- `bike` is merged into `bicycle`;
- `motor` and `motorbike` are merged into `motorcycle`;
- the `train` and `traffic sign` object classes are excluded from labeling
  entirely.

Note: dropping the `train` object class has nothing to do with the
`images/train` split directory. The training split itself is kept.

## 2. Final structure

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

Each image has a matching `.txt` in `labels/<split>`:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the `0`-`1` range.

## 3. Preparing the Kaggle Notebook

Upload `prepare_bdd100k_nvpd.py` to the current notebook, or add it as an
input resource.

Install dependencies:

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

Make sure the script is available:

```python
from pathlib import Path

SCRIPT_PATH = Path("prepare_bdd100k_nvpd.py")
assert SCRIPT_PATH.is_file(), f"Script not found: {SCRIPT_PATH}"
```

If the file is in a different directory, adjust the path in the commands
below accordingly.

## 4. Configuring Kaggle credentials

In the Kaggle Notebook settings, add two secrets:

- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`.

Then export them into environment variables:

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
```

Check authorization:

```python
!kaggle datasets list --mine
```

Do not print the `KAGGLE_KEY` value into the notebook log.

## 5. Option A: BDD100K not downloaded yet

The script downloads the source `solesensei/solesensei_bdd100k` dataset by
itself:

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

At this step the dataset is built locally under
`/kaggle/working/NVPD_UPDATE`, but is not published yet.

## 6. Option B: BDD100K already downloaded

If the source images and annotations already sit inside `bdd100k_raw`, add
`--skip-download`:

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

The script looks for these on its own:

- `images/100k/train`;
- `images/100k/val`;
- `bdd100k_labels_images_train.json`;
- `bdd100k_labels_images_val.json`.

So the outer structure of the BDD100K archive can vary.

## 7. Build and publish in one command

To automatically choose between creating and updating, use
`--publish --publish-mode auto`:

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

`auto` mode:

- creates a new dataset if `KAGGLE_USERNAME/nvpdyf-bdd100k` does not exist
  yet;
- creates a new version if the dataset already exists.

The `--public` flag only applies when creating a new dataset. Without it,
a new dataset is private.

## 8. Explicitly creating a new dataset

Use this mode only once, when the dataset does not exist yet:

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

## 9. Updating an existing dataset

To publish a new version, use:

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

## 10. Checking the local result

Check the core files and directories:

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
    assert path.exists(), f"Not found: {path}"

for split in ("train", "val", "test"):
    image_count = len(list((DATASET_ROOT / "images" / split).glob("*.jpg")))
    label_count = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))

    assert image_count == label_count
    print(f"{split}: images={image_count:,}, labels={label_count:,}")

for filename in ("train.json", "val.json", "test.json"):
    assert not (DATASET_ROOT / filename).exists(), filename

print("Local structure is correct")
```

Class ID check:

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

print("Found class_id values:", sorted(found_class_ids))
```

## 11. Checking the publication

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

## 12. Useful script parameters

See the full list of parameters:

```python
!python "/kaggle/working/prepare_bdd100k_nvpd.py" --help
```

Main parameters:

| Parameter | Purpose |
|---|---|
| `--raw-dir` | Source BDD100K directory |
| `--output-dir` | Output YOLO dataset directory |
| `--skip-download` | Do not re-download BDD100K |
| `--force-download` | Force re-downloading BDD100K |
| `--overwrite-output` | Fully rebuild a non-empty output directory |
| `--workers` | Number of processing/copying threads |
| `--random-state` | Seed for a reproducible train/val split |
| `--kaggle-owner` | Kaggle username or organization |
| `--output-dataset-slug` | Slug of the output Kaggle Dataset |
| `--output-dataset-title` | Display title of the Kaggle Dataset |
| `--publish` | Publish the result after building |
| `--publish-mode auto` | Create the dataset or update the existing one automatically |
| `--publish-mode create` | Explicitly create a new dataset |
| `--publish-mode version` | Explicitly create a new dataset version |
| `--public` | Make the new dataset public |
| `--version-message` | Message for the new version |

## 13. Common errors

### Output directory is not empty

```text
Output directory is not empty
```

Add `--overwrite-output`. The script only deletes and rebuilds the
explicitly specified output directory.

### Source BDD100K not found

If `--skip-download` is used, check the `--raw-dir` value. It must contain
the source BDD100K images and JSON annotations.

### Kaggle CLI not found

```text
The Kaggle CLI was not found
```

Install the package:

```python
!pip install -q --upgrade kaggle
```

### Stratification package not installed

```text
The iterative-stratification package is required
```

Install it:

```python
!pip install -q iterative-stratification
```

### Kaggle authorization error

Double-check the `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets, then run:

```python
!kaggle datasets list --mine
```

### Dataset already exists

Use `--publish-mode version` or `--publish-mode auto` instead of
`--publish-mode create`.
