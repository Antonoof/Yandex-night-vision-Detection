# Night Vision Project

Fine-tunes a YOLOv8 detector on [BDD100K](https://www.kaggle.com/datasets/nikitakukuzey/nvpd-bdd100k)
and evaluates it separately on night and day frames, to measure and reduce
the night-time performance gap. Runs the same way locally, on Kaggle, and on
a cloud GPU VM — everything is a plain Python script driven by
[Hydra](https://hydra.cc) configs, no notebook required.

## Requirements

- Python 3.11 (see `.python-version`; newer versions may not have wheels for
  the pinned/older ML packages yet).
- A GPU is not required but strongly recommended for anything beyond a smoke
  test — YOLO training on CPU is very slow.

## Setup

### Local machine

```bash
# create + activate an env (conda or venv+pyenv both work)
conda create -n project_env python=3.11 && conda activate project_env
# or: ~/.pyenv/versions/3.11/bin/python3 -m venv project_env && source project_env/bin/activate

# install torch matching your platform/CUDA version first, see
# https://pytorch.org/get-started/locally/ - e.g. for CPU-only:
pip install torch torchvision

pip install -r requirements.txt
pre-commit install
```

### Kaggle notebook

Kaggle already ships a CUDA-linked `torch`/`torchvision` - do not let pip
reinstall them. In a notebook cell:

```bash
!pip install -q ultralytics torchmetrics pycocotools hydra-core
```

Note: the package is `hydra-core`, not `hydra` - the latter is an
unrelated, abandoned PyPI package and will fail to build.

Then clone this repo into the notebook's working directory:

```bash
!git clone <this-repo-url> repo
%cd repo
```

To use Comet tracking, pull the API key from Kaggle Secrets before invoking
the script (the key just needs to be in the environment - `!python` inherits
it from the notebook process):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret("COMET_API_KEY")
```

Point `datasets.input_dir` at `/kaggle/input` itself, not at a specific
dataset subfolder - Kaggle's exact mount path varies by notebook (e.g.
`/kaggle/input/datasets/<owner>/<slug>/` vs. the older
`/kaggle/input/<slug>/`), and the dataset search is recursive, so pointing
at the whole input root finds it either way:

```bash
!python train.py datasets.input_dir=/kaggle/input trainer.device=0
```

If it still can't find the dataset (`FileNotFoundError` from
`find_dataset_root`), run `!find /kaggle/input -maxdepth 5 -iname val.json`
to see exactly where it landed and point `datasets.input_dir` there
directly.

### Cloud GPU VM

Same as the local setup: install a CUDA-matched `torch`/`torchvision` for
the VM's driver version, then `pip install -r requirements.txt`. `device:
auto` in the configs picks up the GPU automatically.

## Data

Point `datasets.input_dir` (default `data/bdd100k`) at a folder that
contains, somewhere inside it (nesting is fine - the code searches for it):

```
images/{train,val,test}/*.jpg
train.json
val.json
```

Each `<split>.json` holds per-frame annotations with a `timeofday` field and
`box_yolo` boxes (already normalized `[cx, cy, w, h]`) with `coco_category`
labels - this is the schema produced by the
[nvpd-bdd100k](https://www.kaggle.com/datasets/nikitakukuzey/nvpd-bdd100k)
dataset. `src/datasets/bdd100k.py` converts it into the YOLO-format dataset
(`images/`, `labels/`, `data.yaml`) that `ultralytics` expects, written to
`datasets.work_dir` (default `data/bdd_yolo`).

### nvpdyf-bdd100k (EDA baseline)

`nvpdyf-bdd100k` (mounted at `/kaggle/input/nvpdyf-bdd100k` on Kaggle) is a
newer version of the same data, already pre-converted to the YOLO layout
(`images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`,
7 classes) - so no conversion step is needed for it, unlike the dataset
above. `src/datasets/nvpdyf_bdd100k.py` just locates it, parses the label
files, and computes EDA stats (class distribution, class imbalance,
objects/image, box size and aspect ratio - overall and split by day/night,
day/night distribution, resolution check, missing/empty/orphan label
checks). [`dataset_baseline.ipynb`](dataset_baseline.ipynb) walks through
that plus drawing a couple of ground-truth scenes (including one explicit
day/night pair) - open it for the class-balance/data-quality picture before
training against this dataset version.

The original YOLO-format dataset didn't record per-image day/night, so a
sidecar `timeofday.csv` (`image,timeofday`, e.g.
`train/0000f77c-6257be58.jpg,night`) was added afterwards to fill that gap.
`find_timeofday_csv` searches for it the same recursive way
`find_dataset_root` searches for `data.yaml` - it doesn't have to live in
any particular spot under `input_dir`. Everything still works without it,
just without a day/night breakdown.

**Known duplication:** the `origin/nvpdyf-dataset-loader` branch (not yet
merged) rewrote `src/datasets/bdd100k.py` itself to read this same dataset -
since `train.py`/`inference.py`/`metrics` already import from there, that's
the module anything training/evaluation-related should use. This EDA module
stays separate for now rather than depending on an unmerged branch, and the
two disagree on more than style: that branch's loader requires
`timeofday.csv` and treats it as the source of truth for split membership
(an image missing a row is invisible to training), while this module treats
`images/` as the source of truth and falls back to `unknown` for missing
rows. Revisit once that branch merges - either retire this module and point
`dataset_baseline.ipynb` at the merged `bdd100k.py`, or keep both but make
that divergence a deliberate, documented choice instead of an accident.

## How To Use

To train (fine-tune the YOLO head and evaluate night vs. day), run:

```bash
python3 train.py HYDRA_CONFIG_ARGUMENTS
```

The default config is `src/configs/baseline.yaml`. It trains with all
ultralytics augmentations disabled (`src/configs/augment/none.yaml`) - this
is meant as a clean baseline to measure the night/day gap on, not a final
recipe. Useful overrides:

```bash
# quick smoke test on a subsample
python3 train.py trainer.epochs=1 trainer.batch=2 datasets.max_train_images=200

# freeze the backbone instead of fine-tuning the whole network
python3 train.py trainer.freeze=10

# force a specific device (e.g. Apple Silicon)
python3 train.py trainer.device=mps

# turn individual augmentations back on to try improving on the baseline
python3 train.py augment.fliplr=0.5 augment.mosaic=1.0
```

Checkpoints, logs and `results.json` are written under
`saved/runs/<trainer.run_name>/`. Re-running with the same `trainer.run_name`
raises an error unless you pass `trainer.override=true`.

`train.py` only: set `COMET_API_KEY` in the environment to log the
zero-shot and fine-tuned evaluation runs to Comet ML; without it, tracking
is skipped and everything still runs normally. `inference.py` never touches
Comet, regardless of this variable.

To evaluate a trained checkpoint (and save a prediction visualization):

```bash
python3 inference.py inferencer.weights=saved/runs/01_finetune_yolov8n_head/weights/best.pt
```

Results are written under `saved/eval/<inferencer.save_path>/`.

## Project layout

```
src/datasets/bdd100k.py        BDD100K -> YOLO dataset conversion, class balance
src/datasets/nvpdyf_bdd100k.py nvpdyf-bdd100k (pre-converted) EDA: stats, sampling
src/model/yolo_model.py        ultralytics YOLO wrapper + head diagnostics
src/metrics/detection.py       COCO-style mAP, computed separately for night/day
src/logger/comet_writer.py     Comet ML run logging
src/utils/visualize.py         prediction / ground-truth visualization
src/configs/                   Hydra configs (baseline.yaml / inference.yaml + subconfigs)
train.py / inference.py        entry points
dataset_baseline.ipynb         nvpdyf-bdd100k EDA notebook (dataset info + sample scenes)
```
