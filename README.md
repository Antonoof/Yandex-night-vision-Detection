# Night Vision Project

Fine-tunes a YOLOv8 detector on BDD100K (the `nvpdyf-bdd100k` YOLO-format
build, see [Data](#data))
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

See **[docs/KAGGLE.md](docs/KAGGLE.md)** for the full runbook (notebook
settings, cloning, the smoke test, getting results out, and the usual
failure modes). The short version:

```python
!git clone -b nvpdyf-dataset-loader <this-repo-url> repo
%cd repo
!pip install -q ultralytics==8.4.120 torchmetrics pycocotools hydra-core comet_ml
!python3 train.py datasets.input_dir=/kaggle/input trainer.device=0
```

Kaggle ships a CUDA-linked `torch`/`torchvision` already - do not
`pip install -r requirements.txt` there, it can replace it with a CPU-only
build. Turn **Internet** on in the notebook settings: `git clone`, the
`yolov8n.pt` download and Comet all need it.

### Cloud GPU VM

Same as the local setup: install a CUDA-matched `torch`/`torchvision` for
the VM's driver version, then `pip install -r requirements.txt`. `device:
auto` in the configs picks up the GPU automatically.

## Data

Point `datasets.input_dir` (default `data/nvpdyf-bdd100k`) at a folder that
contains, somewhere inside it (nesting is fine - the code searches for it):

```
images/{train,val,test}/*.jpg
labels/{train,val,test}/*.txt
data.yaml
timeofday.csv
```

This is the `nvpdyf-bdd100k` dataset, already in YOLO format, so nothing is
converted. `data.yaml` fixes the class order, which the label files encode as
bare integers - `src/datasets/bdd100k.py` refuses to run if it disagrees with
`CLASSES`. `timeofday.csv` maps `<split>/<file name>` to `daytime`/`night`
and is what every night-vs-day split in this project is built on; it is a
separate file because the YOLO format has nowhere to put it.

`build_yolo_dataset` still rewrites the tree into `datasets.work_dir`
(default `data/bdd_yolo`): `max_train_images` may have dropped frames, and
the source tree also holds `test/`, which training must never see.

## How To Use

To train (fine-tune the YOLO head and evaluate night vs. day), run:

```bash
python3 train.py HYDRA_CONFIG_ARGUMENTS
```

The default config is `src/configs/adaptive.yaml`. It applies cached Zero-DCE
only to night frames, weights night loss x5, derives inverse-frequency class
weights from the selected train split, enables only a safe horizontal flip,
and trains at `imgsz=960`. See
**[docs/ADAPTIVE_TRAINING.md](docs/ADAPTIVE_TRAINING.md)** for the exact loss
definition and ablations. The original raw 640px baseline remains available
with `--config-name baseline`.

Useful overrides:

```bash
# quick smoke test on a subsample
python3 train.py trainer.epochs=1 trainer.batch=2 datasets.max_train_images=200

# original unweighted/raw-image baseline
python3 train.py --config-name baseline trainer.run_name=01_raw_baseline

# freeze the backbone instead of fine-tuning the whole network
python3 train.py trainer.freeze=10

# force a specific device (e.g. Apple Silicon)
python3 train.py trainer.device=mps

# derive the night weight from the actual frame ratio instead of manual x5
python3 train.py loss.adaptive.timeofday.mode=inverse_frequency

# stronger/weaker manual night emphasis
python3 train.py loss.adaptive.timeofday.weights.night=3.0

# class reweighting strength: 0 disables it, 1 is full inverse frequency
python3 train.py loss.adaptive.classes.power=0.5

# increase input detail; reduce batch if VRAM is insufficient
python3 train.py trainer.imgsz=1280 trainer.batch=4

# ablations
python3 train.py transforms=none
python3 train.py loss=standard
python3 train.py augment=none
```

Checkpoints, logs and `results.json` are written under
`saved/runs/<trainer.run_name>/`. Re-running with the same `trainer.run_name`
raises an error unless you pass `trainer.override=true`.

`train.py` only: set `COMET_API_KEY` in the environment to log to Comet ML;
without it, tracking is skipped and everything still runs normally.
`inference.py` never touches Comet, regardless of this variable.

One adaptive training run produces two Comet experiments:
`00_zero_shot_zero_dce` (the un-finetuned COCO model on the configured
preprocessed night/day inputs) and `<trainer.run_name>`, which
holds both the per-epoch curves — box/cls/dfl losses, learning rate,
gradient norm, per-class AP and recall — and the final night/day mAP,
including the small/medium/large breakdown. **Read
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) before starting a run**: it fixes
the run naming, the tags and the metric names, and runs that break those
conventions cannot be compared with the previous ones afterwards.

To evaluate an adaptive checkpoint (Zero-DCE is applied to night validation
frames from the same cache) and save a prediction visualization:

```bash
python3 inference.py inferencer.weights=saved/runs/01_finetune_yolov8n_head/weights/best.pt
```

Results are written under `saved/eval/<inferencer.save_path>/`.
Use `transforms=none inferencer.imgsz=640` when evaluating the raw baseline.

## Project layout

```
src/datasets/bdd100k.py     BDD100K -> YOLO dataset conversion, class balance
src/model/yolo_model.py     ultralytics YOLO wrapper + head diagnostics
src/training/               adaptive time-of-day + class-balanced loss
src/transforms/             cached conditional Zero-DCE preprocessing
src/metrics/detection.py    COCO-style mAP, computed separately for night/day
src/logger/comet_writer.py  Comet ML run logging (training curves + evaluation)
src/utils/visualize.py      ground-truth and prediction visualization
src/configs/                Hydra configs (adaptive/baseline/inference + groups)
train.py / inference.py     entry points
docs/READING_GUIDE.md       every file, in the order worth reading them
docs/PROJECT_STRUCTURE.md   what every file does, where each constant lives
docs/EXPERIMENTS.md         run naming, tags and the metric contract
docs/ADAPTIVE_TRAINING.md   adaptive loss, transforms, resolution and ablations
docs/KAGGLE.md              step-by-step runbook for Kaggle GPU
notebooks/                  thin viewers over src/ - no logic of their own
```

New to the codebase? Start with
**[docs/READING_GUIDE.md](docs/READING_GUIDE.md)** — it walks all 33 files in
dependency order, about two hours, with what to look for in each.
[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) is the reference to
keep open beside it: the config system, the data flow, and how to tell from
the log that the detection head was replaced.

## Notebooks

[`notebooks/01_dataset_overview.ipynb`](notebooks/01_dataset_overview.ipynb)
builds the YOLO dataset, reports the class balance, and draws a couple of
annotated scenes. It contains no logic of its own — every step is a call
into `src/`, so the notebook cannot drift away from what `train.py` does.
Training is not started from a notebook; it runs from the CLI as above.
