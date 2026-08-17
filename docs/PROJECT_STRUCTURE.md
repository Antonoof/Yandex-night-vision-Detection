# Project structure

How the pieces fit together, what lives where, and what happens when you
run `python3 train.py`.

## The shape of the project

There is no notebook in the training path. Everything is a plain Python
module; the two entry points are `train.py` and `inference.py`, and every
number you might want to change lives in a [Hydra](https://hydra.cc) config
under `src/configs/`.

The rule the layout follows: **configs hold what varies between runs; code
holds what defines what the data means.** `epochs` is a config value — you
change it per experiment. The class list is not: changing it changes what
the labels mean, and every previously logged metric would silently stop
being comparable.

```
train.py                     entry point: build dataset -> fine-tune -> evaluate
inference.py                 entry point: evaluate an existing checkpoint
requirements.txt             dependencies (torch deliberately unpinned, see README)
.python-version              3.11
.pre-commit-config.yaml      black, isort, whitespace/YAML hooks

src/configs/                 every tunable value (see "Configs" below)
  baseline.yaml                train.py's root config
  inference.yaml               inference.py's root config
  model/baseline.yaml          which weights to start from
  datasets/bdd100k.yaml        where the data is, where to build it
  metrics/detection.yaml       inference thresholds for the mAP sweep
  writer/comet.yaml            Comet project name + dataset version
  augment/none.yaml            ultralytics augmentation hyperparameters

src/datasets/bdd100k.py      BDD100K JSON -> YOLO dataset; CLASSES; COCO80_TO_OURS
src/model/yolo_model.py      build_model (YOLO wrapper) + log_head_info
src/metrics/detection.py     COCO mAP via torchmetrics, night/day separately
src/logger/comet_writer.py   Comet: training curves + evaluation runs
src/logger/logger.py         stdlib logging setup (console + info.log)
src/logger/logger_config.json  logging handlers/formatters
src/utils/init_utils.py      set_random_seed, resolve_device
src/utils/io_utils.py        ROOT_PATH, read_json/write_json
src/utils/visualize.py       draw_ground_truth, draw_predictions

docs/EXPERIMENTS.md          run naming, tags, the metric contract
docs/PROJECT_STRUCTURE.md    this file
notebooks/                   thin viewers over src/ - no logic of their own
```

Two paths are created at runtime and are not in git:

* `data/bdd_yolo/` — the generated YOLO-format dataset (`datasets.work_dir`).
* `saved/runs/<run_name>/` — weights, `info.log`, `results.json`.

## Configs

`src/configs/baseline.yaml` is the root config for training. Its
`defaults:` block pulls in one file from each group:

```yaml
defaults:
  - model: baseline      # -> src/configs/model/baseline.yaml
  - datasets: bdd100k    # -> src/configs/datasets/bdd100k.yaml
  - metrics: detection
  - writer: comet
  - augment: none
  - _self_               # this file's own keys win over the groups
```

Hydra merges them into one object, and `train.py` reads it as
`config.trainer.epochs`, `config.model.weights`, and so on. Anything is
overridable from the command line without editing files:

```bash
python3 train.py trainer.epochs=1 datasets.max_train_images=200
python3 train.py augment=none augment.fliplr=0.5     # swap a group, then a key
```

### Where each constant lives

| What | Where |
| --- | --- |
| `epochs`, `imgsz`, `batch`, `seed`, `device`, `workers`, `patience` | `baseline.yaml` → `trainer:` |
| `freeze` (0 = train everything, 10 = freeze the backbone) | `baseline.yaml` → `trainer:` |
| `run_name`, `save_dir`, `override`, `eval_zero_shot` | `baseline.yaml` → `trainer:` |
| starting weights (`yolov8n.pt`) | `model/baseline.yaml` |
| dataset location, build location, `max_train_images` | `datasets/bdd100k.yaml` |
| evaluation `conf` / `iou` / `max_det` | `metrics/detection.yaml` |
| Comet project name, `dataset_version` | `writer/comet.yaml` |
| augmentation hyperparameters | `augment/none.yaml` |
| checkpoint to evaluate, visualization settings | `inference.yaml` → `inferencer:` |
| **class list, COCO mapping, frame size** | `src/datasets/bdd100k.py` (code, on purpose) |
| **metric names** | `src/metrics/detection.py` (`RESULT_KEYS`) and `src/logger/comet_writer.py` |
| logging handlers/format | `src/logger/logger_config.json` |

`ROOT_PATH` (`src/utils/io_utils.py`) is derived from the file's own
location, so relative config paths resolve against the repo root no matter
where you invoke the script from. An absolute value (e.g.
`datasets.input_dir=/kaggle/input`) overrides it — `Path("/repo") /
"/kaggle/input"` is `/kaggle/input`.

## Where the data comes from

Nothing is downloaded automatically. You provide a folder that contains,
at any nesting depth:

```
images/{train,val,test}/*.jpg
train.json   val.json   test.json
```

`datasets.input_dir` points at a root to search under, not at the dataset
itself: `find_dataset_root` walks it looking for a folder that has both an
`images/` subfolder and a `val.json` next to it. That is deliberate — on
Kaggle the mount path varies between `/kaggle/input/<slug>/` and
`/kaggle/input/datasets/<owner>/<slug>/`, and a recursive search survives
both.

`load_records(data_root, split)` reads `<split>.json` and normalizes each
frame into `{"name", "path", "timeofday", "boxes"}`, where `boxes` is a list
of `(class_id, cx, cy, w, h)` — coordinates normalized to the frame size,
which is the format YOLO wants. It also spot-checks the first frame against
`IMG_W × IMG_H = 1280 × 720`: predictions are rescaled to each image's real
size by ultralytics, so a frame-size mismatch would silently corrupt every
mAP number rather than raise.

`build_yolo_dataset` then writes the tree ultralytics actually reads:

```
data/bdd_yolo/
  images/train/*.jpg     symlinks to the source frames (copy as a fallback)
  labels/train/*.txt     one line per object: <class> <cx> <cy> <w> <h>
  images/val/  labels/val/
  data.yaml              path, train, val, names
```

Frames with no objects still get an (empty) `.txt` — those are background
images, and they teach the model not to invent objects.

**`test.json` is never read.** It stays untouched until the end of the
project; it is the only independent check that survives repeated
experimentation on `val`.

## Which model, and what "replacing the head" means here

`model/baseline.yaml` says `weights: yolov8n.pt` — the COCO-pretrained
YOLOv8-nano (~3.2M parameters), downloaded by ultralytics on first use.
Point it at any other checkpoint (`yolov8s.pt`, or a local `best.pt`) to
change the starting point.

**There is no "replace the head" flag.** It is a consequence of the data:

1. `CLASSES` in `src/datasets/bdd100k.py` has 8 entries.
2. `build_yolo_dataset` writes those 8 into `data.yaml` as `names:`.
3. `model.train(data=data.yaml)` makes ultralytics build a `Detect` head
   sized for 8 classes.
4. Weights are transferred from `yolov8n.pt` only where the shapes match.
   The classification branch `cv3` does not match — its internal width is
   `max(ch[0], min(nc, 100))`, which is 80 for COCO and 64 for us — so the
   whole branch stays randomly initialized. The backbone, the neck and the
   box-regression branch `cv2` (which does not depend on the class count)
   are inherited.

Three places in the log prove it happened:

| Evidence | Where |
| --- | --- |
| `nc (classes): 80`, `cv3 ... out_channels=80` | `log_head_info` on the pretrained model, before training |
| `Transferred .../... items from pretrained weights` | ultralytics, at the start of training — the 33 missing tensors are `cv3` |
| `nc (classes): 7`, `cv3 ... out_channels=7` | `log_head_info` on `best.pt`, after training |

The parameter count drops with it: 3,157,200 for the 80-class model versus
3,012,408 for ours.

`trainer.freeze` is the separate knob: `0` fine-tunes everything, `10`
freezes the first 10 modules (the backbone) so only the neck and head move.

## What `python3 train.py` does, in order

1. `set_random_seed(config.trainer.seed)` — torch, numpy, python `random`,
   `PYTHONHASHSEED`, plus deterministic cuDNN.
2. Refuses to start if `saved/runs/<run_name>/weights/` already exists,
   unless `trainer.override=true`. A silently clobbered run is worse than
   an error.
3. `setup_logging` — everything from here on goes to the console *and* to
   `saved/runs/<run_name>/info.log`.
4. `resolve_device` — `"auto"` becomes GPU `0` if CUDA is available, else
   `"cpu"`; anything else passes through (e.g. `"mps"`).
5. Finds the dataset, loads `train`/`val` records, logs the class balance,
   optionally subsamples, and asserts that no frame is in both splits.
6. Builds the YOLO dataset tree and `data.yaml`.
7. If `trainer.eval_zero_shot` (default true): loads the COCO-pretrained
   model, dumps the head layout, and evaluates it on the night and day
   subsets with `class_map=COCO80_TO_OURS` (COCO's 80 indices → our 8).
   Logged to Comet as `00_baseline_zeroshot`. This is the "how bad is it
   before we do anything" number.
8. Opens one Comet experiment for the fine-tuning run, attaches the
   training callbacks (losses, lr, gradient norm, per-class AP/R), and
   calls `model.train(...)` with the augmentation config unpacked into it.
9. Loads `best.pt`, dumps its head layout, evaluates night and day, prints
   the side-by-side table, and logs the results into the same experiment.
10. Writes `results.json` next to the weights.

`inference.py` is steps 3–5 and 9 for a checkpoint you already have, plus
an optional `predictions.png`. It never touches Comet.

## Outputs

```
saved/runs/<run_name>/
  weights/best.pt  weights/last.pt
  info.log                  everything the run logged
  results.json              config + all four metric sets
  results.csv, *.png        ultralytics' own curves and plots
saved/eval/<save_path>/     inference.py: results.json, predictions.png
```
