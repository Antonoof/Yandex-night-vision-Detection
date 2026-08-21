# Reading guide

Every file in the project, in the order that makes them easiest to
understand: 33 files in 31 steps (the five `__init__.py` are one step),
about 1440 lines of code and config. Roughly two focused hours.

The order is not alphabetical and not by importance. It follows the
**dependency direction**: leaves first, then the things built on them, then
the entry point that ties it all together. Reading `train.py` first is the
usual mistake — it is a list of calls into code you have not met yet.

Line numbers are current as of this document; treat them as hints, not
addresses.

---

## Stage 0 — the map (5 min)

| # | File | Lines | What it is |
|---|---|---|---|
| 1 | `README.md` | 154 | What the project does, how to install and run it |
| 2 | `docs/PROJECT_STRUCTURE.md` | 216 | The layout, the config system, what `train.py` does step by step |

Read these two for orientation only. Do not try to retain details — the
point is to know what exists.

---

## Stage 1 — the vocabulary (15 min)

Two small files that define the terms every later file uses. If you skip
them, everything downstream reads as magic.

| # | File | Lines | What to look for |
|---|---|---|---|
| 3 | `src/datasets/bdd100k.py` — **top only, lines 1–33** | 33 | `CLASSES` (17), `CLASS_TO_ID` (27), `COCO80_TO_OURS` (31), `IMG_W/IMG_H` (33) |
| 4 | `src/utils/io_utils.py` | 32 | `ROOT_PATH` (5) — derived from the file's own location, so paths resolve against the repo root no matter where you run from |

**Why the class order matters.** `CLASSES` is written in COCO's order, so
`COCO80_TO_OURS` is a plain lookup rather than a name-matching step. That is
what makes the zero-shot measurement possible at all: the pretrained model
speaks COCO indices, and we translate them with a dict.

`CLASSES` is also the *only* place the number 8 exists. It is never written
into a config — see stage 5.

---

## Stage 2 — the data path (30 min)

The heart of the project. Read top to bottom; the functions are in call
order.

| # | File | Lines | What to look for |
|---|---|---|---|
| 5 | `src/datasets/bdd100k.py` — **the rest** | 284 | see below |

| Function | Line | Responsibility |
|---|---|---|
| `find_dataset_root` | 47 | Recursively finds the folder with `images/` + `data.yaml` + `timeofday.csv`. Recursive on purpose: Kaggle's mount path is not stable |
| `_check_class_names` | 68 | Refuses to run if the dataset's `data.yaml` orders classes differently from `CLASSES`. Labels are bare integers, so a mismatch relabels every box **silently** — this is the project's one known silent-corruption path |
| `_load_timeofday` | 89 | `timeofday.csv` → `{"<split>/<name>": "night" \| "daytime"}`. The YOLO format has nowhere to store this, hence a side file |
| `load_records` | 95 | `labels/<split>/*.txt` → `{name, path, timeofday, boxes}`. `boxes` are `(class_id, cx, cy, w, h)`, normalized. Iterates the **CSV**, not the label dir: an object-free frame has no label file and would be dropped |
| `_check_frame_size` | 144 | Spot-checks the first frame against 1280×720. Without it a different-sized dataset would silently corrupt every mAP number instead of raising |
| `describe_split_balance` | 158 | Logs the class/night/day balance **and returns it**, so notebooks can plot it |
| `subsample` | 204 | Shrinks the train set keeping the night/day ratio — for quick runs |
| `_link_or_copy` | 228 | Symlink, falling back to copy where symlinks are unavailable |
| `build_yolo_dataset` | 237 | Rewrites `images/`, `labels/*.txt`, `data.yaml` for training. The source is already YOLO-format, but `subsample` may have dropped frames and the source also holds `test/`, which training must never see. **Line 278 writes `names:` from `CLASSES` — remember this line, it is where the head replacement begins** |

**The one idea to take away:** everything downstream — metrics,
visualization, training — consumes the record dict from `load_records`. It
is the project's internal data format, and it is deliberately plain (no
classes, no torch), so it is trivial to inspect and to test.

---

## Stage 3 — the leaves (20 min)

Small, independent modules. Each does one thing and depends only on stage 1.

| # | File | Lines | Responsibility |
|---|---|---|---|
| 6 | `src/utils/init_utils.py` | 36 | `set_random_seed` (8) — torch, numpy, `random`, `PYTHONHASHSEED`, deterministic cuDNN. `resolve_device` (23) — `"auto"` → GPU 0 or `"cpu"` |
| 7 | `src/model/yolo_model.py` | 52 | `build_model` (10) — a one-line `YOLO(weights)` wrapper. `log_head_info` (22) — dumps `nc`, `reg_max`, the `cv2`/`cv3` branches and parameter counts. **This is the diagnostic that proves the head was replaced** |
| 8 | `src/utils/visualize.py` | 102 | `draw_ground_truth` (14) — annotated frames, the sanity check before training. `draw_predictions` (65) — what the model actually outputs |
| 9 | `src/logger/logger.py` + `logger_config.json` | 33 + 36 | Stdlib logging: console + `info.log` in the run directory. Unchanged from the original template |

Read `log_head_info` slowly — it is short, and it is the only place in the
project that looks inside the network.

---

## Stage 4 — measurement (30 min)

| # | File | Lines | What to look for |
|---|---|---|---|
| 10 | `src/metrics/detection.py` | 145 | see below |

| Piece | Line | Responsibility |
|---|---|---|
| `RESULT_KEYS` | 13 | The metric contract: `map`, `map_50`, `map_75`, `map_small/medium/large`, `mar_100`. Renaming these breaks comparison with every past run |
| `targets_to_tensors` | 24 | Record → torchmetrics format: normalized `cxcywh` → pixel `xyxy`. The only coordinate conversion in the project |
| `evaluate_detector` | 54 | Runs the model over a split, accumulates into `MeanAveragePrecision`, computes **once at the end** |
| `print_results` | 129 | The night/day side-by-side table |

Three decisions worth understanding here:

* **mAP is accumulated, not averaged over batches.** It is defined through a
  PR curve over *all* predictions sorted by confidence; a mean of per-batch
  mAPs is a different, batch-size-dependent quantity.
* **`conf=0.001`, not the usual `0.25`.** A high threshold cuts the tail off
  the PR curve and undercounts mAP. The 0.25 you see elsewhere is for
  showing results to a human.
* **`class_map`** (line 63 onward) is what makes the zero-shot run possible:
  it keeps only predictions in `COCO80_TO_OURS` and renumbers them.

---

## Stage 5 — configuration (20 min)

Now the configs make sense, because you have met everything they configure.
Read the root config first, then follow its `defaults:` list.

| # | File | Lines | Responsibility |
|---|---|---|---|
| 11 | `src/configs/baseline.yaml` | 21 | Root config for `train.py`. The `defaults:` block composes the groups below; `trainer:` holds epochs, imgsz, batch, freeze, seed, device, run_name, override, eval_zero_shot |
| 12 | `src/configs/model/baseline.yaml` | 2 | `_target_: src.model.build_model`, `weights: yolov8n.pt`. Hydra calls the target with the remaining keys |
| 13 | `src/configs/datasets/bdd100k.yaml` | 6 | `input_dir` (where to search), `work_dir` (where to build), `max_train_images` |
| 14 | `src/configs/metrics/detection.yaml` | 5 | `conf`, `iou`, `max_det` for evaluation |
| 15 | `src/configs/writer/comet.yaml` | 3 | Comet project name, `dataset_version` |
| 16 | `src/configs/augment/none.yaml` | 21 | Every ultralytics augmentation, zeroed. The baseline is deliberately un-augmented |
| 17 | `src/configs/inference.yaml` | 16 | Root config for `inference.py`: which checkpoint, where to save, visualization settings. Note it pulls **no** `writer` and no `augment` group |

**The line worth staring at:** `augment/none.yaml` exists as a *group*, so
a later experiment adds `augment/night.yaml` next to it and switches with
`python3 train.py augment=night` — no code change, and the config that
produced any run is recorded in Comet.

**Notice what is *not* in any config:** the class list, the frame size and
the metric names. Those are code (stages 1 and 4). The split is: configs
hold what varies between runs, code holds what defines what the data means.

---

## Stage 6 — tracking (30 min)

The largest single file, and the only one with real machinery in it.

| # | File | Lines | What to look for |
|---|---|---|---|
| 18 | `src/logger/comet_writer.py` | 334 | see below |

| Piece | Line | Responsibility |
|---|---|---|
| `_KEY_RENAMES`, `_normalize_key` | 24, 33 | ultralytics' metric names → this project's names |
| `_clean` | 48 | Drops non-numeric values before logging |
| `_ultralytics_comet_disabled` | 66 | Turns off ultralytics' own Comet integration for the run. It would open a *second* experiment and close it when training ends, leaving the final night/day metrics homeless |
| `_grad_norm` | 89 | Gradient L2 norm, divided by the AMP scale factor |
| `CometRunLogger` | 119 | One experiment for one fine-tuning run |
| `.attach` | 191 | Registers the ultralytics callbacks |
| `._on_train_start` | 202 | Wraps `optimizer_step` to measure gradients **before ultralytics zeroes them** |
| `._on_fit_epoch_end` | 220 | Per-epoch: losses, lr, grad_norm, per-class AP/R |
| `._per_class_metrics` | 246 | Per-class numbers, which live on the validator rather than in `trainer.metrics` |
| `.log_eval` | 273 | The final night/day metrics, into the same experiment |
| `log_evaluation_run` | 301 | A standalone experiment for the zero-shot run, which has no training curves |

Every callback body is wrapped in `try/except` deliberately: a bug in
logging must never kill a training run that has been going for hours.

Read `docs/EXPERIMENTS.md` (152 lines) right after this file — it is the
policy that this code implements.

---

## Stage 7 — the entry points (30 min)

Now `train.py` reads as a story, because every call in it is to something
you have already read.

| # | File | Lines |
|---|---|---|
| 19 | `train.py` | 220 |
| 20 | `inference.py` | 96 |

`train.py` in order:

| Line | What happens |
|---|---|
| 5–7 | `USE_COMET`, and `import comet_ml` **before** torch/ultralytics — a requirement of the library, hence the odd placement above the other imports |
| 37 | `set_random_seed` |
| 44 | Refuses to start if the run directory already has weights, unless `trainer.override` |
| 49 | `setup_logging` — everything after this goes to console *and* `info.log` |
| 52 | `resolve_device` |
| 57–60 | Find the dataset, load train/val, log the balance |
| 62–69 | Subsample, then assert no frame is in both splits |
| 71 | `build_yolo_dataset` → `data.yaml` |
| 75–76 | Split val into night and day — the whole point of the project |
| 84 | `eval_kwargs`, shared by every evaluation so they stay comparable |
| 94–127 | **Measurement A**: zero-shot. `log_head_info` (96) prints `nc=80` |
| 129 | Open the Comet experiment for the fine-tuning run |
| 151 | `comet_run.attach(model)` — callbacks on |
| 153–154 | `model.train(data=str(data_yaml), ...)` — **the head replacement is triggered here** |
| 174–178 | Load `best.pt`, `log_head_info` prints `nc=8` |
| 180–193 | **Measurement B**: fine-tuned, night and day, into the same experiment |
| 214 | `results.json` |

`inference.py` is a strict subset: steps 37–76 and 174–190 for a checkpoint
you already have, plus `predictions.png`. It never touches Comet.

---

## Stage 8 — the periphery (10 min)

| # | File | Lines | Responsibility |
|---|---|---|---|
| 21 | `requirements.txt` | 20 | Dependencies. `torch` is deliberately unpinned — read the comment at the top |
| 22 | `.python-version` | 1 | 3.11 |
| 23 | `.pre-commit-config.yaml` | 9 | black, isort, whitespace/YAML hooks |
| 24 | `.gitignore` | 17 | `data/`, `saved/`, `*.pt` are generated, never committed |
| 25 | `docs/KAGGLE.md` | 160 | The GPU runbook |
| 26 | `notebooks/01_dataset_overview.ipynb` | 330 | A thin viewer over `src/` — no logic of its own |
| 27–31 | `src/*/__init__.py` | 0–8 | Re-exports. `src/logger/__init__.py` and friends define the public surface of each package |

---

## Where the head replacement happens

The single most-asked question, as a path through the files you have now read:

| Step | Where |
|---|---|
| 8 class names exist | `src/datasets/bdd100k.py:17` |
| written into `data.yaml` as `names:` | `src/datasets/bdd100k.py:223` |
| `data.yaml` built | `train.py:71` |
| handed to ultralytics | `train.py:154` |
| **ultralytics rebuilds the head** | `DetectionModel.__init__` — logs `Overriding model.yaml nc=80 with nc=8` |
| **matching weights grafted back** | `BaseModel.load` — logs `Transferred 322/355 items from pretrained weights` |
| confirmed in our log | `train.py:96` (before, `nc=80`) and `train.py:178` (after, `nc=8`) |

Note there is no `nc` in any config. ultralytics derives it as
`len(names)`, so `CLASSES` is the single source of truth.

---

## Suggested exercises

Reading is not understanding. Each of these takes minutes and fails loudly
if your mental model is wrong:

1. Run `python3 train.py trainer.epochs=1 datasets.max_train_images=200
   trainer.run_name=_smoke trainer.override=true` and find all four head-log
   lines in `saved/runs/_smoke/info.log`.
2. Open `data/bdd_yolo/data.yaml` and one `labels/train/*.txt`. Check by hand
   that a line's `cx cy w h` matches the box in the corresponding
   `<split>.json`.
3. Add a class to `CLASSES`, rerun, and watch `Overriding model.yaml` change
   to `nc=9`. Then revert — and note that the previous runs' metrics are now
   incomparable, which is exactly why that list is not a config value.
4. Set `trainer.eval_zero_shot=false` and see which Comet experiment stops
   being created.
