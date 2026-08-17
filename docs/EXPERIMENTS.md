# Experiment conventions

The goal of the project is to measure how much a pretrained detector
degrades at night and to raise its mAP back. This only works if runs stay
comparable, so the rules below are about comparability, not bureaucracy.

## CometML structure

| Tier | Value |
| --- | --- |
| Workspace | the team workspace |
| Project | `night-vision-detection` — **one for everything** (`src/configs/writer/comet.yaml`) |
| Experiment | one evaluation or fine-tuning run |

**All approaches live in one project.** CometML compares experiments inside
a project; it cannot put runs from two projects on one chart. Splitting
"preprocessing" and "domain adaptation" into separate projects would destroy
the baseline-vs-approach comparison, which is the deliverable of the task.

Approaches are separated by **tags**, not by projects.

One `python3 train.py` produces **two** experiments:

| Experiment | What it holds |
| --- | --- |
| `00_baseline_zeroshot` | the COCO-pretrained model measured on night/day, no training behind it |
| `<trainer.run_name>` | the fine-tuning run: per-epoch curves **and** the final night/day metrics |

The fine-tuning run keeps its training curves and its final metrics in the
same experiment on purpose. ultralytics ships its own Comet integration that
would open a second experiment and close it the moment training ends —
leaving the night/day numbers, which are only computed afterwards, with
nowhere to go. `CometRunLogger` switches that integration off for the
duration of the run and restores the setting on the way out.

## Run naming

Set `trainer.run_name`. Format:

```
<NN>_<approach>_<model>_<data>
```

```
00_baseline_frcnn-r50_day
01_baseline_yolov8n_night
10_clahe_yolov8n_night
20_synthnight_yolov8n_mixed
```

* `NN` — a two-digit stage prefix, so sorting by name matches the order of
  the work (`0x` baseline, `1x` preprocessing, `2x` synthetic data, ...).
* Lowercase, `-` inside a part, `_` between parts, no spaces.

**`run_name` is also the output directory** (`saved/runs/<run_name>`). It
must be unique: a rerun that would clobber an existing run fails with a
clear error. Either bump the name or pass `trainer.override=true` if the
previous run is genuinely disposable.

## Tags

Use `key:value` so tags stay filterable:

| Tag | Examples |
| --- | --- |
| `stage:` | `stage:baseline`, `stage:preproc`, `stage:synth`, `stage:da`, `stage:aug` |
| `method:` | `method:zero-shot`, `method:finetune-head`, `method:clahe` |
| `model:` | `model:yolov8n`, `model:yolov8s` |
| `data:` | `data:night-real`, `data:day-real`, `data:mixed` |

## Metrics

The names below must be **identical in every run**. A renamed metric is a
metric CometML cannot chart against the previous runs, and it cannot be
fixed after the fact.

### Final evaluation — `src/metrics/detection.py`

Logged as `night/<name>` and `day/<name>`, from the same checkpoint.

| Name | Meaning |
| --- | --- |
| `map` | **primary metric**, mAP@[.50:.95] |
| `map_50`, `map_75` | mAP at fixed IoU |
| `map_small`, `map_medium`, `map_large` | by object size — small/distant objects degrade first at night, this is the main diagnostic split |
| `map_<class>` | per-class mAP — pedestrians and cars behave differently at night |
| `mar_100` | recall — a missed pedestrian costs more than a false positive |

### Training curves — `src/logger/comet_writer.py`

Logged once per epoch, from ultralytics' own trainer.

| Name | Meaning |
| --- | --- |
| `train/box_loss`, `val/box_loss` | regression loss — *where* the object is (CIoU + DFL) |
| `train/cls_loss`, `val/cls_loss` | classification loss — *what* the object is (BCE) |
| `train/dfl_loss`, `val/dfl_loss` | the distributional part of the box loss |
| `train/grad_norm` | gradient L2 norm before clipping — spikes mean instability |
| `train/lr_pg0`, `..._pg1`, `..._pg2` | learning rate per parameter group |
| `val/precision`, `val/recall` | aggregate P/R at the best F1 threshold |
| `val/mAP50`, `val/mAP50-95` | ultralytics' own mAP, over the whole val split |
| `val/AP50-95_<class>`, `val/AP50_<class>` | per-class AP |
| `val/P_<class>`, `val/R_<class>` | per-class precision/recall |

### Error analysis — logged with every fine-tuning run

* `predictions_night`, `predictions_day` — ground truth beside predictions on
  a **fixed** set of frames (first N by name, `trainer.num_visualization_samples`).
  Fixed on purpose: a random sample cannot be compared between runs.
* the confusion matrix of the final `best.pt` validation pass.

The night/day and size-split metrics are **single points, not curves**: each
needs a full extra inference pass over the split (~12 min for this dataset),
which is not worth repeating every epoch. They are stamped with the last
training step so they sit at the end of the shared x-axis.

> `val/mAP50-95` (ultralytics) and `night/map` + `day/map` (torchmetrics) are
> **different implementations** — different PR-curve interpolation, different
> NMS settings, different averaging. Compare within one implementation, never
> across.

A value of `-1` means the metric is undefined on this partition (COCO
convention) — usually a class that never appears in the data.

**Always measure the night set and the day set with the same checkpoint.**
Augmentations and domain adaptation easily buy night mAP at the cost of day
mAP, and that is not a solution to the task. Without a day number in every
run the trade-off is invisible.

The success criterion of the project is `night/map` on the **real** night
drives — synthetic data is a training tool, never the thing being reported.

### Why mAP is not averaged over batches

mAP is defined through a precision-recall curve built over all predictions
of the dataset sorted by confidence. The mean of per-batch mAP values is a
different quantity, and it depends on the batch size, so it can be compared
neither with the COCO reference nor between runs. `evaluate_detector`
therefore accumulates predictions over a whole split and computes once.

The same reasoning sets `metrics.conf = 0.001` rather than the usual `0.25`:
a high threshold truncates the tail of the PR curve and undercounts mAP.

## Reproducibility

* The full Hydra config is what defines a run — **never** keep settings
  outside `src/configs/`. Everything is overridable from the CLI:
  `python3 train.py trainer.epochs=1 datasets.max_train_images=200`.
* `writer.dataset_version` must be bumped on every change of the dataset.
  Two runs trained on different data are not comparable, and without this
  field it is impossible to tell afterwards whether a run differed by model
  or by data.
* The split comes from the dataset itself and must not be re-cut.
  `test` stays untouched until the end of the project — it is the only
  remaining independent check.

## Checklist for a new run

1. `trainer.run_name` follows the format and is unique.
2. Tags describe the stage and the method.
3. `writer.dataset_version` matches the data actually used.
4. Metric names are unchanged.
5. Both the night and the day subsets are evaluated.
6. One thing changed relative to the run being compared against.
