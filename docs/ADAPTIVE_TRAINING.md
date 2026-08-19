# Adaptive night/day training

The default `python3 train.py` composes `src/configs/adaptive.yaml` and makes
all requested behavior configurable. The raw reference run is still:

```bash
python3 train.py --config-name baseline trainer.run_name=01_raw_baseline
```

## 1. Conditional Zero-DCE

`transforms=night_zero_dce` reads `timeofday.csv` and applies Zero-DCE only
when `timeofday == night` and the split is listed in `zero_dce.splits`.
Enhanced frames are written once to `data/preprocessed_cache/` and reused.
Day frames remain unchanged. The resulting paths are used consistently by:

- training;
- Ultralytics' native validation;
- periodic night/day validation;
- final evaluation and visualizations;
- `inference.py`.

This consistency matters: training on enhanced nights but validating on raw
nights would measure a different input distribution.

```yaml
transforms:
  zero_dce:
    enabled: true
    apply_to: [night]
    splits: [train, val]
    device: same_as_trainer
    use_amp: true
    batch_size: 8
  cache:
    dir: data/preprocessed_cache
    overwrite: false
    jpeg_quality: 95
```

The cache namespace contains a hash of the Zero-DCE weights and output
settings, so changing them creates a new cache automatically.

## 2. Time-of-day loss

The default manual weights are:

```yaml
loss:
  adaptive:
    timeofday:
      mode: manual
      weights:
        daytime: 1.0
        night: 5.0
```

The native YOLO box, classification and DFL losses are computed separately
for the daytime and night portions of a batch. The night result is multiplied
by five. A fixed dataset-mean normalization preserves approximately the
original global loss scale without cancelling x5 in an all-night batch.

For automatic weighting from the selected train split:

```bash
python3 train.py loss.adaptive.timeofday.mode=inverse_frequency
```

With an 80/20 split this produces approximately `daytime=1`, `night=4`.

## 3. Class-balanced classification loss

Counts are calculated from the actual training subset after
`datasets.max_train_images` is applied. For class `c`:

```text
weight_c ∝ 1 / (count_c + smoothing) ^ power
```

Weights are normalized and clipped with `min_weight`/`max_weight` from
`src/configs/loss/adaptive.yaml`. Ultralytics 8.4.120 applies them directly
to the per-anchor BCE classification component. Box and DFL keep their native
class-independent scale.

Actual counts and resolved weights are printed at training start, stored in
`results.json`, and logged to Comet as `loss.class_count.*` and
`loss.class_weight.*`.

## 4. Road-safe augmentation

`augment=road_safe` enables only:

```yaml
fliplr: 0.5
```

Ultralytics mirrors the boxes together with the image. Vertical flip,
rotation, perspective, crop/translate, scale, mosaic, mixup, cutmix and
erasing are all disabled. Use `augment=none` for a no-augmentation ablation.

## 5. Input resolution

The source frames are `1280x720`. Ultralytics letterboxes them to the square
`trainer.imgsz`; the previous `640` setting therefore downscaled them.
The adaptive recipe uses `960` by default:

```bash
python3 train.py trainer.imgsz=960 trainer.batch=8
python3 train.py trainer.imgsz=1280 trainer.batch=4
```

`imgsz` should be divisible by 32. Higher resolution helps small traffic
lights and distant objects but increases VRAM and training time roughly with
the number of input pixels.

The metric config also sets `metrics.nms_max_time_img=0.5`. This only raises
the NMS time budget for the low-confidence (`conf=0.001`) mAP pass; it does
not alter boxes or scores, and prevents timeout-truncated validation batches.

## 6. Multi-GPU (2 devices)

```bash
python3 train.py trainer.device=\'0,1\'
```

The value must be quoted (`\'0,1\'`, escaped so the literal quotes reach
Hydra): Hydra's override grammar treats a bare comma as list/sweep syntax and
rejects `trainer.device=0,1` with "Ambiguous value for argument". Quoting
makes it a plain string, which is what `split_devices` and ultralytics both
expect - `trainer.device=[0,1]` (Hydra list syntax) would train fine but
`split_devices` would not recognize it as two devices, so the parallel
eval/preprocessing paths below would silently fall back to single-device.

`trainer.device="0,1"` is the ultralytics convention for DDP training across
two GPUs, but `model.train()` **never** receives it as-is: it always gets a
single device (`train_device = eval_devices[0]`, resolved in `train.py`).

This is deliberate, not an oversight. Ultralytics' multi-GPU training does
not run in this process - it serializes the trainer's hyperparameters into a
temp script and re-launches it as a subprocess via `torch.distributed.run`
(`ultralytics.utils.dist.generate_ddp_file`). That subprocess rebuilds the
trainer from scratch from those hyperparameters alone; it has no idea about
`model.add_callback(...)` calls made on the original trainer object in this
process. `CometRunLogger` and `PeriodicNightDayEval` both attach through
exactly that mechanism (`comet_run.attach(model)`, `periodic.attach(model)`
in `train.py`), so under real DDP training their callbacks silently never
fire: no per-epoch losses/lr/grad_norm/per-class AP, no periodic night/day
curve - only the final post-training `log_eval` call (made in this process,
after `model.train()` returns) reaches Comet. A run trained this way shows
one lonely point in Comet instead of a curve.

So the training step itself is always single-GPU. Everything else in the
pipeline (Zero-DCE preprocessing, zero-shot eval, periodic in-training eval,
the final night/day pass) is still a pair of independent single-GPU jobs,
and *that* is what the second GPU is for. `train.py` splits `trainer.device`
into individual devices (`src.utils.init_utils.split_devices`) and, whenever
there are two, runs the independent halves concurrently, one per GPU
(`src.utils.parallel.run_paired`):

* Zero-DCE: the train split on one GPU, the val split on the other.
* Zero-shot / periodic / final evaluation: the night subset on one GPU, the
  day subset on the other.

With a single device (the default), behavior and timing are unchanged from
before this existed - `split_devices` returns a one-element list and every
parallel path collapses back to the original sequential calls.

`trainer.batch` is a **per-GPU** batch size again, same as single-device
runs: since `model.train()` no longer sees two devices, ultralytics has
nothing to split it across.

## Recommended ablation sequence

Change one axis per run:

```bash
# raw reference
python3 train.py --config-name baseline trainer.run_name=01_raw

# only Zero-DCE
python3 train.py loss=standard augment=none trainer.run_name=02_zero_dce

# Zero-DCE + time loss
python3 train.py loss.adaptive.classes.enabled=false augment=none \
  trainer.run_name=03_time_weight

# full requested recipe
python3 train.py trainer.run_name=04_full_adaptive
```
