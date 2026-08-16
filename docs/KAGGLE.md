# Running on Kaggle

A copy-paste runbook for training on Kaggle's GPU. Everything below goes
into notebook cells, but the notebook is only a shell around
`python3 train.py` — no project logic lives in it.

## Before the first cell

In the notebook's right-hand panel:

1. **Accelerator** → `GPU T4 x2` (or `P100`). Without this, training runs on
   CPU and will not finish.
2. **Internet** → **On**. Off by default, and three separate things need it:
   `git clone`, ultralytics downloading `yolov8n.pt`, and Comet. A failure
   here looks like a DNS/connection error, not a permissions error.
3. **Add Input** → the `nvpd-bdd100k` dataset (images + `train/val/test.json`).
4. *(optional)* **Add-ons → Secrets** → add `COMET_API_KEY`. Without it the
   run still works, it just skips tracking.

## 1. Get the code

```python
!git clone -b yolo-baseline-scripts \
    https://github.com/Antonoof/Yandex-night-vision-Detection.git repo
%cd repo
```

Use `%cd`, not `!cd` — `!cd` runs in a subshell and the directory change is
gone by the next cell.

If the repository is private, put a GitHub personal access token in Kaggle
Secrets as `GITHUB_TOKEN` and clone with it:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -b yolo-baseline-scripts \
    https://{token}@github.com/Antonoof/Yandex-night-vision-Detection.git repo
%cd repo
```

## 2. Install dependencies

Kaggle already ships a CUDA-linked `torch`/`torchvision`. Do **not** run
`pip install -r requirements.txt` there — it can pull a CPU-only build over
the working one. Install only what is missing:

```python
!pip install -q ultralytics torchmetrics pycocotools hydra-core comet_ml
```

The package is `hydra-core`, not `hydra` — the latter is an unrelated,
abandoned package that fails to build.

## 3. Environment

```python
import os
from kaggle_secrets import UserSecretsClient

try:
    os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret("COMET_API_KEY")
    print("Comet enabled")
except Exception as e:
    print("no Comet secret, tracking will be skipped:", type(e).__name__)
```

`train.py` reads `COMET_API_KEY` from the environment, and `!python`
inherits the notebook process's environment, so setting it here is enough.

## 4. Smoke test first

Never start a multi-hour run before proving the pipeline end to end. One
epoch on 200 frames takes a couple of minutes and exercises every step:
dataset discovery, conversion, zero-shot evaluation, training, the head
swap, the final evaluation, and Comet.

```python
!python3 train.py \
    datasets.input_dir=/kaggle/input \
    trainer.device=0 \
    trainer.epochs=1 \
    trainer.batch=8 \
    datasets.max_train_images=200 \
    trainer.run_name=_smoke \
    trainer.override=true
```

Point `datasets.input_dir` at `/kaggle/input` itself, not at a specific
subfolder: the mount path varies between notebook versions
(`/kaggle/input/<slug>/` vs `/kaggle/input/datasets/<owner>/<slug>/`) and
the dataset search is recursive, so the input root works either way.

What the log should show:

* `train: ... frames | val: night=1860, day=7354`
* `nc (classes): 80` before training, then
  `Transferred 322/355 items from pretrained weights`, then
  `nc (classes): 8` after — the head swap, see
  [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
* two side-by-side night/day tables.

If `find_dataset_root` raises, find where the data actually landed:

```python
!find /kaggle/input -maxdepth 5 -iname val.json
```

## 5. The real run

```python
!python3 train.py \
    datasets.input_dir=/kaggle/input \
    trainer.device=0 \
    trainer.run_name=01_baseline_yolov8n_night
```

Roughly 2295 iterations per epoch on the full train split; 25 epochs is
about 2–4 hours on a T4. A Kaggle session is limited (and the weekly GPU
quota more so), so for the full run use **Save Version → Run All** — it
executes in the background and survives the browser closing, unlike an
interactive session.

Pick `trainer.run_name` per [EXPERIMENTS.md](EXPERIMENTS.md); it is also
the output directory, and a rerun under an existing name fails unless you
pass `trainer.override=true`.

## 6. Get the results out

Everything lands under `/kaggle/working/repo/saved/runs/<run_name>/`, which
Kaggle only keeps if it is in the notebook's output. Copy what matters to
`/kaggle/working` so it is saved with the version:

```python
!cp -r saved/runs/01_baseline_yolov8n_night /kaggle/working/
```

Weights are `weights/best.pt`; the metrics are in `results.json` and
`info.log`, and in Comet if it was enabled.

To evaluate that checkpoint later without retraining:

```python
!python3 inference.py \
    datasets.input_dir=/kaggle/input \
    inferencer.device=0 \
    inferencer.weights=saved/runs/01_baseline_yolov8n_night/weights/best.pt
```

## Gotchas

| Symptom | Cause |
| --- | --- |
| `FileNotFoundError` from `find_dataset_root` | dataset not added as Input, or added under an unexpected path — run the `find` above |
| connection/DNS errors | **Internet** is off in the notebook settings |
| `yolov8n.pt` fails to download | same — Internet off |
| training is absurdly slow | Accelerator is `None`, or `trainer.device` was not set to `0` |
| `ValueError: ... already has a finished run` | `trainer.run_name` was reused; rename it or pass `trainer.override=true` |
| directory changes don't stick between cells | `!cd` instead of `%cd` |
| `pip install hydra` fails to build | the package is `hydra-core` |
