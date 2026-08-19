# Running on Yandex DataSphere

A runbook for DataSphere Jobs, for when the Kaggle GPU quota is gone. Same
rule as [KAGGLE.md](KAGGLE.md): the platform is a shell around
`python3 train.py`.

Two differences from Kaggle are worth knowing before the first launch.

**Jobs do not run from JupyterLab.** The `datasphere` CLI on your own machine
ships the config to a VM, streams the logs back, and downloads the results.
Nothing in the JupyterLab file manager is visible to a job by default — hence
the `attach-project-disk` flag below, which mounts the project storage into
the job read-only.

**It is billed per second** — training, the periodic evaluations, and pip
installing torch. See [Cost](#cost).

## One-time setup

1. **Dataset** — it already lives in the project storage (the folders you see
   in the JupyterLab root). Nothing else to do: [../jobs/run.py](../jobs/run.py)
   finds it by the same rule `train.py` uses, a folder holding `data.yaml`,
   `images/` and `timeofday.csv`.
2. **Comet** — project page → **Secrets** → add `COMET_API_KEY`. Project
   secrets become environment variables inside the job automatically, so
   [train.py:5](../train.py#L5) picks it up exactly as on Kaggle. It is *not*
   declared in the job config.
3. **Local CLI:**

   ```bash
   mkdir ~/nvpdyf && cd ~/nvpdyf
   python3 -m venv .venv && source .venv/bin/activate
   pip install datasphere yandexcloud
   git clone -b master https://github.com/Antonoof/Yandex-night-vision-Detection.git repo
   yc init          # or pass a token later: datasphere -t <oauth-token> ...
   ```

   The project id is in the console URL, or from
   `datasphere project list -c <community-id>`.

The local clone is used for two files only — `jobs/run.py` (the job's entry
point, uploaded automatically) and `jobs/requirements.txt` (read by the CLI to
build the VM's environment). The code that actually trains is whatever
`run.py` clones on the VM, so **push your branch before launching**.

## Launching

```bash
cd ~/nvpdyf/repo
datasphere project job execute -p <project-id> -c jobs/smoke.yaml   # ~15 min
datasphere project job execute -p <project-id> -c jobs/train.yaml   # ~3 h
```

Everything you normally change lives in one place, the `cmd:` block:

```yaml
cmd: >
  python3 jobs/run.py master              # <- branch to clone
  trainer.imgsz=960                       # <- from here on: train.py overrides,
  loss=box_heavy                          #    exactly as on the Kaggle command line
  trainer.run_name=30_boxheavy-imgsz960_yolov8n_nvpdyf
```

Keep those lines at the same indentation, tempting as it is to indent the
overrides under the entry point. A YAML `>` block folds equally-indented lines
into one, but keeps the newline of any line indented deeper — and a `cmd`
split across lines runs `train.py` with no overrides at all, silently. The
example in Yandex's own documentation is indented the second way.

Also bump `name:` when the run is a different experiment — job names are
unique within a project, the same discipline as `trainer.run_name`. To repeat
an identical run, `datasphere project job fork`.

`execute` blocks the shell. **Ctrl+C cancels the job**, it does not detach.
If the terminal dies the job survives but stops recording logs; come back with
`datasphere project job attach --id <job-id>` (the id is on the project's
**DataSphere Jobs** tab, or from `datasphere project job list`).

## What the smoke test proves

One epoch on 200 frames, at the real `imgsz=960` / `batch=16`. Read the log
top to bottom — it is structured as four banners:

| banner | what to check |
| --- | --- |
| `=== VM ===` | `cuda: True` (not a CPU-only torch), and free disk |
| `=== code ===` | the clone succeeded — i.e. the VM has internet, which Comet and `yolov8n.pt` also need |
| `=== data ===` | the dataset was found on the project disk and copied |
| `=== train ===` | `val: night=1860, day=7354`, then `nc: 80` → `nc: 7` (the head swap), then the two night/day tables |

## Results

**Logs.** `stdout.txt`, `stderr.txt`, `system.log` (VM setup and package
installation), `log.txt` and `gpu_stats.tsv` appear in the local working
directory; the path is printed as the very first line of `execute`.
`run.py` redirects training to stdout, so `stdout.txt` is the complete,
correctly ordered log — `stderr.txt` should be empty. `gpu_stats.tsv` answers
"is the dataloader starving the GPU", which is what `trainer.workers` is for.

**Files.** One archive, `artifacts.tgz`, holding the whole `saved/` tree:

```bash
tar xzf artifacts.tgz          # -> saved/runs/<run_name>/
```

Inside it: `weights/best.pt` and `last.pt`, `results.json`, `info.log`,
ultralytics' `results.csv` and its plots, and `predictions_night.png` /
`predictions_day.png`. Packing one archive under a fixed name means the job
config never needs editing when `trainer.run_name` changes, and nothing is
lost because a file was not listed by name.

Two caveats: outputs are downloaded **on success only** — a run that dies at
epoch 20 of 25 returns nothing — and the CLI will not download more than 1 GB
per job (fine here: the archive is tens of MB). Job data in DataSphere (cache,
logs, results) is kept 14 days; change with
`datasphere project job set-data-ttl --days`.

## Cost

`g1.1` is 72 billing units per second — roughly 340 ₽/hour; check the
calculator, it is the one number here not taken from the docs. Run 14 took
~5 hours on a Kaggle T4, so budget **~1000–1300 ₽** for run 30, plus ~100 ₽
for the first environment build (installing torch is billed at GPU rates;
later runs of the same job reuse the cache).

Two settings exist to hold that number down:

* `trainer.eval_zero_shot=false` — the zero-shot baseline is already measured
  (run 7), and leaving it on adds a full pass over 9214 val frames.
* `trainer.eval_every_k_epochs` stays at 3, i.e. **eight** extra night+day
  passes over 25 epochs — about a third of the bill. That is the price of the
  per-epoch night/day curve, which ultralytics' own aggregate mAP cannot
  show. Raise it to 5 if the budget matters more than the curve.

## Gotchas

| Symptom | Cause |
| --- | --- |
| training is absurdly slow, `cuda: False` in the first lines | `python: auto` would ship your laptop's CPU-only torch; the configs use `type: manual` for exactly this |
| `Python root module(-s) was not found automatically or set in config` | `cmd` does not start with a Python entry point — the CLI derives the root module from it, which is why `run.py` is Python and not a shell script |
| `InvalidRequirement: Expected package name...` | the CLI parses every line of `requirements-file` as a pip requirement and does not skip `#` comments — hence the separate, comment-free `jobs/requirements.txt` |
| `Main script ... must have line if __name__ == '__main__'` | the CLI imports the entry point **on your laptop** to inspect its dependencies; everything in `run.py` must stay under that guard |
| train.py ran with none of your overrides | the `cmd` lines were indented unequally, see above |
| `DS_PROJECT_HOME is not set` | the `attach-project-disk` flag is missing from the config |
| `no dataset under ...` | the dataset is deeper than 5 levels in the project storage, or is missing `timeofday.csv` |
| no VM ever becomes available | `g2.1` / `gt4.1` / `g1.4` have a **default quota of 0**; they open after topping up the billing account by ≥500 ₽ or a support request. `g1.1` does not |
| job dies on Ctrl+C when you only wanted to stop watching | that is what Ctrl+C does here; use `attach` to come back |
| "no space left" during the dataset copy | uncomment `working-storage` in `jobs/train.yaml` |
| job rejected on `name` | job names are unique per project — bump it |
| a hydra error at startup | the branch in `cmd` does not have the config group you passed (e.g. `loss=box_heavy` landed on `master` only after the contrast-analysis merge) |
