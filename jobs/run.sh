#!/usr/bin/env bash
# What runs on the DataSphere VM. A job VM starts empty, so: check the VM,
# clone the code, find the dataset on the mounted project storage, train,
# pack everything up.
#
#   bash jobs/run.sh <branch> [any train.py overrides...]
set -euo pipefail

BRANCH="$1"; shift
REPO_URL="https://github.com/Antonoof/Yandex-night-vision-Detection.git"

echo "=== VM ==============================================================="
nvidia-smi || echo "WARNING: no GPU on this VM - training will be unusably slow"
python3 -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo "disk: $(df -h . | tail -1)"

echo "=== code: $BRANCH ===================================================="
git clone --depth 1 -b "$BRANCH" "$REPO_URL" repo
cd repo
echo "commit $(git rev-parse --short HEAD)"

echo "=== data ============================================================="
# The project storage is mounted read-only by the attach-project-disk flag.
# Find the dataset by the same rule train.py uses (a folder holding data.yaml,
# images/ and timeofday.csv) so nothing has to be configured by hand.
HOME_DIR="${DS_PROJECT_HOME:-}"
if [ -z "$HOME_DIR" ]; then
    echo "DS_PROJECT_HOME is not set - add 'flags: [attach-project-disk]'"; exit 1
fi
CSV="$(find "$HOME_DIR" -maxdepth 5 -name timeofday.csv -print -quit)"
if [ -z "$CSV" ]; then
    echo "no dataset under $HOME_DIR (need a folder with data.yaml + images/ + timeofday.csv)"
    ls -la "$HOME_DIR"; exit 1
fi
# Copied to the local disk on purpose: the project storage is network-backed,
# and 36709 images get read on every one of 25 epochs.
mkdir -p data
cp -r "$(dirname "$CSV")" data/
du -sh data/*
echo "disk: $(df -h . | tail -1)"

echo "=== train ============================================================"
# 2>&1 so the ultralytics tables and the tqdm bars end up in one file in the
# right order, instead of split across stdout.txt and stderr.txt.
python3 train.py "$@" 2>&1

echo "=== artifacts ========================================================"
# One archive of the whole run directory, under a fixed name: the job config
# then never has to be edited when trainer.run_name changes, and nothing is
# lost because a plot ultralytics writes was not listed by name.
cd ..
tar czf artifacts.tgz -C repo saved
ls -lh artifacts.tgz
tar tzf artifacts.tgz
