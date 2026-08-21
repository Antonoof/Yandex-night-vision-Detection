"""What runs on the DataSphere VM: check the VM, clone the code, find the
dataset on the mounted project storage, train, pack the results up.

    python3 jobs/run.py <branch> [any train.py overrides...]

Two things about this file are dictated by the `datasphere` CLI rather than
by taste, and both are load-bearing:

* **It is Python, not a shell script.** The CLI derives the job's "root
  module" from the first argument of ``cmd`` and refuses to build an
  environment without one (``Python root module(-s) was not found
  automatically or set in config``). A ``bash ...`` entry point has no root
  module. As the root module, this file is also uploaded automatically - it
  does not need to be listed in the config's ``inputs``.
* **Everything runs under the ``__main__`` guard.** To find out which
  packages the job needs, the CLI *imports this file on your laptop* at
  submit time (``datasphere/utils.py:_get_module_namespace``) - and it
  refuses to run a main script that has no such guard. Anything at module
  level would execute at home, on submit.

For the same reason nothing here imports torch or any project code: the
import happens locally, where neither is necessarily installed. The checks
run as subprocesses instead.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/Antonoof/Yandex-night-vision-Detection.git"
DATASET_MARKER = "timeofday.csv"  # the same marker bdd100k.find_dataset_root uses


def banner(text):
    print(f"\n=== {text} " + "=" * max(0, 68 - len(text)), flush=True)


def run(cmd, **kwargs):
    """Run a command, streaming its output into ours, and stop on failure."""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def find_dataset(root, max_depth=5):
    """Locate the dataset folder under the project storage.

    Looks for the same marker file train.py does, so the dataset can sit
    anywhere in the project storage without being configured here.
    """
    root = root.rstrip("/")
    base_depth = root.count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        if DATASET_MARKER in filenames:
            return dirpath
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: run.py <branch> [train.py overrides...]")
    branch, overrides = sys.argv[1], sys.argv[2:]

    banner("VM")
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(
        [sys.executable, "-c", "import torch; print('torch', torch.__version__, "
         "'| cuda:', torch.cuda.is_available())"],
        check=False,
    )
    run(["df", "-h", "."])

    banner(f"code: {branch}")
    run(["git", "clone", "--depth", "1", "-b", branch, REPO_URL, "repo"])
    os.chdir("repo")
    run(["git", "log", "-1", "--oneline"])

    banner("data")
    # The project storage is mounted read-only by the attach-project-disk flag.
    project_home = os.environ.get("DS_PROJECT_HOME")
    if not project_home:
        sys.exit("DS_PROJECT_HOME is not set - add 'flags: [attach-project-disk]'")
    src = find_dataset(project_home)
    if src is None:
        subprocess.run(["ls", "-la", project_home], check=False)
        sys.exit(f"no dataset under {project_home}: need a folder holding "
                 f"data.yaml, images/ and {DATASET_MARKER}")
    print(f"found {src}", flush=True)
    # Copied to the local disk on purpose: the project storage is network
    # backed, and every one of 36709 images is read on every one of 25 epochs.
    os.makedirs("data", exist_ok=True)
    run(["cp", "-r", src, "data/"])
    run(["du", "-sh", "data"])
    run(["df", "-h", "."])

    banner("train")
    # stderr into stdout: the ultralytics tables and the tqdm bars then land
    # in one file in the right order, instead of split across stdout.txt and
    # stderr.txt with the ordering lost.
    run([sys.executable, "train.py", *overrides], stderr=subprocess.STDOUT)

    banner("artifacts")
    # One archive of the whole run directory, under a fixed name: the job
    # config then never needs editing when trainer.run_name changes, and
    # nothing is lost because a plot ultralytics writes was not listed.
    os.chdir("..")
    run(["tar", "czf", "artifacts.tgz", "-C", "repo", "saved"])
    run(["ls", "-lh", "artifacts.tgz"])
    run(["tar", "tzf", "artifacts.tgz"])


if __name__ == "__main__":
    main()
