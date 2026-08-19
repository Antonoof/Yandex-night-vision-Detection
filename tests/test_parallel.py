import threading

from src.utils.init_utils import split_devices
from src.utils.parallel import run_paired


def test_split_devices_comma_pair():
    assert split_devices("0,1") == ["0", "1"]


def test_split_devices_single_value_unchanged():
    assert split_devices(0) == [0]
    assert split_devices("cpu") == ["cpu"]
    assert split_devices("mps") == ["mps"]


def test_split_devices_non_numeric_comma_is_not_split():
    # e.g. an already-single device string that happens to contain a comma
    # for some other reason must not be mistaken for a device list.
    assert split_devices("cuda:0,extra") == ["cuda:0,extra"]


def test_run_paired_single_device_runs_sequentially_in_caller_thread():
    caller_thread = threading.current_thread().ident
    seen_threads = []

    def job():
        seen_threads.append(threading.current_thread().ident)
        return "result"

    results = run_paired([job], devices=["0"])

    assert results == ["result"]
    assert seen_threads == [caller_thread]


def test_run_paired_two_devices_runs_concurrently():
    # both jobs must be running before either can finish, so this only
    # passes if they actually overlap in time rather than running one after
    # another.
    barrier = threading.Barrier(2, timeout=2)

    def job_sync(tag):
        barrier.wait()
        return tag

    results = run_paired(
        [lambda: job_sync("night"), lambda: job_sync("day")], devices=["0", "1"]
    )

    assert set(results) == {"night", "day"}


def test_run_paired_preserves_job_order():
    results = run_paired([lambda: 1, lambda: 2], devices=["0", "1"])
    assert results == [1, 2]
