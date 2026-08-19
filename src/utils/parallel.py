"""Run independent per-device jobs concurrently on multiple GPUs.

ultralytics' own multi-GPU support only covers ``model.train()`` (DDP via
``device="0,1"``). Everything else in this project's pipeline - Zero-DCE
preprocessing, zero-shot eval, periodic in-training eval, the final
night/day pass - is a pair of independent single-GPU jobs (night vs. day, or
the train split vs. the val split). This is what lets that second half of
the pipeline actually use a second T4 instead of leaving it idle.

Threads, not processes: torch releases the GIL around CUDA kernels and image
I/O, so two independent forward passes on two GPUs really do overlap in wall
clock time without the overhead/pickling cost of separate processes. With a
single device, callers are expected to skip this module entirely and just
call their function directly - see ``run_paired``'s single-device shortcut.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def run_paired(jobs: list[Callable[[], T]], devices: list) -> list[T]:
    """Run zero-arg callables in parallel, one per device.

    Args:
        jobs: zero-arg callables, each already bound (e.g. via
            functools.partial) to the device it should run on.
        devices: devices available for this call - only its length decides
            whether jobs run concurrently or sequentially; the callables
            themselves already know which device they target.
    Returns:
        Results in the same order as ``jobs``.
    """
    if len(devices) < 2 or len(jobs) < 2:
        return [job() for job in jobs]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [future.result() for future in futures]
