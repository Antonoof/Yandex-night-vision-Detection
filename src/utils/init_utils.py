import os
import random

import numpy as np
import torch


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device_cfg):
    """
    Resolve a device config value to what ultralytics expects.

    Args:
        device_cfg (str): "auto", or an explicit device (e.g. "cpu", "mps",
            "0", "0,1").
    Returns:
        device (int | str): 0 for the first CUDA GPU, "cpu" otherwise, or
            device_cfg unchanged if it wasn't "auto".
    """
    if device_cfg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device_cfg


def split_devices(device):
    """
    Split a resolved device into the individual GPUs it names.

    ``trainer.device="0,1"`` is also the ultralytics convention for launching
    DDP training across two GPUs; this reuses the same string to decide
    whether the *other* per-split work in the pipeline (Zero-DCE on
    train/val, night/day evaluation) can run two single-GPU jobs in parallel
    instead of one after another.

    Args:
        device (int | str): output of resolve_device, e.g. 0, "cpu", "0,1".
    Returns:
        devices (list[int | str]): ["0", "1"] for "0,1"; otherwise a single-
            element list holding `device` unchanged.
    """
    text = str(device)
    if "," in text and all(part.strip().isdigit() for part in text.split(",")):
        return [part.strip() for part in text.split(",")]
    return [device]
