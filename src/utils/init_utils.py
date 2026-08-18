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
