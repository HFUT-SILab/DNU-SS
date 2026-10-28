"""Reproducible execution settings and portable device selection."""

import random
import secrets
import numpy as np
import torch


def configure(seed=None, device="cuda:0", strict=False):
    seed = secrets.randbelow(100000) + 1 if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strict:
        random.seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use --device cpu for smoke tests"
        )
    return seed, selected
