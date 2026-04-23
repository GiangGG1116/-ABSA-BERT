import random
from typing import List

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return preds.eq(labels).float().mean().item()


def avg(values: List[float]) -> float:
    """Return the mean of a list; returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0