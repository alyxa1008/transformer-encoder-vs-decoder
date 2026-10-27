"""랜덤 시드 고정 (python / numpy / torch)."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # 데이터 전처리 단계에서는 torch 없이도 동작
        pass
