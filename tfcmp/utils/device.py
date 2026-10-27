"""디바이스 자동 선택: Apple Silicon이면 mps, 아니면 cpu."""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
