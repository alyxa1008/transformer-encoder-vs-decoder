"""YAML 설정 로더. 모든 실험 설정은 configs/*.yaml 에 두고 코드에 하드코딩하지 않는다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "task" not in cfg:
        raise ValueError(f"config {path} has no 'task' key")
    cfg["_config_path"] = str(path)
    return cfg
