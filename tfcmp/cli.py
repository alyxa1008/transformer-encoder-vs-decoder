"""단일 진입점: python -m tfcmp run --config configs/xxx.yaml

config의 `task` 키로 무엇을 실행할지 정한다. 모델 학습 task는 M2~M4에서 추가된다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tfcmp.config import load_config
from tfcmp.utils.logging import get_logger
from tfcmp.utils.seed import set_seed


def _dispatch(cfg: dict, log):
    task = cfg["task"]
    if task == "prepare_nsmc":
        from tfcmp.data.nsmc import prepare_nsmc

        return prepare_nsmc(cfg, log)
    if task == "make_reverse_data":
        from tfcmp.data.reverse import make_reverse_data

        return make_reverse_data(cfg, log)
    raise ValueError(f"unknown task: {task}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tfcmp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a task described by a yaml config")
    run.add_argument("--config", required=True)
    run.add_argument("--split", default=None, help="override split (test must be requested explicitly)")
    run.add_argument("--seed", type=int, default=None, help="override seed")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split is not None:
        cfg["split"] = args.split
    name = Path(args.config).stem
    log = get_logger("tfcmp", log_file=Path(cfg.get("log_dir", "results/logs")) / f"{name}.log")
    log.info(f"config={args.config} task={cfg['task']} seed={cfg.get('seed')}")
    set_seed(int(cfg.get("seed", 0)))
    _dispatch(cfg, log)
    return 0
