"""NSMC 원본(ratings_train.txt / ratings_test.txt)을 data/raw/ 에 내려받는다.

출처: https://github.com/e9t/nsmc (Lucy Park, 2016). 원본 데이터는 커밋하지 않는다.
사용: python scripts/download_nsmc.py [--out data/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/e9t/nsmc/master/"
FILES = ["ratings_train.txt", "ratings_test.txt"]


def download(url: str, dst: Path) -> None:
    print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out / name
        if dst.exists() and not args.force:
            print(f"[skip] {dst} exists")
        else:
            download(BASE + name, dst)
        data = dst.read_bytes()
        n_lines = data.count(b"\n")
        print(f"[ok] {name}: {len(data):,} bytes, {n_lines:,} lines, sha256={hashlib.sha256(data).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
