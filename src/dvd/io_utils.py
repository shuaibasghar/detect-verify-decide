"""Resumable JSONL caches, JSON helpers and seeding.

Colab sessions disconnect. Every expensive stage writes one JSON line per image as soon as
it is processed, and skips images already present on restart.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JsonlCache:
    """Append-only per-image record store keyed by `key` (default: image_id)."""

    def __init__(self, path: str | os.PathLike, key: str = "image_id"):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # half-written last line after a disconnect
                    self._records[rec[key]] = rec

    def __contains__(self, k) -> bool:
        return k in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, k):
        return self._records.get(k)

    def add(self, rec: dict) -> None:
        self._records[rec[self.key]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_to_jsonable) + "\n")
            f.flush()

    def todo(self, keys: Iterable) -> list:
        return [k for k in keys if k not in self._records]

    def records(self, keys: Iterable | None = None) -> dict:
        if keys is None:
            return dict(self._records)
        return {k: self._records[k] for k in keys if k in self._records}
