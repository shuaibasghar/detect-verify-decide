"""Resumable JSONL caches, JSON helpers and seeding.

Colab sessions disconnect. Every expensive stage writes one JSON line per image as soon as
it is processed, and skips images already present on restart.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# json: read/write the cache lines and result files.
import json
# os: only used for the `os.PathLike` type hint.
import os
# random: Python's own random generator, seeded for reproducibility.
import random
# pathlib.Path: portable file paths.
from pathlib import Path
# Iterable: type hint for "any collection of keys".
from typing import Iterable

# numpy: seeding numpy's generator and converting numpy values to plain JSON types.
import numpy as np


def set_seed(seed: int) -> None:
    """Seed every random generator we use so reruns give the same split, noise and sampling."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        # torch is optional here (tests run without it), hence the guarded import.
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _to_jsonable(o):
    """`default=` hook for json.dump: turn numpy scalars/arrays into plain Python so they can be saved."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    # Anything else is a genuine bug: fail loudly instead of writing garbage.
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path: str | os.PathLike) -> None:
    """Write `obj` as pretty-printed JSON, creating the folder if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path: str | os.PathLike):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JsonlCache:
    """Append-only per-image record store keyed by `key` (default: image_id)."""

    def __init__(self, path: str | os.PathLike, key: str = "image_id"):
        self.path = Path(path)
        # Field that identifies a record (image_id for detections, idx for POPE questions).
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory index of everything already on disk.
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
        """`k in cache`: has this key already been processed?"""
        return k in self._records

    def __len__(self) -> int:
        """Number of cached records."""
        return len(self._records)

    def get(self, k):
        """Record for key `k`, or None."""
        return self._records.get(k)

    def add(self, rec: dict) -> None:
        """Store a record in memory AND append it to the file immediately (so a crash loses nothing)."""
        self._records[rec[self.key]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_to_jsonable) + "\n")
            f.flush()

    def todo(self, keys: Iterable) -> list:
        """The keys that still have to be processed (not yet in the cache), in the given order."""
        return [k for k in keys if k not in self._records]

    def records(self, keys: Iterable | None = None) -> dict:
        """All records (dict key -> record), or only those for `keys` (missing keys are skipped)."""
        if keys is None:
            return dict(self._records)
        return {k: self._records[k] for k in keys if k in self._records}
