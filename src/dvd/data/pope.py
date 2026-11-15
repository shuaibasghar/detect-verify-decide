"""POPE (Li et al., EMNLP 2023): binary object-existence probing, three negative-sampling settings.

Primary source: the `lmms-lab/POPE` Hugging Face dataset (images embedded, no COCO val2014 zip).
Fallback: the original question files from the POPE GitHub repo + per-image download of the
500 referenced COCO val2014 images.
Items are yielded lazily so 3000 decoded images never sit in RAM at once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from PIL import Image

from .download import download

POPE_GITHUB = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{setting}.json"
COCO2014_IMG = "http://images.cocodataset.org/val2014/{name}"
_OBJ_RE = re.compile(r"is there an? (.+?) in the im", re.IGNORECASE)


def question_object(question: str) -> str:
    m = _OBJ_RE.search(question)
    return m.group(1).strip() if m else ""


def _from_hf(cfg: dict, setting: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg["pope"]["hf_dataset"], split="test")
    ds = ds.filter(lambda c: c == setting, input_columns=["category"])
    for n, row in enumerate(ds):
        q = row["question"]
        yield {
            "question_id": int(row.get("question_id") or row.get("id") or n),
            "question": q,
            "object": question_object(q),
            "label": str(row["answer"]).strip().lower(),
            "image": row["image"].convert("RGB"),
            "image_source": row.get("image_source", ""),
        }


def _from_github(cfg: dict, setting: str) -> Iterator[dict]:
    root = Path(cfg["paths"]["data_root"]) / "pope"
    qfile = download(POPE_GITHUB.format(setting=setting), root / f"coco_pope_{setting}.json")
    with open(qfile, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        img_path = root / "images" / r["image"]
        if not img_path.exists():
            download(COCO2014_IMG.format(name=r["image"]), img_path)
        yield {
            "question_id": int(r["question_id"]),
            "question": r["text"],
            "object": question_object(r["text"]),
            "label": r["label"].strip().lower(),
            "image": Image.open(img_path).convert("RGB"),
            "image_source": r["image"],
        }


def iter_pope(cfg: dict, setting: str, max_n: int | None = None) -> Iterator[dict]:
    """Yield POPE items for `setting` in {random, popular, adversarial}."""
    try:
        gen = _from_hf(cfg, setting)
        first = next(gen)
    except Exception as e:  # noqa: BLE001 - any hub/network/schema failure -> fallback
        print(f"[pope] HF dataset unavailable ({type(e).__name__}: {e}); using GitHub fallback")
        gen = _from_github(cfg, setting)
        first = next(gen)
    # `idx` = position within the setting (deterministic order); used as the resumable cache key
    first["idx"] = 0
    yield first
    n = 1
    for item in gen:
        if max_n is not None and n >= max_n:
            return
        item["idx"] = n
        yield item
        n += 1


def load_pope(cfg: dict, setting: str, max_n: int | None = None) -> list[dict]:
    return list(iter_pope(cfg, setting, max_n))
