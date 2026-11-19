"""POPE (Li et al., EMNLP 2023): binary object-existence probing, three negative-sampling settings.

Primary source: the `lmms-lab/POPE` Hugging Face dataset (images embedded, no COCO val2014 zip).
Fallback: the original question files from the POPE GitHub repo + per-image download of the
500 referenced COCO val2014 images.
Items are yielded lazily so 3000 decoded images never sit in RAM at once.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# json: parse the POPE question files (one JSON object per line).
import json
# re: regular expression that extracts the object name from a question.
import re
# random: seeded balanced sampling of the questions.
import random
# pathlib.Path: portable file paths.
from pathlib import Path
# Iterator: type hint for the lazy generators below.
from typing import Iterator

# Pillow: open the downloaded images.
from PIL import Image

# Download helper (used for the question files and the individual images).
from .download import download

# Original POPE question files on GitHub; {setting} = random | popular | adversarial.
POPE_GITHUB = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{setting}.json"
# POPE uses COCO *val2014* images; this is the URL of one of them.
COCO2014_IMG = "http://images.cocodataset.org/val2014/{name}"
# Pulls "dog" out of "Is there a dog in the image?" (also handles "an").
_OBJ_RE = re.compile(r"is there an? (.+?) in the im", re.IGNORECASE)


def question_object(question: str) -> str:
    """The object a POPE question asks about ('' if the question does not match the template)."""
    m = _OBJ_RE.search(question)
    return m.group(1).strip() if m else ""


def _github_rows(cfg: dict, setting: str) -> list[dict]:
    """The raw question rows of one setting (small JSON-lines file from the POPE repo; no images are downloaded)."""
    root = Path(cfg["paths"]["data_root"]) / "pope"
    qfile = download(POPE_GITHUB.format(setting=setting), root / f"coco_pope_{setting}.json")
    with open(qfile, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def select_rows(rows: list[dict], setting: str, max_n: int | None, sample: str = "head", seed: int = 42) -> list[dict]:
    """Which questions are used. "head": the first `max_n` rows of the file (original behaviour).
    "balanced_seeded": `max_n` rows, half "yes" and half "no", drawn with a seed that includes the setting name, so
    different settings draw their own questions; the original file order is kept."""
    if sample == "head" or max_n is None:
        return rows if max_n is None else rows[:max_n]
    if sample != "balanced_seeded":
        raise ValueError(f"Unknown POPE sample mode: {sample}")
    rng = random.Random(f"{seed}-{setting}")
    yes = [k for k, r in enumerate(rows) if r["label"].strip().lower() == "yes"]
    no = [k for k, r in enumerate(rows) if r["label"].strip().lower() != "yes"]
    keep = sorted(rng.sample(yes, min(max_n // 2, len(yes))) + rng.sample(no, min(max_n - max_n // 2, len(no))))
    return [rows[k] for k in keep]


def question_keys(cfg: dict, setting: str, max_n: int | None, sample: str = "head") -> list[tuple]:
    """(image, question text) of the questions that would be used - for comparing settings without any model run."""
    return [(r["image"], r["text"]) for r in select_rows(_github_rows(cfg, setting), setting, max_n, sample,
                                                         cfg.get("seed", 42))]


def overlap_report(cfg: dict, settings, max_n: int | None, samples: dict | None = None) -> list[dict]:
    """Pairwise overlap of the question sets of the settings (fraction of identical (image, question) pairs).
    Used to detect that two settings are really the same 300 questions (then the results are trivially equal)."""
    samples = samples or {}
    keys = {s: set(question_keys(cfg, s, max_n, samples.get(s, "head"))) for s in settings}
    out = []
    for i, a in enumerate(settings):
        for b in settings[i + 1:]:
            out.append({"a": a, "b": b, "n_a": len(keys[a]), "n_b": len(keys[b]),
                        "identical_fraction": len(keys[a] & keys[b]) / max(min(len(keys[a]), len(keys[b])), 1)})
    return out


def _from_hf(cfg: dict, setting: str, max_n: int | None = None, sample: str = "head") -> Iterator[dict]:
    """Yield POPE items from the Hugging Face dataset (large download; images are embedded).
    (`sample` is only supported by the GitHub source; here the first questions are used.)"""
    # `datasets` is imported lazily: it is only needed on this path.
    from datasets import load_dataset

    ds = load_dataset(cfg["pope"]["hf_dataset"], split="test")
    # Keep only the rows of the requested setting (random / popular / adversarial).
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


def _from_github(cfg: dict, setting: str, max_n: int | None = None, sample: str = "head") -> Iterator[dict]:
    """Yield POPE items from the original question file; each image is downloaded only when first needed."""
    root = Path(cfg["paths"]["data_root"]) / "pope"
    # The question file is small (a few hundred KB); `sample` decides which questions are used.
    rows = select_rows(_github_rows(cfg, setting), setting, max_n, sample, cfg.get("seed", 42))
    for r in rows:
        img_path = root / "images" / r["image"]
        # Lazy download: a run that stops after 300 questions only fetches ~50 images.
        if not img_path.exists():
            download(COCO2014_IMG.format(name=r["image"]), img_path, quiet=True)
        yield {
            "question_id": int(r["question_id"]),
            "question": r["text"],
            "object": question_object(r["text"]),
            "label": r["label"].strip().lower(),
            "image": Image.open(img_path).convert("RGB"),
            "image_source": r["image"],
        }


def iter_pope(cfg: dict, setting: str, max_n: int | None = None, sample: str = "head") -> Iterator[dict]:
    """Yield POPE items for `setting` in {random, popular, adversarial}.
    `sample`: "head" = first `max_n` questions of the file (default, as before); "balanced_seeded" = a seeded,
    yes/no-balanced draw of `max_n` questions of this setting (GitHub source only)."""
    # cfg["pope"]["source"]: "github" = question files + only the images actually used (fast for small max_n),
    # "hf" = the full lmms-lab/POPE dataset (large download). The other one is the fallback.
    order = [_from_github, _from_hf] if cfg["pope"].get("source", "hf") == "github" else [_from_hf, _from_github]
    gen = first = None
    for k, src in enumerate(order):
        try:
            gen = src(cfg, setting, max_n, sample)
            # Generators are lazy: errors only surface when the first item is requested.
            first = next(gen)
            break
        except Exception as e:  # noqa: BLE001 - any hub/network/schema failure -> fallback
            # If the last source also fails there is nothing left to try.
            if k == len(order) - 1:
                raise
            print(f"[pope] {src.__name__} unavailable ({type(e).__name__}: {e}); trying {order[k + 1].__name__}")
    # `idx` = position within the setting (deterministic order); used as the resumable cache key
    first["idx"] = 0
    yield first
    n = 1
    for item in gen:
        # Stop after `max_n` items (None = all 3000).
        if max_n is not None and n >= max_n:
            return
        item["idx"] = n
        yield item
        n += 1


def load_pope(cfg: dict, setting: str, max_n: int | None = None) -> list[dict]:
    """Eager version of `iter_pope`: all items in a list (uses more RAM)."""
    return list(iter_pope(cfg, setting, max_n))
