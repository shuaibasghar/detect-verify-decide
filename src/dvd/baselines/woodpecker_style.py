"""Baseline (ii): Woodpecker-style single-agent self-verification.

One model both validates and decides, on global evidence, with no role separation:
the same frozen VLM as the Verifier Agent sees the FULL image plus the detector's candidate list
(labels + boxes) in ONE prompt and outputs keep/reject for every item. Its answer is the final
decision; detector confidence is not fused in.

The official Woodpecker code (https://github.com/BradyFU/Woodpecker) orchestrates GPT-3.5 +
Grounding DINO + BLIP-2 to correct captions. Re-implementing its validate-then-decide structure
with the same VLM and detector as our pipeline keeps the comparison like-for-like, so differences
are attributable to pipeline structure rather than model capacity (see docs/DATASETS_AND_BASELINES.md).
"""
from __future__ import annotations

import re

import numpy as np
from tqdm.auto import tqdm

from ..cost import Timer
from ..cropping import fit_max_side
from ..io_utils import JsonlCache

PROMPT = (
    "You are checking the output of an object detector. Below is a numbered list of objects it claims "
    "are in this image, each with a bounding box [x1, y1, x2, y2] on a 0-1000 scale.\n{items}\n"
    "For each item, look at the image and decide whether that object is really present at that location. "
    "Reply with one line per item in the form '<number>: yes' or '<number>: no', and nothing else."
)
_LINE = re.compile(r"(\d+)\s*[:.)\-]\s*\**\s*(yes|no)", re.IGNORECASE)


def build_prompt(labels, boxes, width: int, height: int) -> str:
    lines = []
    for k, (l, b) in enumerate(zip(labels, boxes), 1):
        x1, y1, x2, y2 = b
        nb = [round(1000 * x1 / width), round(1000 * y1 / height), round(1000 * x2 / width), round(1000 * y2 / height)]
        lines.append(f"{k}. {l} {nb}")
    return PROMPT.format(items="\n".join(lines))


def parse_answer(text: str, n: int) -> list:
    """List of True/False per item; None where the model gave no parsable answer."""
    out = [None] * n
    for m in _LINE.finditer(text):
        k = int(m.group(1)) - 1
        if 0 <= k < n and out[k] is None:
            out[k] = m.group(2).lower() == "yes"
    return out


def run_woodpecker_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn,
                         desc: str = "woodpecker-style") -> dict:
    """Per image: one generation call over the top-K candidates. Unlisted candidates are rejected;
    listed-but-unparsed ones keep the detector's claim (recorded in `parse_ok`)."""
    k_max = cfg["woodpecker"]["max_candidates"]
    side = cfg["verifier"]["full_image_max_side"]
    cache = JsonlCache(cache_path)
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        det = det_records[iid]
        n = len(det["scores"])
        order = list(np.argsort(-np.asarray(det["scores"]), kind="stable")[:k_max]) if n else []
        rec = {"image_id": int(iid), "verdict": [False] * n, "raw": "", "parse_ok": 1.0,
               "time_s": 0.0, "invocations": 1 if order else 0, "listed": len(order)}
        if order:
            image = image_fn(iid)
            w, h = image.size
            prompt = build_prompt([det["labels"][k] for k in order], [det["boxes"][k] for k in order], w, h)
            with Timer() as t:
                raw = verifier.generate([fit_max_side(image, side)], [prompt], max_new_tokens=8 * len(order) + 16)[0]
            parsed = parse_answer(raw, len(order))
            for k, p in zip(order, parsed):
                rec["verdict"][int(k)] = True if p is None else p
            rec.update(raw=raw, time_s=t.elapsed, parse_ok=float(np.mean([p is not None for p in parsed])))
        cache.add(rec)
    return cache.records(image_ids)
