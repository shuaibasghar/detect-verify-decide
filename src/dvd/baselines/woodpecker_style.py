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
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# re: parse the model's "1: yes" style answer lines.
import re

# numpy: sort candidates by score, average the parse rate.
import numpy as np
# tqdm: progress bar over images.
from tqdm.auto import tqdm

# CUDA-synchronised stopwatch.
from ..cost import Timer
# Shrinks the full image to the configured size.
from ..cropping import fit_max_side
# Resumable per-image JSONL cache.
from ..io_utils import JsonlCache

# Instruction given to the VLM together with the image; boxes are normalised to a 0-1000 grid.
PROMPT = (
    "You are checking the output of an object detector. Below is a numbered list of objects it claims "
    "are in this image, each with a bounding box [x1, y1, x2, y2] on a 0-1000 scale.\n{items}\n"
    "For each item, look at the image and decide whether that object is really present at that location. "
    "Reply with one line per item in the form '<number>: yes' or '<number>: no', and nothing else."
)
# Matches "3: yes", "3. No", "3) **yes**" ... -> (item number, yes|no).
_LINE = re.compile(r"(\d+)\s*[:.)\-]\s*\**\s*(yes|no)", re.IGNORECASE)


def build_prompt(labels, boxes, width: int, height: int) -> str:
    """Numbered candidate list (label + box on a 0-1000 scale) inserted into the instruction."""
    lines = []
    for k, (l, b) in enumerate(zip(labels, boxes), 1):
        x1, y1, x2, y2 = b
        # Normalise pixel coordinates to 0-1000 so the prompt does not depend on the image size.
        nb = [round(1000 * x1 / width), round(1000 * y1 / height), round(1000 * x2 / width), round(1000 * y2 / height)]
        lines.append(f"{k}. {l} {nb}")
    return PROMPT.format(items="\n".join(lines))


def parse_answer(text: str, n: int) -> list:
    """List of True/False per item; None where the model gave no parsable answer."""
    out = [None] * n
    for m in _LINE.finditer(text):
        # Model numbers items from 1; convert to a 0-based index.
        k = int(m.group(1)) - 1
        # Ignore out-of-range numbers and keep the first answer if an item is repeated.
        if 0 <= k < n and out[k] is None:
            out[k] = m.group(2).lower() == "yes"
    return out


def run_woodpecker_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn,
                         desc: str = "woodpecker-style") -> dict:
    """Per image: one generation call over the top-K candidates. Unlisted candidates are rejected;
    listed-but-unparsed ones keep the detector's claim (recorded in `parse_ok`)."""
    k_max = cfg["woodpecker"]["max_candidates"]
    # How many images share one generation call (1 = true per-image latency).
    bs = max(1, int(cfg["woodpecker"].get("batch_size", 1)))
    side = cfg["verifier"]["full_image_max_side"]
    cache = JsonlCache(cache_path)
    todo = cache.todo(image_ids)
    bar = tqdm(total=len(todo), desc=desc)
    for s in range(0, len(todo), bs):
        # `chunk` = the images of this generation call; `recs` their records; `jobs` the ones needing the VLM.
        chunk, recs, jobs = todo[s:s + bs], {}, []
        for iid in chunk:
            det = det_records[iid]
            n = len(det["scores"])
            # Candidates listed in the prompt: the K most confident ones.
            order = list(np.argsort(-np.asarray(det["scores"]), kind="stable")[:k_max]) if n else []
            # Default record: every candidate rejected (unlisted candidates stay rejected).
            recs[iid] = {"image_id": int(iid), "verdict": [False] * n, "raw": "", "parse_ok": 1.0,
                         "time_s": 0.0, "invocations": 1 if order else 0, "listed": len(order)}
            if order:
                image = image_fn(iid)
                w, h = image.size
                prompt = build_prompt([det["labels"][k] for k in order], [det["boxes"][k] for k in order], w, h)
                jobs.append((iid, order, fit_max_side(image, side), prompt))
        if jobs:  # one generation call for the whole chunk; time_s is amortised per image
            with Timer() as t:
                # Enough new tokens for "k: yes\n" per item, at most the largest list in this chunk.
                raws = verifier.generate([j[2] for j in jobs], [j[3] for j in jobs],
                                         max_new_tokens=max(8 * len(j[1]) + 16 for j in jobs))
            for (iid, order, _, _), raw in zip(jobs, raws):
                parsed = parse_answer(raw, len(order))
                for k, p in zip(order, parsed):
                    # Unparsable item (None): keep the detector's claim (True) instead of guessing "no".
                    recs[iid]["verdict"][int(k)] = True if p is None else p
                recs[iid].update(raw=raw, time_s=t.elapsed / len(jobs),
                                 parse_ok=float(np.mean([p is not None for p in parsed])))
        for iid in chunk:
            cache.add(recs[iid])
        bar.update(len(chunk))
    bar.close()
    return cache.records(image_ids)
