"""Baseline (ii): Woodpecker-style single-agent self-verification.

One model both validates and decides, on global evidence, with no role separation:
the same frozen VLM as the Verifier Agent sees the FULL image plus the detector's candidate list
(labels + boxes) in ONE prompt and outputs keep/reject for every item. Its answer is the final
decision; detector confidence is not fused in.

The official Woodpecker code (https://github.com/BradyFU/Woodpecker) orchestrates GPT-3.5 +
Grounding DINO + BLIP-2 to correct captions. Re-implementing its validate-then-decide structure
with the same VLM and detector as our pipeline keeps the comparison like-for-like, so differences
are attributable to pipeline structure rather than model capacity (see docs/DATASETS_AND_BASELINES.md).

v2 (this file): the answer is requested as strict JSON ({"1": "keep", "2": "reject", ...}) with a one-line
example that contains a "reject". v1 asked for "N: yes" lines; the 2B VLM then echoed the list and answered
"yes" to everything (only ~52 % of the answers were parsable), which made the baseline unfair. Items that still
cannot be parsed are NOT defaulted to "yes": they are re-asked one by one as a plain yes/no on the full image.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# json: first-choice parser of the model's answer.
import json
# re: fallback parser ("3: yes", "3. reject", ...) and extraction of the JSON object from surrounding text.
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
    "Many detector claims are wrong, so reject every item you cannot clearly see.\n"
    "Reply with ONLY one JSON object that maps every item number to \"keep\" (really there) or \"reject\" "
    "(not there), for example for 3 items: {{\"1\": \"keep\", \"2\": \"reject\", \"3\": \"keep\"}}\n"
    "Give exactly {n} entries, no other text."
)
# Words that mean "the object is present" / "the object is not present" (JSON values and plain-text answers).
_YES = {"keep", "yes", "true", "y", "present"}
_NO = {"reject", "no", "false", "n", "absent"}
# Matches "3: yes", "3. No", "3) **reject**", '"3": "keep"' ... -> (item number, word).
_LINE = re.compile(r"(\d+)\W{0,3}?\s*[:.)\-=]\s*\W{0,3}(yes|no|keep|reject|true|false)\b", re.IGNORECASE)
# The first {...} block of the answer (the model may add text around the JSON).
_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def build_prompt(labels, boxes, width: int, height: int) -> str:
    """Numbered candidate list (label + box on a 0-1000 scale) inserted into the instruction."""
    lines = []
    for k, (l, b) in enumerate(zip(labels, boxes), 1):
        x1, y1, x2, y2 = b
        # Normalise pixel coordinates to 0-1000 so the prompt does not depend on the image size.
        nb = [round(1000 * x1 / width), round(1000 * y1 / height), round(1000 * x2 / width), round(1000 * y2 / height)]
        lines.append(f"{k}. {l} {nb}")
    return PROMPT.format(items="\n".join(lines), n=len(lines))


def _as_bool(v):
    """keep/yes/true -> True, reject/no/false -> False, anything else -> None."""
    if isinstance(v, bool):
        return v
    w = str(v).strip().strip("\"'*. ").lower()
    return True if w in _YES else (False if w in _NO else None)


def parse_answer(text: str, n: int) -> list:
    """List of True/False per item; None where the model gave no parsable answer.
    Order: (1) a JSON object {"1": "keep", ...}; (2) regex over "N: yes/no/keep/reject" lines for the items
    JSON did not settle. Out-of-range numbers are ignored and the first answer for an item wins."""
    out = [None] * n
    # (1) JSON object (also when it is wrapped in a markdown code fence or followed by chatter).
    for block in _JSON_BLOCK.findall(text or ""):
        try:
            obj = json.loads(block)
        except ValueError:
            continue
        if isinstance(obj, dict):
            for key, val in obj.items():
                try:
                    k = int(str(key).strip()) - 1
                except ValueError:
                    continue
                b = _as_bool(val)
                if 0 <= k < n and out[k] is None and b is not None:
                    out[k] = b
            break
    # (2) Regex fallback for whatever is still missing.
    for m in _LINE.finditer(text or ""):
        k = int(m.group(1)) - 1
        if 0 <= k < n and out[k] is None:
            out[k] = _as_bool(m.group(2))
    return out


def run_woodpecker_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn,
                         desc: str = "woodpecker-style") -> dict:
    """Per image: one generation call over the top-K candidates (full image + numbered list, JSON answer).
    Unlisted candidates are rejected. Listed items whose answer could not be parsed are re-asked one by one
    (plain yes/no on the full image); those extra calls are counted in `invocations` and `time_s`.
    Record keys: verdict, raw, parse_ok (first pass, before the fallback), n_fallback, parse_ok_final, time_s,
    invocations, listed."""
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
            recs[iid] = {"image_id": int(iid), "verdict": [False] * n, "raw": "", "parse_ok": 1.0, "n_fallback": 0,
                         "parse_ok_final": 1.0, "time_s": 0.0, "invocations": 1 if order else 0, "listed": len(order)}
            if order:
                image = image_fn(iid)
                w, h = image.size
                prompt = build_prompt([det["labels"][k] for k in order], [det["boxes"][k] for k in order], w, h)
                jobs.append((iid, order, fit_max_side(image, side), prompt))
        if jobs:  # one generation call for the whole chunk; time_s is amortised per image
            with Timer() as t:
                # Enough new tokens for '"k": "reject", ' per item, at most the largest list in this chunk.
                raws = verifier.generate([j[2] for j in jobs], [j[3] for j in jobs],
                                         max_new_tokens=max(10 * len(j[1]) + 16 for j in jobs))
            for (iid, order, view, _), raw in zip(jobs, raws):
                det, rec = det_records[iid], recs[iid]
                parsed = parse_answer(raw, len(order))
                rec.update(raw=raw, time_s=t.elapsed / len(jobs), parse_ok=float(np.mean([p is not None for p in parsed])))
                # Unparsed items are NOT defaulted to "yes": ask each of them once as a plain yes/no on the full image.
                miss = [j for j, p in enumerate(parsed) if p is None]
                if miss:
                    from ..agents.verifier import article
                    q = [cfg["verifier"]["prompt_existence"].format(article=article(det["labels"][order[j]]),
                                                                     label=det["labels"][order[j]]) for j in miss]
                    with Timer() as tf:
                        p_yes = verifier.yes_no([view] * len(miss), q)
                    for j, p in zip(miss, p_yes):
                        parsed[j] = bool(p > 0.5)
                    rec.update(n_fallback=len(miss), invocations=1 + len(miss), time_s=rec["time_s"] + tf.elapsed)
                for k, p in zip(order, parsed):
                    rec["verdict"][int(k)] = bool(p)
                rec["parse_ok_final"] = float(np.mean([p is not None for p in parsed]))
        for iid in chunk:
            cache.add(recs[iid])
        bar.update(len(chunk))
    bar.close()
    return cache.records(image_ids)


def parse_summary(recs: dict) -> dict:
    """Parse statistics of a finished stage: first-pass parse rate, final parse rate, items re-asked, keep rate."""
    r = [x for x in recs.values() if x.get("listed", 0)]
    listed = sum(x["listed"] for x in r)
    kept = sum(int(sum(x["verdict"])) for x in r)
    return {"parse_rate_first_pass": float(np.mean([x["parse_ok"] for x in r])) if r else 1.0,
            "parse_rate_final": float(np.mean([x.get("parse_ok_final", 1.0) for x in r])) if r else 1.0,
            "n_fallback_items": int(sum(x.get("n_fallback", 0) for x in r)), "n_listed_items": int(listed),
            "keep_rate": kept / listed if listed else float("nan")}
