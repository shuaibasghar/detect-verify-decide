"""Orchestration of the three agents.

Two ways of running the pipeline:

1. Offline, cached stages (used for all experiments):
     run_detection_stage    -> results/detections/<name>.jsonl   (every candidate, c >= box_threshold)
     run_verification_stage -> results/verifier/<name>.jsonl     (verdict for every candidate)
   Because the Decision Agent is a cheap rule, the whole (tau_low, tau_high) grid is then
   evaluated from the caches without re-running any model.

2. Online, gated (`DetectVerifyDecide`): the deployable single fixed pass. Only candidates in the
   uncertain band tau_low <= c < tau_high are cropped and verified. Used to measure real latency.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# time: measure the time spent in UI callbacks.
import time
# Callable: type hint for functions passed in (image loader, event callback).
from typing import Callable

# numpy: score/verdict arrays.
import numpy as np
# tqdm: progress bars for the long offline loops.
from tqdm.auto import tqdm

# The deployable decision rule object.
from .agents.decision import RuleDecisionAgent
# make_views builds the crops (or full-image views) the verifier looks at.
from .agents.verifier import make_views
# CUDA-synchronised stopwatch.
from .cost import Timer
# Resumable per-image JSONL cache.
from .io_utils import JsonlCache


def run_detection_stage(detector, image_ids, cache_path, image_fn: Callable, desc: str = "detect") -> dict:
    """Run the Detector Agent over `image_ids` (resumable). image_fn(image_id) -> PIL image."""
    cache = JsonlCache(cache_path)
    # `todo` skips images that are already in the cache, so a restart continues where it stopped.
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        rec = detector.detect(image_fn(iid))
        rec["image_id"] = int(iid)
        cache.add(rec)
    return cache.records(image_ids)


def verify_global(verifier, image, labels: list[str], cfg: dict) -> dict:
    """Global-view verification. The question depends only on the label and the image is the same for every
    candidate, so each DISTINCT label is asked once and its answer is reused for every candidate carrying that
    label (identical input -> identical output). Saves the repeated full-image forward passes."""
    v = cfg["verifier"]
    # Distinct labels, sorted so the order is deterministic.
    uniq = sorted(set(labels))
    views = make_views(image, uniq, cfg, "full")          # the full view ignores boxes; only the count matters
    out = verifier.verify(views, uniq, "existence", v["prompt_existence"], v["prompt_open"], v["open_max_new_tokens"])
    # label -> answer lookup tables, then expand back to one answer per candidate.
    p, d = dict(zip(uniq, out["p_yes"])), dict(zip(uniq, out["verdict"]))
    return {"p_yes": [p[l] for l in labels], "verdict": [bool(d[l]) for l in labels], "answers": None}


def run_verification_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn: Callable,
                           view: str | None = None, margin: float | None = None, mode: str | None = None,
                           desc: str = "verify") -> dict:
    """Run the Verifier Agent on every cached candidate (resumable).

    `verify_invocations` counts candidates (the unit of the cost model), also for the global view, where the
    measured time is smaller because identical (image, label) questions are asked once."""
    v = cfg["verifier"]
    # Defaults come from the config; the margin search passes its own margin explicitly.
    view, mode = view or v["view"], mode or v["mode"]
    margin = v["crop_margin"] if margin is None else margin
    cache = JsonlCache(cache_path)
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        det = det_records[iid]
        n = len(det["scores"])
        # Record skeleton; images without candidates keep the empty lists.
        rec = {"image_id": int(iid), "view": view, "margin": margin, "mode": mode,
               "p_yes": [], "verdict": [], "answers": None, "verify_time_s": 0.0, "verify_invocations": n}
        if n:
            image = image_fn(iid)
            with Timer() as t:
                if view == "full" and mode == "existence":
                    out = verify_global(verifier, image, det["labels"], cfg)
                else:
                    views = make_views(image, det["boxes"], cfg, view, margin)
                    out = verifier.verify(views, det["labels"], mode, v["prompt_existence"], v["prompt_open"],
                                          v["open_max_new_tokens"])
            rec.update(out, verify_time_s=t.elapsed)
        cache.add(rec)
    return cache.records(image_ids)


def verdicts_from(ver_records: dict, key: str = "verdict") -> dict:
    """Convert cached verifier records into numpy arrays per image (bool verdicts or float P(yes))."""
    return {i: np.asarray(r[key], dtype=bool if key == "verdict" else np.float32) for i, r in ver_records.items()}


class _Emitter:
    """Forwards progress events to an optional callback and tracks the time spent inside it,
    so UI refreshes are never counted as model latency."""

    def __init__(self, callback):
        # `spent` accumulates the seconds used inside the callback.
        self.callback, self.spent = callback, 0.0

    def __call__(self, stage: str, msg: str, progress: float, **data) -> None:
        """Emit one event {stage, msg, progress, ...}; a no-op when no callback was given."""
        if self.callback is None:
            return
        t0 = time.perf_counter()
        self.callback({"stage": stage, "msg": msg, "progress": float(progress), **data})
        self.spent += time.perf_counter() - t0


class DetectVerifyDecide:
    """The deployable pipeline: detect -> verify uncertain candidates on crops -> decide.

    `pipe(image, on_event=cb)` streams progress events {"stage", "msg", "progress" in [0, 1], ...}
    with stage in detector | gate | crop | verifier | decision | done (used by the Streamlit app).
    """

    def __init__(self, detector, verifier, decision: RuleDecisionAgent, cfg: dict):
        # The three agents plus the config (crop size, margin, prompts).
        self.detector, self.verifier, self.decision, self.cfg = detector, verifier, decision, cfg

    def __call__(self, image, on_event=None) -> dict:
        """Run the full pipeline on one image and return the detection record plus decisions and timings."""
        v = self.cfg["verifier"]
        tl, th = self.decision.tau_low, self.decision.tau_high
        emit = _Emitter(on_event)

        # ---- 1. Detector Agent (progress 2 % -> 52 %, one event per text prompt) ----
        emit("detector", "Detector Agent: scanning the image for candidates", 0.02)
        det = self.detector.detect(image, on_progress=lambda k, n, found: emit(
            "detector", f"Detector prompt {k}/{n} done, {found} candidate(s) so far", 0.02 + 0.5 * k / n))
        scores = np.asarray(det["scores"], np.float32)
        # Defaults for candidates that are NOT verified: verdict False, P(yes) unknown (NaN).
        verdict = np.zeros(len(scores), bool)
        p_yes = np.full(len(scores), np.nan, np.float32)
        # ---- 2. Gate: only the uncertain band goes to the verifier ----
        need = self.decision.needs_verification(scores)
        idx = np.where(need)[0]
        n_hi, n_lo = int((scores >= th).sum()), int((scores < tl).sum())
        emit("gate", f"{len(scores)} candidates: {n_hi} auto-accepted (c >= {th:.2f}), "
                     f"{n_lo} auto-rejected (c < {tl:.2f}), {len(idx)} sent to the verifier", 0.57,
             n_verify=int(len(idx)))

        # Remember how much callback time had accumulated, to subtract only the part inside this block.
        s0 = emit.spent
        with Timer() as tv:
            if len(idx):
                labels = [det["labels"][k] for k in idx]
                views = []
                # ---- 3. Crop and zoom each uncertain candidate (progress 58 % -> 68 %) ----
                for j, k in enumerate(idx):
                    views += make_views(image, [det["boxes"][k]], self.cfg, "crop")
                    emit("crop", f"Crop and zoom {j + 1}/{len(idx)}: {labels[j]}", 0.58 + 0.10 * (j + 1) / len(idx))
                # ---- 4. Verifier Agent in batches (progress 68 % -> 95 %) ----
                bs = self.verifier.batch_size
                for s in range(0, len(idx), bs):
                    out = self.verifier.verify(views[s:s + bs], labels[s:s + bs], v["mode"], v["prompt_existence"],
                                               v["prompt_open"], v["open_max_new_tokens"])
                    # Write the answers back to the positions of the verified candidates.
                    verdict[idx[s:s + bs]] = out["verdict"]
                    p_yes[idx[s:s + bs]] = out["p_yes"]
                    done = min(s + bs, len(idx))
                    emit("verifier", f"Verifier Agent: {done}/{len(idx)} crops checked", 0.68 + 0.27 * done / len(idx),
                         results=[{"label": labels[j], "score": float(scores[idx[j]]), "p_yes": float(p),
                                   "verdict": bool(vd)}
                                  for j, (p, vd) in enumerate(zip(out["p_yes"], out["verdict"]), s)])
        # Pure model time: wall time of the block minus the time the UI callback took inside it.
        verify_s = tv.elapsed - (emit.spent - s0)

        # ---- 5. Decision Agent (progress 96 %) and finish (100 %) ----
        emit("decision", "Decision Agent: fusing detector confidence with the verifier verdicts", 0.96)
        with Timer() as td:
            accept = self.decision.decide(scores, verdict)
        emit("done", f"Done: {int(accept.sum())} of {len(scores)} candidates accepted", 1.0)
        return {**det, "verified": need.tolist(), "verdict": verdict.tolist(), "p_yes": p_yes.tolist(),
                "accept": accept.tolist(), "verify_time_s": verify_s, "decision_time_s": td.elapsed,
                "verify_invocations": int(len(idx)),
                "total_time_s": det["det_time_s"] + verify_s + td.elapsed}
