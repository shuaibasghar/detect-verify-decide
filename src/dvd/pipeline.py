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
from __future__ import annotations

from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from .agents.decision import RuleDecisionAgent
from .agents.verifier import make_views
from .cost import Timer
from .io_utils import JsonlCache


def run_detection_stage(detector, image_ids, cache_path, image_fn: Callable, desc: str = "detect") -> dict:
    """Run the Detector Agent over `image_ids` (resumable). image_fn(image_id) -> PIL image."""
    cache = JsonlCache(cache_path)
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        rec = detector.detect(image_fn(iid))
        rec["image_id"] = int(iid)
        cache.add(rec)
    return cache.records(image_ids)


def run_verification_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn: Callable,
                           view: str | None = None, margin: float | None = None, mode: str | None = None,
                           desc: str = "verify") -> dict:
    """Run the Verifier Agent on every cached candidate (resumable)."""
    v = cfg["verifier"]
    view, mode = view or v["view"], mode or v["mode"]
    margin = v["crop_margin"] if margin is None else margin
    cache = JsonlCache(cache_path)
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        det = det_records[iid]
        n = len(det["scores"])
        rec = {"image_id": int(iid), "view": view, "margin": margin, "mode": mode,
               "p_yes": [], "verdict": [], "answers": None, "verify_time_s": 0.0, "verify_invocations": n}
        if n:
            image = image_fn(iid)
            with Timer() as t:
                views = make_views(image, det["boxes"], cfg, view, margin)
                out = verifier.verify(views, det["labels"], mode, v["prompt_existence"], v["prompt_open"],
                                      v["open_max_new_tokens"])
            rec.update(out, verify_time_s=t.elapsed)
        cache.add(rec)
    return cache.records(image_ids)


def verdicts_from(ver_records: dict, key: str = "verdict") -> dict:
    return {i: np.asarray(r[key], dtype=bool if key == "verdict" else np.float32) for i, r in ver_records.items()}


class DetectVerifyDecide:
    """The deployable pipeline: detect -> verify uncertain candidates on crops -> decide."""

    def __init__(self, detector, verifier, decision: RuleDecisionAgent, cfg: dict):
        self.detector, self.verifier, self.decision, self.cfg = detector, verifier, decision, cfg

    def __call__(self, image) -> dict:
        v = self.cfg["verifier"]
        det = self.detector.detect(image)
        scores = np.asarray(det["scores"], np.float32)
        verdict = np.zeros(len(scores), bool)
        p_yes = np.full(len(scores), np.nan, np.float32)
        need = self.decision.needs_verification(scores)
        idx = np.where(need)[0]
        with Timer() as tv:
            if len(idx):
                views = make_views(image, [det["boxes"][k] for k in idx], self.cfg, "crop")
                out = self.verifier.verify(views, [det["labels"][k] for k in idx], v["mode"],
                                           v["prompt_existence"], v["prompt_open"], v["open_max_new_tokens"])
                verdict[idx] = out["verdict"]
                p_yes[idx] = out["p_yes"]
        with Timer() as td:
            accept = self.decision.decide(scores, verdict)
        return {**det, "verified": need.tolist(), "verdict": verdict.tolist(), "p_yes": p_yes.tolist(),
                "accept": accept.tolist(), "verify_time_s": tv.elapsed, "decision_time_s": td.elapsed,
                "verify_invocations": int(len(idx)),
                "total_time_s": det["det_time_s"] + tv.elapsed + td.elapsed}
