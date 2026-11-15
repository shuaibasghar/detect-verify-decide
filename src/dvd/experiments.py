"""Loading cached stage outputs as comparable "methods", sweeping them and costing them.

Every method is expressed over the SAME detector candidates of a split:
  kind="none" : detector only,                       accept = c >= tau
  kind="rule" : verifier verdict fused by the rule,  accept = (c >= tau_high) or (v and c >= tau_low)
  kind="gate" : verdict alone decides,               accept = (c >= tau) and v
so differences between methods come only from how candidates are verified and decided.
"""
from __future__ import annotations

import numpy as np

from . import paths
from .baselines.vcd import candidate_verdicts
from .cost import detector_cost, per_call_seconds
from .io_utils import JsonlCache
from .metrics import (DetectionEvaluator, rule_masks, select_at_recall, select_best, sweep_detector_only,
                      sweep_gate, sweep_rule)

PROPOSED = "DVD: crop verifier + rule (proposed)"
DET_ONLY = "Detector only"


def cached(path, ids) -> dict | None:
    """Records for all `ids`, or None if the cache is missing / incomplete."""
    if not path.exists():
        return None
    c = JsonlCache(path)
    return c.records(ids) if all(i in c for i in ids) else None


def require(path, ids) -> dict:
    recs = cached(path, ids)
    if recs is None:
        n = len(JsonlCache(path)) if path.exists() else 0
        raise RuntimeError(f"{path.name}: {n}/{len(ids)} images cached - run the notebook that produces it.")
    return recs


class SplitData:
    """Detector candidates + ground truth + evaluator for one split."""

    def __init__(self, cfg: dict, coco, split: str, ids: list[int], det_tag: str = "", image_fn=None):
        self.cfg, self.split, self.ids = cfg, split, ids
        self.det = require(paths.det_cache(cfg, split, det_tag), ids)
        self.gts = coco.gts(ids)
        self.ev = DetectionEvaluator(self.det, self.gts, cfg["coco"]["iou_threshold"])
        self.labels = self.ev.candidate_labels()


def available_methods(cfg: dict, data: SplitData) -> dict:
    """All methods whose caches are complete for this split."""
    s, ids = data.split, data.ids
    m = {DET_ONLY: {"kind": "none"}}
    crop = cached(paths.ver_cache(cfg, s, "crop"), ids)
    if crop:
        v = {i: np.asarray(r["verdict"], bool) for i, r in crop.items()}
        p = {i: np.asarray(r["p_yes"], np.float32) for i, r in crop.items()}
        t = per_call_seconds(crop, "verify_time_s", "verify_invocations")
        m[PROPOSED] = {"kind": "rule", "verdicts": v, "p_yes": p, "t_call": t}
        m["Ablation: crop verifier decides alone"] = {"kind": "gate", "verdicts": v, "p_yes": p, "t_call": t}
    full = cached(paths.ver_cache(cfg, s, "full"), ids)
    if full:
        m["Ablation: global-view verifier + rule"] = {
            "kind": "rule", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in full.items()},
            "t_call": per_call_seconds(full, "verify_time_s", "verify_invocations")}
    wp = cached(paths.woodpecker_cache(cfg, s), ids)
    if wp:
        m["Woodpecker-style single agent"] = {
            "kind": "gate", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in wp.items()},
            "fixed_time": float(np.mean([r["time_s"] for r in wp.values()])),
            "fixed_calls": float(np.mean([r["invocations"] for r in wp.values()]))}
    vcd = cached(paths.vcd_cache(cfg, s), ids)
    if vcd:
        ft = float(np.mean([r["time_s"] for r in vcd.values()]))
        fc = float(np.mean([r["invocations"] for r in vcd.values()]))
        m["VCD global yes/no"] = {"kind": "gate", "verdicts": candidate_verdicts(data.det, vcd, "p_vcd"),
                                  "fixed_time": ft, "fixed_calls": fc}
        m["Plain global yes/no (no contrast)"] = {"kind": "gate", "verdicts": candidate_verdicts(data.det, vcd, "p_plain"),
                                                 "fixed_time": ft / 2, "fixed_calls": fc / 2}
    return m


def sweep(name: str, method: dict, data: SplitData, cfg: dict) -> list[dict]:
    d = cfg["decision"]
    if method["kind"] == "none":
        rows = sweep_detector_only(data.ev, d["detector_only_grid"])
    elif method["kind"] == "rule":
        rows = sweep_rule(data.ev, method["verdicts"], d["tau_low_grid"], d["tau_high_grid"])
    else:
        rows = sweep_gate(data.ev, method["verdicts"], d["detector_only_grid"], name)
    for r in rows:
        r["method"] = name
    return rows


def masks_for(method: dict, data: SplitData, row: dict) -> dict:
    ev = data.ev
    if method["kind"] == "none":
        return {i: ev.scores(i) >= row["tau"] for i in ev.ids}
    if method["kind"] == "rule":
        return rule_masks(ev, method["verdicts"], row["tau_low"], row["tau_high"])
    return {i: (ev.scores(i) >= row["tau"]) & method["verdicts"][i] for i in ev.ids}


def operating_point(rows: list[dict], criterion: str = "best_f1", ref_recall: float | None = None) -> dict:
    """Select a configuration on held-out. For matched_recall, `recall_matched` says whether any
    configuration reached `ref_recall`; if none did, the best-F1 one is returned (flagged False)."""
    if criterion == "matched_recall":
        r = select_at_recall(rows, ref_recall)
        return {**r, "recall_matched": True} if r is not None else {**select_best(rows), "recall_matched": False}
    return select_best(rows)


def config_of(row: dict) -> dict:
    return {k: row[k] for k in ("tau", "tau_low", "tau_high") if k in row}


def evaluate_row(method: dict, data: SplitData, row: dict, n_boot: int = 0) -> dict:
    """Re-evaluate a (held-out-selected) configuration on `data` (usually the test split)."""
    from .metrics import bootstrap_ci
    masks = masks_for(method, data, row)
    out = data.ev.evaluate(masks, per_image=bool(n_boot))
    out.update(config_of(row))
    out["n_verified"] = _n_verified(method, data, row)
    if n_boot:
        out["ci"] = bootstrap_ci(out.pop("_per_image"), n_boot)
    out["_masks"] = masks
    return out


def _n_verified(method: dict, data: SplitData, row: dict) -> float:
    ev = data.ev
    if method["kind"] == "none":
        return 0.0
    if method["kind"] == "rule":
        return float(np.mean([((ev.scores(i) >= row["tau_low"]) & (ev.scores(i) < row["tau_high"])).sum()
                              for i in ev.ids]))
    return float(np.mean([(ev.scores(i) >= row["tau"]).sum() for i in ev.ids]))


def method_cost(method: dict, data: SplitData, n_verified: float) -> dict:
    """Per-image latency + invocations of a configuration (cost model, see docs/EVALUATION_PROTOCOL.md)."""
    d = detector_cost(data.det)
    if "t_call" in method:
        added, calls = n_verified * method["t_call"], n_verified
    elif "fixed_time" in method:
        added, calls = method["fixed_time"], method["fixed_calls"]
    else:
        added, calls = 0.0, 0.0
    return {"det_latency_s": d["det_latency_s"], "added_latency_s": added,
            "total_latency_s": d["det_latency_s"] + added, "det_invocations": d["det_invocations"],
            "verify_invocations": calls, "total_invocations": d["det_invocations"] + calls}
