"""Loading cached stage outputs as comparable "methods", sweeping them and costing them.

Every method is expressed over the SAME detector candidates of a split:
  kind="none" : detector only,                       accept = c >= tau
  kind="rule" : verifier verdict fused by the rule,  accept = (c >= tau_high) or (v and c >= tau_low)
  kind="gate" : verdict alone decides,               accept = (c >= tau) and v
so differences between methods come only from how candidates are verified and decided.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# numpy: verdict arrays and averages.
import numpy as np

# Cache-file name helpers (they encode backend, VLM, view, margin ...).
from . import paths
# Map per-label VCD answers back onto candidates.
from .baselines.vcd import candidate_verdicts
# Latency helpers of the cost model.
from .cost import detector_cost, per_call_seconds
# Resumable JSONL cache reader.
from .io_utils import JsonlCache
# Evaluator, decision masks, operating-point selection and threshold sweeps.
from .metrics import (DetectionEvaluator, rule_masks, select_at_recall, select_best, summarize, sweep_detector_only,
                      sweep_gate, sweep_rule)

# Display names of the main method and the baseline; used as dictionary keys and in tables.
PROPOSED = "DVD: crop verifier + rule (proposed)"
DET_ONLY = "Detector only"


def cached(path, ids) -> dict | None:
    """Records for all `ids`, or None if the cache is missing / incomplete."""
    if not path.exists():
        return None
    c = JsonlCache(path)
    return c.records(ids) if all(i in c for i in ids) else None


def require(path, ids) -> dict:
    """Like `cached`, but raises a helpful error saying which stage still has to be run."""
    recs = cached(path, ids)
    if recs is None:
        n = len(JsonlCache(path)) if path.exists() else 0
        raise RuntimeError(f"{path.name}: {n}/{len(ids)} images cached - run the notebook that produces it.")
    return recs


class SplitData:
    """Detector candidates + ground truth + evaluator for one split."""

    def __init__(self, cfg: dict, coco, split: str, ids: list[int], det_tag: str = "", image_fn=None):
        self.cfg, self.split, self.ids = cfg, split, ids
        # Cached detector output (must be complete for all `ids`).
        self.det = require(paths.det_cache(cfg, split, det_tag), ids)
        self.gts = coco.gts(ids)
        # Evaluator with precomputed IoU matrices, and the TP/FP/IGNORE label of every candidate.
        self.ev = DetectionEvaluator(self.det, self.gts, cfg["coco"]["iou_threshold"])
        self.labels = self.ev.candidate_labels()


def available_methods(cfg: dict, data: SplitData) -> dict:
    """All methods whose caches are complete for this split."""
    s, ids = data.split, data.ids
    # The detector-only baseline needs no cache beyond the detections.
    m = {DET_ONLY: {"kind": "none"}}
    crop = cached(paths.ver_cache(cfg, s, "crop"), ids)
    if crop:
        # Boolean verdicts and P(yes) per candidate, and the measured seconds per verifier call.
        v = {i: np.asarray(r["verdict"], bool) for i, r in crop.items()}
        p = {i: np.asarray(r["p_yes"], np.float32) for i, r in crop.items()}
        t = per_call_seconds(crop, "verify_time_s", "verify_invocations")
        m[PROPOSED] = {"kind": "rule", "verdicts": v, "p_yes": p, "t_call": t}
        # Same verdicts, different decision structure: the verifier alone decides (no confidence fusion).
        m["Ablation: crop verifier decides alone"] = {"kind": "gate", "verdicts": v, "p_yes": p, "t_call": t}
    full = cached(paths.ver_cache(cfg, s, "full"), ids)
    if full:
        # Ablation: same rule, but the verifier sees the whole image instead of the crop.
        m["Ablation: global-view verifier + rule"] = {
            "kind": "rule", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in full.items()},
            "t_call": per_call_seconds(full, "verify_time_s", "verify_invocations")}
    wp = cached(paths.woodpecker_cache(cfg, s), ids)
    if wp:
        # Baseline (ii): cost is a fixed per-image time (one generation call per image).
        m["Woodpecker-style single agent"] = {
            "kind": "gate", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in wp.items()},
            "fixed_time": float(np.mean([r["time_s"] for r in wp.values()])),
            "fixed_calls": float(np.mean([r["invocations"] for r in wp.values()]))}
    vcd = cached(paths.vcd_cache(cfg, s), ids)
    if vcd:
        ft = float(np.mean([r["time_s"] for r in vcd.values()]))
        fc = float(np.mean([r["invocations"] for r in vcd.values()]))
        # Baseline (iii): VCD answers per label, mapped back to candidates.
        m["VCD global yes/no"] = {"kind": "gate", "verdicts": candidate_verdicts(data.det, vcd, "p_vcd"),
                                  "fixed_time": ft, "fixed_calls": fc}
        # Same passes without the contrast: half the cost (only the clean-image pass is needed).
        m["Plain global yes/no (no contrast)"] = {"kind": "gate", "verdicts": candidate_verdicts(data.det, vcd, "p_plain"),
                                                 "fixed_time": ft / 2, "fixed_calls": fc / 2}
    return m


def sweep(name: str, method: dict, data: SplitData, cfg: dict) -> list[dict]:
    """Evaluate a method over its threshold grid; one row per configuration."""
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
    """Accept masks of one configuration (a row of a sweep) for every image of `data`."""
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
    """Only the threshold fields (tau / tau_low / tau_high) of a sweep row."""
    return {k: row[k] for k in ("tau", "tau_low", "tau_high") if k in row}


def evaluate_row(method: dict, data: SplitData, row: dict, n_boot: int = 0) -> dict:
    """Re-evaluate a (held-out-selected) configuration on `data` (usually the test split)."""
    from .metrics import bootstrap_ci
    masks = masks_for(method, data, row)
    out = data.ev.evaluate(masks, per_image=bool(n_boot))
    out.update(config_of(row))
    out["n_verified"] = _n_verified(method, data, row)
    if n_boot:
        # Replace the per-image counts by bootstrap confidence intervals.
        out["ci"] = bootstrap_ci(out.pop("_per_image"), n_boot)
    # Keep the masks: the decision breakdown and the paired bootstrap need them.
    out["_masks"] = masks
    return out


def per_image_counts(method: dict, data: SplitData, row: dict) -> dict:
    """Per-image tp / fp / n_gt arrays of a configuration (input of the bootstrap functions)."""
    return data.ev.evaluate(masks_for(method, data, row), per_image=True)["_per_image"]


def main_comparison(cfg: dict, H: SplitData, T: SplitData, MH: dict, MT: dict, n_boot: int = 1000):
    """The main results table: every method with complete caches, operating point selected on held-out
    (best-F1 and matched-recall), evaluated unchanged on test, with bootstrap CIs and a PAIRED bootstrap of every
    method against the detector-only baseline (FP reduction, recall / precision / F1 differences).
    Returns (table, sweeps_held_out, sweeps_test, detector_only_operating_point)."""
    # pandas only needed here, so the rest of the module stays importable without it.
    import pandas as pd
    from .metrics import decision_breakdown, paired_bootstrap

    # Methods available on BOTH splits.
    names = [n for n in MT if n in MH]
    # Held-out sweeps select the configuration; test sweeps draw the PR frontier figure.
    sweeps_h = {n: sweep(n, MH[n], H, cfg) for n in names}
    sweeps_t = {n: sweep(n, MT[n], T, cfg) for n in names}
    # Reference: the detector-only best-F1 point (its recall is the "matched recall" target).
    base_h = operating_point(sweeps_h[DET_ONLY])
    ref_recall, tau_ref = base_h["recall"], base_h["tau"]
    scores_t = {i: T.ev.scores(i) for i in T.ids}
    # Per-image counts of the baseline on test, used by every paired bootstrap below.
    base_pi = per_image_counts(MT[DET_ONLY], T, base_h)
    fp_base = int(base_pi["fp"].sum())
    rows = []
    for crit in ["best_f1", "matched_recall"]:
        for n in names:
            # Configuration chosen on held-out, applied unchanged to test.
            sel = operating_point(sweeps_h[n], crit, ref_recall)
            r = evaluate_row(MT[n], T, sel, n_boot=n_boot)
            masks = r.pop("_masks")
            b = decision_breakdown(T.labels, scores_t, masks, tau_ref)
            pb = paired_bootstrap(base_pi, T.ev.evaluate(masks, per_image=True)["_per_image"], n_boot)
            rows.append({
                "criterion": crit, "method": n, **{k: r.get(k) for k in ["tau", "tau_low", "tau_high"]},
                "recall_matched": sel.get("recall_matched", None),
                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "fp": r["fp"],
                "fp_per_image": r["fp_per_image"], "fp_reduction": 1 - r["fp"] / fp_base if fp_base else float("nan"),
                "f1_ci": r["ci"]["f1"], "precision_ci": r["ci"]["precision"], "recall_ci": r["ci"]["recall"],
                "fp_reduction_ci": pb["fp_reduction"]["ci"], "d_recall": pb["d_recall"]["point"],
                "d_recall_ci": pb["d_recall"]["ci"], "d_f1": pb["d_f1"]["point"], "d_f1_ci": pb["d_f1"]["ci"],
                "p_fp_reduction_gt0": pb["fp_reduction"]["p_gt0"], "p_f1_gt0": pb["d_f1"]["p_gt0"],
                "fp_removed_rate": b["fp_removed_rate"], "over_rejection_rate": b["over_rejection_rate"],
                **method_cost(MT[n], T, r["n_verified"])})
    return pd.DataFrame(rows), sweeps_h, sweeps_t, base_h


def paired_vs_proposed(cfg: dict, T: SplitData, MT: dict, sweeps_h: dict, base_h: dict, criterion: str = "best_f1",
                       n_boot: int = 1000):
    """Paired bootstrap of the PROPOSED pipeline against every other method (each at its own held-out-selected
    operating point). Positive d_f1 / fp_reduction = the proposed pipeline is better than that method."""
    import pandas as pd
    from .metrics import paired_bootstrap

    # Operating point of every method, chosen on held-out with the same criterion.
    sel = {n: operating_point(sweeps_h[n], criterion, base_h["recall"]) for n in sweeps_h if n in MT}
    # Per-image counts of every method on test.
    pi = {n: per_image_counts(MT[n], T, sel[n]) for n in sel}
    rows = []
    for n in sel:
        if n == PROPOSED:
            continue
        # base = the other method, method = the proposed pipeline -> positive numbers favour the pipeline.
        pb = paired_bootstrap(pi[n], pi[PROPOSED], n_boot)
        rows.append({"criterion": criterion, "versus": n, "fp_reduction": pb["fp_reduction"]["point"],
                     "fp_reduction_ci": pb["fp_reduction"]["ci"], "d_precision": pb["d_precision"]["point"],
                     "d_recall": pb["d_recall"]["point"], "d_recall_ci": pb["d_recall"]["ci"],
                     "d_f1": pb["d_f1"]["point"], "d_f1_ci": pb["d_f1"]["ci"], "p_f1_gt0": pb["d_f1"]["p_gt0"]})
    return pd.DataFrame(rows)


def extended_comparison(cfg: dict, T: SplitData, TX: SplitData, MT: dict, MTX: dict, n_boot: int = 1000,
                        save: bool = True):
    """Detector-only vs the proposed pipeline (crop verifier + rule) on the COMBINED test + test_ext images, and on
    test_ext alone (replication on images nobody has looked at). Only the operating points already selected on
    held-out (cfg["selected"]) are used - nothing is retuned. Both operating points are reported:
    best-F1 (tau_low, tau_high) and matched-recall (matched_tau_low, matched_tau_high); the detector-only point is
    detector_only_tau. Intervals are 1000-resample PAIRED image-level bootstraps (same resampled images for both).
    Saves tables/extended_comparison.csv/.md; returns the DataFrame."""
    import pandas as pd
    from .metrics import decision_breakdown, paired_bootstrap

    s = cfg.get("selected") or {}
    need = ["detector_only_tau", "tau_low", "tau_high", "matched_tau_low", "matched_tau_high"]
    if any(k not in s for k in need):
        raise RuntimeError(f"selected.json lacks {[k for k in need if k not in s]} - run the decision-agent step first.")
    det_row = {"tau": s["detector_only_tau"]}
    points = {"best_f1": {"tau_low": s["tau_low"], "tau_high": s["tau_high"]},
              "matched_recall": {"tau_low": s["matched_tau_low"], "tau_high": s["matched_tau_high"]}}
    cat = lambda parts: {k: np.concatenate([p[k] for p in parts]) for k in ("tp", "fp", "n_gt")}  # noqa: E731
    sets = {"test + test_ext": [(T, MT), (TX, MTX)], "test_ext only (replication)": [(TX, MTX)], "test only": [(T, MT)]}
    rows = []
    for set_name, parts in sets.items():
        # Per-image counts of the detector-only baseline (identical for both criteria), over all images of the set.
        base = cat([per_image_counts(M[DET_ONLY], D, det_row) for D, M in parts])
        n_img = len(base["tp"])
        labels = {i: D.labels[i] for D, _ in parts for i in D.ids}
        scores = {i: D.ev.scores(i) for D, _ in parts for i in D.ids}
        for crit, row in points.items():
            pro = cat([per_image_counts(M[PROPOSED], D, row) for D, M in parts])
            masks = {}
            for D, M in parts:
                masks.update(masks_for(M[PROPOSED], D, row))
            br = decision_breakdown(labels, scores, masks, det_row["tau"])
            pb = paired_bootstrap(base, pro, n_boot)
            for name, c in ((DET_ONLY, base), (PROPOSED, pro)):
                tot = summarize(c["tp"].sum(), c["fp"].sum(), c["n_gt"].sum(), n_img)
                is_p = name == PROPOSED
                rows.append({"set": set_name, "n_images": n_img, "criterion": crit, "method": name,
                             **({"tau_low": row["tau_low"], "tau_high": row["tau_high"]} if is_p else {"tau": det_row["tau"]}),
                             "precision": tot["precision"], "recall": tot["recall"], "f1": tot["f1"],
                             "fp_per_image": tot["fp_per_image"],
                             "fp_reduction": pb["fp_reduction"]["point"] if is_p else 0.0,
                             "fp_reduction_ci": pb["fp_reduction"]["ci"] if is_p else None,
                             "d_recall": pb["d_recall"]["point"] if is_p else 0.0,
                             "d_recall_ci": pb["d_recall"]["ci"] if is_p else None,
                             "d_f1": pb["d_f1"]["point"] if is_p else 0.0, "d_f1_ci": pb["d_f1"]["ci"] if is_p else None,
                             "over_rejection_rate": br["over_rejection_rate"] if is_p else 0.0})
    df = pd.DataFrame(rows)
    if save:
        from .config import results_dir
        tdir = results_dir(cfg, "tables")
        df.to_csv(tdir / "extended_comparison.csv", index=False)
        fmt = df.copy()
        for c in ["precision", "recall", "f1", "fp_per_image", "fp_reduction", "d_recall", "d_f1", "over_rejection_rate"]:
            fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
        for c in ["fp_reduction_ci", "d_recall_ci", "d_f1_ci"]:
            fmt[c] = fmt[c].map(lambda t: "" if t is None else f"[{t[0]:+.3f}, {t[1]:+.3f}]")
        (tdir / "extended_comparison.md").write_text(
            fmt[["set", "n_images", "criterion", "method", "precision", "recall", "f1", "fp_per_image", "fp_reduction",
                 "fp_reduction_ci", "d_recall_ci", "d_f1_ci", "over_rejection_rate"]].to_markdown(index=False),
            encoding="utf-8")
    return df


def _n_verified(method: dict, data: SplitData, row: dict) -> float:
    """Average number of candidates per image that this configuration sends to the verifier."""
    ev = data.ev
    if method["kind"] == "none":
        return 0.0
    if method["kind"] == "rule":
        # Only the uncertain band is verified.
        return float(np.mean([((ev.scores(i) >= row["tau_low"]) & (ev.scores(i) < row["tau_high"])).sum()
                              for i in ev.ids]))
    # Gate methods verify every candidate above tau.
    return float(np.mean([(ev.scores(i) >= row["tau"]).sum() for i in ev.ids]))


def method_cost(method: dict, data: SplitData, n_verified: float) -> dict:
    """Per-image latency + invocations of a configuration (cost model, see docs/EVALUATION_PROTOCOL.md)."""
    d = detector_cost(data.det)
    if "t_call" in method:
        # Per-candidate verifier: added latency = calls x measured seconds per call.
        added, calls = n_verified * method["t_call"], n_verified
    elif "fixed_time" in method:
        # Baselines with a fixed per-image cost (Woodpecker-style, VCD).
        added, calls = method["fixed_time"], method["fixed_calls"]
    else:
        added, calls = 0.0, 0.0
    return {"det_latency_s": d["det_latency_s"], "added_latency_s": added,
            "total_latency_s": d["det_latency_s"] + added, "det_invocations": d["det_invocations"],
            "verify_invocations": calls, "total_invocations": d["det_invocations"] + calls}
