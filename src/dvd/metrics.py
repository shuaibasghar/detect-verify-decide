"""Detection-level and POPE metrics, decision breakdown, threshold sweeps and bootstrap CIs.

Matching protocol (see docs/EVALUATION_PROTOCOL.md):
  * detections of one image are processed in descending score order;
  * a detection is a TP if its IoU with an unmatched, non-crowd GT box of the same class is
    >= iou_thr (greedy, highest IoU first); each GT box can be matched once;
  * otherwise, if it overlaps a crowd GT region of the same class with IoA >= iou_thr it is
    ignored (neither TP nor FP), as in the COCO protocol;
  * otherwise it is a FP.
Accepted subsets are always re-matched, so a duplicate may become a TP once the original is rejected.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

TP, FP, IGNORE = 1, 0, -1


# ---------------------------------------------------------------- geometry
def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    inter = _inter(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def box_ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of `a` (used for crowd regions)."""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    return _inter(a, b) / np.maximum(box_area(a)[:, None], 1e-9)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-wise greedy NMS. Returns kept indices sorted by descending score."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        while idx.size:
            i = idx[0]
            keep.append(i)
            if idx.size == 1:
                break
            ious = box_iou(boxes[i:i + 1], boxes[idx[1:]])[0]
            idx = idx[1:][ious < iou_thr]
    keep = np.array(keep, dtype=np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")] if keep.size else keep


# ---------------------------------------------------------------- matching
class DetectionEvaluator:
    """Pre-computes per-image IoU matrices so threshold sweeps re-match quickly.

    cands: {image_id: {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "category_ids": [...]}}
    gts:   {image_id: {"boxes": (M,4), "category_ids": (M,), "iscrowd": (M,)}}
    """

    def __init__(self, cands: dict, gts: dict, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.ids = sorted(cands)
        self.data = {}
        for i in self.ids:
            c, g = cands[i], gts[i]
            boxes = np.asarray(c["boxes"], np.float32).reshape(-1, 4)
            gb = np.asarray(g["boxes"], np.float32).reshape(-1, 4)
            self.data[i] = {
                "scores": np.asarray(c["scores"], np.float32),
                "cats": np.asarray(c["category_ids"], np.int64),
                "boxes": boxes,
                "gt_cats": np.asarray(g["category_ids"], np.int64),
                "gt_crowd": np.asarray(g["iscrowd"], bool),
                "iou": box_iou(boxes, gb),
                "ioa": box_ioa(boxes, gb),
            }

    def scores(self, i) -> np.ndarray:
        return self.data[i]["scores"]

    def _match(self, i, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        d = self.data[i]
        n = d["scores"].shape[0]
        labels = np.full(n, IGNORE, dtype=np.int64)
        sel = np.arange(n) if mask is None else np.where(np.asarray(mask, bool))[0]
        order = sel[np.argsort(-d["scores"][sel], kind="stable")]
        used = np.zeros(d["gt_cats"].shape[0], bool)
        for k in order:
            same = d["gt_cats"] == d["cats"][k]
            cand = same & ~d["gt_crowd"] & ~used
            if cand.any():
                ious = np.where(cand, d["iou"][k], -1.0)
                j = int(ious.argmax())
                if ious[j] >= self.iou_thr:
                    labels[k] = TP
                    used[j] = True
                    continue
            crowd = same & d["gt_crowd"]
            if crowd.any() and d["ioa"][k][crowd].max() >= self.iou_thr:
                labels[k] = IGNORE
                continue
            labels[k] = FP
        return labels, int((~d["gt_crowd"]).sum())

    def candidate_labels(self) -> dict:
        """TP/FP/IGNORE label of every candidate, matched on the full candidate pool."""
        return {i: self._match(i)[0] for i in self.ids}

    def evaluate(self, masks: dict | Callable | None = None, per_image: bool = False) -> dict:
        """Evaluate the accepted subset. `masks`: {image_id: bool array} or f(image_id)->bool array."""
        tp = np.zeros(len(self.ids), np.int64)
        fp = np.zeros_like(tp)
        ngt = np.zeros_like(tp)
        for n, i in enumerate(self.ids):
            m = None if masks is None else (masks(i) if callable(masks) else masks[i])
            labels, g = self._match(i, m)
            tp[n], fp[n], ngt[n] = (labels == TP).sum(), (labels == FP).sum(), g
        out = summarize(tp.sum(), fp.sum(), ngt.sum(), len(self.ids))
        if per_image:
            out["_per_image"] = {"tp": tp, "fp": fp, "n_gt": ngt}
        return out


def summarize(tp: int, fp: int, n_gt: int, n_images: int) -> dict:
    tp, fp, n_gt = int(tp), int(fp), int(n_gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "n_gt": n_gt, "n_det": tp + fp, "precision": p, "recall": r, "f1": f1,
            "fdr": 1 - p if tp + fp else 0.0, "fp_per_image": fp / max(n_images, 1)}


def bootstrap_ci(per_image: dict, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Image-level bootstrap CIs for precision / recall / F1 / FP-per-image."""
    rng = np.random.default_rng(seed)
    tp, fp, ng = per_image["tp"], per_image["fp"], per_image["n_gt"]
    n = len(tp)
    stats = {k: [] for k in ("precision", "recall", "f1", "fp_per_image")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = summarize(tp[idx].sum(), fp[idx].sum(), ng[idx].sum(), n)
        for k in stats:
            stats[k].append(s[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in stats.items()}


# ---------------------------------------------------------------- decision-level breakdown
def decision_breakdown(labels: dict, scores: dict, accept: dict, tau_ref: float) -> dict:
    """Woodpecker-style correction breakdown at the decision level.

    Reference pool = candidates the single-pass detector would output at threshold `tau_ref`
    (usually tau_low). Within that pool:
      correct_rejection: FP removed          omission: FP kept
      over_rejection:    TP removed          correct_acceptance: TP kept
    """
    cr = om = orj = ca = 0
    for i in labels:
        lab, sc, acc = labels[i], np.asarray(scores[i]), np.asarray(accept[i], bool)
        pool = sc >= tau_ref
        fp, tp = pool & (lab == FP), pool & (lab == TP)
        cr += int((fp & ~acc).sum())
        om += int((fp & acc).sum())
        orj += int((tp & ~acc).sum())
        ca += int((tp & acc).sum())
    n_fp, n_tp = cr + om, orj + ca
    return {"correct_rejection": cr, "omission": om, "over_rejection": orj, "correct_acceptance": ca,
            "fp_removed_rate": cr / n_fp if n_fp else 0.0,
            "over_rejection_rate": orj / n_tp if n_tp else 0.0}


# ---------------------------------------------------------------- sweeps
def rule_masks(ev: DetectionEvaluator, verdicts: dict, tau_low: float, tau_high: float) -> dict:
    from .agents.decision import rule_accept
    return {i: rule_accept(ev.scores(i), verdicts[i], tau_low, tau_high) for i in ev.ids}


def sweep_detector_only(ev: DetectionEvaluator, thresholds) -> list[dict]:
    rows = []
    for t in thresholds:
        r = ev.evaluate({i: ev.scores(i) >= t for i in ev.ids})
        r.update(method="detector_only", tau=float(t), n_verified=0.0)
        rows.append(r)
    return rows


def sweep_rule(ev: DetectionEvaluator, verdicts: dict, tau_lows, tau_highs, method: str = "dvd") -> list[dict]:
    """Decision rule accept = (c >= tau_high) or (v and c >= tau_low) over a (tau_low, tau_high) grid."""
    rows = []
    for tl in tau_lows:
        for th in tau_highs:
            if th < tl:
                continue
            r = ev.evaluate(rule_masks(ev, verdicts, tl, th))
            band = [((ev.scores(i) >= tl) & (ev.scores(i) < th)).sum() for i in ev.ids]
            r.update(method=method, tau_low=float(tl), tau_high=float(th), n_verified=float(np.mean(band)))
            rows.append(r)
    return rows


def sweep_gate(ev: DetectionEvaluator, verdicts: dict, thresholds, method: str) -> list[dict]:
    """Verdict alone decides among candidates with c >= tau (no confidence fusion)."""
    rows = []
    for t in thresholds:
        masks = {i: (ev.scores(i) >= t) & np.asarray(verdicts[i], bool) for i in ev.ids}
        r = ev.evaluate(masks)
        r.update(method=method, tau=float(t),
                 n_verified=float(np.mean([(ev.scores(i) >= t).sum() for i in ev.ids])))
        rows.append(r)
    return rows


def select_best(rows: list[dict], key: str = "f1") -> dict:
    return max(rows, key=lambda r: (r[key], -r["n_verified"]))


def select_at_recall(rows: list[dict], min_recall: float) -> dict | None:
    """Fewest false positives among configurations whose recall >= min_recall."""
    ok = [r for r in rows if r["recall"] >= min_recall]
    return min(ok, key=lambda r: (r["fp"], r["n_verified"])) if ok else None


# ---------------------------------------------------------------- verifier diagnostics
def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Rank-based AUROC (ties averaged). positives: bool array."""
    scores, positives = np.asarray(scores, float), np.asarray(positives, bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pope_metrics(preds, labels) -> dict:
    """POPE accuracy / precision / recall / F1 / yes-rate. preds, labels: iterables of 'yes'/'no'."""
    p = np.array([str(x).lower().startswith("yes") for x in preds])
    y = np.array([str(x).lower().startswith("yes") for x in labels])
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    tn, fn = int((~p & ~y).sum()), int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(p), "accuracy": (tp + tn) / max(len(p), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "yes_rate": float(p.mean()) if len(p) else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
