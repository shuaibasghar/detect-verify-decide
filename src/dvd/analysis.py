"""Per-candidate tables for error analysis (category, size, confidence, co-occurrence, brightness)."""
from __future__ import annotations

import numpy as np

from .metrics import FP, TP, box_area

SIZE_BINS = [(0, 32 ** 2, "small"), (32 ** 2, 96 ** 2, "medium"), (96 ** 2, float("inf"), "large")]


def size_bucket(area: float) -> str:
    for lo, hi, name in SIZE_BINS:
        if lo <= area < hi:
            return name
    return "large"


def cooccurrence_topk(coco, k: int = 10) -> dict:
    """For each category id: the set of k category ids it most often co-occurs with (image level)."""
    cat_ids = coco.cat_ids
    pos = {c: n for n, c in enumerate(cat_ids)}
    m = np.zeros((len(cat_ids), len(cat_ids)), np.int64)
    for iid in coco.image_ids:
        present = {a["category_id"] for a in coco.coco.loadAnns(coco.coco.getAnnIds(imgIds=iid))}
        idx = [pos[c] for c in present]
        for a in idx:
            for b in idx:
                if a != b:
                    m[a, b] += 1
    return {c: {cat_ids[j] for j in np.argsort(-m[pos[c]])[:k]} for c in cat_ids}


def candidate_table(det_records: dict, cand_labels: dict, accept: dict, gts: dict,
                    verdicts: dict | None = None, p_yes: dict | None = None, cooc: dict | None = None,
                    brightness: dict | None = None) -> list[dict]:
    """One row per candidate. outcome in {correct_acceptance, over_rejection, omission, correct_rejection, ignore}."""
    rows = []
    for iid, det in det_records.items():
        boxes = np.asarray(det["boxes"], np.float32).reshape(-1, 4)
        areas = box_area(boxes)
        present = set(np.asarray(gts[iid]["category_ids"]).tolist())
        for k, (lab, sc, cat) in enumerate(zip(det["labels"], det["scores"], det["category_ids"])):
            y, acc = int(cand_labels[iid][k]), bool(accept[iid][k])
            outcome = {(TP, True): "correct_acceptance", (TP, False): "over_rejection",
                       (FP, True): "omission", (FP, False): "correct_rejection"}.get((y, acc), "ignore")
            row = {"image_id": iid, "det_idx": k, "box": [float(x) for x in boxes[k]], "label": lab, "category_id": cat, "score": float(sc), "is_tp": y == TP,
                   "is_fp": y == FP, "accepted": acc, "outcome": outcome, "area": float(areas[k]),
                   "size": size_bucket(float(areas[k])), "class_in_image": cat in present}
            if verdicts is not None:
                row["verdict"] = bool(verdicts[iid][k])
            if p_yes is not None:
                row["p_yes"] = float(p_yes[iid][k])
            if cooc is not None:
                # hallucination-style FP: class absent from the image but a top co-occurring partner is present
                row["cooccurrence_fp"] = (y == FP) and cat not in present and bool(cooc[cat] & present)
            if brightness is not None:
                row["brightness"] = brightness[iid]
            rows.append(row)
    return rows


def mean_brightness(img) -> float:
    return float(np.asarray(img.convert("L"), np.float32).mean() / 255.0)
