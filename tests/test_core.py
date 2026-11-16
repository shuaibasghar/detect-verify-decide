"""CPU-only unit tests for the parts every result depends on (no GPU / model download needed).

Run:  python -m pytest tests -q      (or simply: python tests/test_core.py)
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvd.agents.decision import needs_verification, rule_accept  # noqa: E402
from dvd.agents.detector import map_phrase  # noqa: E402
from dvd.agents.verifier import article, label_matches  # noqa: E402
from dvd.baselines.woodpecker_style import build_prompt, parse_answer  # noqa: E402
from dvd.cropping import crop_and_zoom, expand_box  # noqa: E402
from dvd.metrics import (FP, IGNORE, TP, DetectionEvaluator, auroc, box_iou, decision_breakdown,  # noqa: E402
                         nms_per_class, pope_metrics, sweep_rule)


def _toy():
    gts = {1: {"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 90, 90]], np.float32),
               "category_ids": np.array([1, 1, 2]), "iscrowd": np.array([False, False, True])}}
    cands = {1: {"boxes": [[0, 0, 10, 10], [0, 0, 9, 10], [20, 20, 30, 30], [60, 60, 70, 70], [100, 100, 110, 110]],
                 "scores": [0.9, 0.8, 0.3, 0.5, 0.25], "category_ids": [1, 1, 1, 2, 3]}}
    return cands, gts


def test_iou():
    assert np.isclose(box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0], 1.0)
    assert np.isclose(box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0], 50 / 150)


def test_matching_and_rematch():
    cands, gts = _toy()
    ev = DetectionEvaluator(cands, gts, 0.5)
    lab = ev.candidate_labels()[1]
    # duplicate -> FP, crowd overlap -> IGNORE, wrong class -> FP
    assert lab.tolist() == [TP, FP, TP, IGNORE, FP]
    full = ev.evaluate()
    assert (full["tp"], full["fp"], full["n_gt"]) == (2, 2, 2)
    # rejecting the original lets the duplicate match on re-evaluation
    r = ev.evaluate({1: np.array([False, True, True, True, True])})
    assert r["tp"] == 2 and r["fp"] == 1


def test_decision_rule():
    c = np.array([0.9, 0.5, 0.5, 0.1])
    v = np.array([False, True, False, True])
    assert rule_accept(c, v, 0.3, 0.8).tolist() == [True, True, False, False]
    assert needs_verification(c, 0.3, 0.8).tolist() == [False, True, True, False]


def test_breakdown_and_sweep():
    cands, gts = _toy()
    ev = DetectionEvaluator(cands, gts, 0.5)
    labels = ev.candidate_labels()
    verdicts = {1: np.array([True, False, True, True, False])}
    acc = {1: rule_accept(ev.scores(1), verdicts[1], 0.2, 0.95)}
    b = decision_breakdown(labels, {1: ev.scores(1)}, acc, 0.2)
    assert b["correct_rejection"] == 2 and b["over_rejection"] == 0 and b["omission"] == 0
    rows = sweep_rule(ev, verdicts, [0.2, 0.3], [0.5, 1.01])
    assert all(r["tau_high"] >= r["tau_low"] for r in rows) and len(rows) == 4


def test_nms():
    b = np.array([[0, 0, 10, 10], [1, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    keep = nms_per_class(b, np.array([0.9, 0.8, 0.7]), np.array([1, 1, 2]), 0.7)
    assert sorted(keep.tolist()) == [0, 2]


def test_crop():
    img = Image.new("RGB", (640, 480))
    assert expand_box([100, 100, 200, 150], 0.5, 640, 480) == (50, 75, 250, 175)
    c = crop_and_zoom(img, [100, 100, 120, 110], 0.0, 448)
    assert max(c.size) == 448
    c = crop_and_zoom(img, [630, 470, 640, 480], 0.2, 448)  # clipped at border
    assert min(c.size) >= 28


def test_text_helpers():
    names = ["dining table", "table", "cell phone", "tv"]
    assert map_phrase("dining table", names) == 0
    assert map_phrase("cell", names) == 2
    assert map_phrase("", names) is None
    assert article("apple") == "an" and article("dog") == "a"
    assert label_matches("A man.", "person") and not label_matches("carrot", "car")
    assert label_matches("table", "dining table")
    assert parse_answer("1: yes\n2: No\n4: yes", 3) == [True, False, None]
    assert "[0, 0, 500, 500]" in build_prompt(["dog"], [[0, 0, 50, 100]], 100, 200)


def test_metrics_misc():
    assert np.isclose(auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]), 0.75)
    m = pope_metrics(["yes", "yes", "no", "no"], ["yes", "no", "no", "yes"])
    assert m["accuracy"] == 0.5 and m["yes_rate"] == 0.5


def test_pipeline_events():
    from dvd.mock_agents import MockDetector, MockVerifier
    from dvd.pipeline import DetectVerifyDecide

    cfg = {"verifier": {"crop_margin": 0.15, "crop_target_size": 448, "min_crop_side": 28, "mode": "existence",
                        "prompt_existence": "Is there {article} {label} in the image?", "prompt_open": "",
                        "open_max_new_tokens": 5}}
    pipe = DetectVerifyDecide(MockDetector(), MockVerifier(), __import__("dvd.agents.decision", fromlist=["x"])
                              .RuleDecisionAgent(0.3, 0.7), cfg)
    img, events = Image.new("RGB", (320, 240)), []
    res = pipe(img, on_event=events.append)
    prog = [e["progress"] for e in events]
    assert prog == sorted(prog) and prog[-1] == 1.0 and events[-1]["stage"] == "done"
    assert {e["stage"] for e in events} >= {"detector", "gate", "decision", "done"}
    c = np.array(res["scores"])
    band = (c >= 0.3) & (c < 0.7)
    assert res["verified"] == band.tolist() and res["verify_invocations"] == int(band.sum())
    assert (np.array(res["accept"]) == rule_accept(c, res["verdict"], 0.3, 0.7)).all()
    assert pipe(img)["accept"] == res["accept"]  # works without a callback, same answer


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
