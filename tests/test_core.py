"""CPU-only unit tests for the parts every result depends on (no GPU / model download needed).

Run:  python -m pytest tests -q      (or simply: python tests/test_core.py)
"""
# os: path handling for sys.path and the DVD_CONFIG environment variable.
import os
# sys: add src/ to the import path so `import dvd` works without installing the package.
import sys

# numpy: build small arrays for the toy detections and ground truth.
import numpy as np
# Pillow: tiny blank images stand in for real photos.
from PIL import Image

# Make the project's src/ folder importable from the tests folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Decision rule and gating function under test.
from dvd.agents.decision import needs_verification, rule_accept  # noqa: E402
# Phrase-to-class mapping of the detector.
from dvd.agents.detector import map_phrase  # noqa: E402
# Text helpers of the verifier.
from dvd.agents.verifier import article, label_matches  # noqa: E402
# Prompt builder and answer parser of the Woodpecker-style baseline.
from dvd.baselines.woodpecker_style import build_prompt, parse_answer  # noqa: E402
# Cropping geometry.
from dvd.cropping import crop_and_zoom, expand_box  # noqa: E402
# Metrics: matching, IoU, NMS, breakdown, sweeps, AUROC, POPE.
from dvd.metrics import (FP, IGNORE, TP, DetectionEvaluator, auroc, box_iou, decision_breakdown,  # noqa: E402
                         nms_per_class, pope_metrics, sweep_rule)


def _toy():
    """A tiny image with 3 ground-truth boxes (one is a crowd region) and 5 hand-made candidates."""
    gts = {1: {"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 90, 90]], np.float32),
               "category_ids": np.array([1, 1, 2]), "iscrowd": np.array([False, False, True])}}
    cands = {1: {"boxes": [[0, 0, 10, 10], [0, 0, 9, 10], [20, 20, 30, 30], [60, 60, 70, 70], [100, 100, 110, 110]],
                 "scores": [0.9, 0.8, 0.3, 0.5, 0.25], "category_ids": [1, 1, 1, 2, 3]}}
    return cands, gts


def test_iou():
    """Identical boxes give IoU 1; half-overlapping boxes give 50 / 150."""
    assert np.isclose(box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0], 1.0)
    assert np.isclose(box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0], 50 / 150)


def test_matching_and_rematch():
    """Greedy matching: duplicates, crowd regions and wrong classes; accepted subsets are re-matched."""
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
    """accept = (c >= tau_high) or (verdict and c >= tau_low); the gate selects the uncertain band."""
    c = np.array([0.9, 0.5, 0.5, 0.1])
    v = np.array([False, True, False, True])
    assert rule_accept(c, v, 0.3, 0.8).tolist() == [True, True, False, False]
    assert needs_verification(c, 0.3, 0.8).tolist() == [False, True, True, False]


def test_breakdown_and_sweep():
    """Decision breakdown counts and the (tau_low, tau_high) sweep skips impossible bands."""
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
    """Class-wise NMS: the duplicate of the same class is removed, the other class survives."""
    b = np.array([[0, 0, 10, 10], [1, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    keep = nms_per_class(b, np.array([0.9, 0.8, 0.7]), np.array([1, 1, 2]), 0.7)
    assert sorted(keep.tolist()) == [0, 2]


def test_crop():
    """Margin expansion, zoom to the target size and clipping at the image border."""
    img = Image.new("RGB", (640, 480))
    assert expand_box([100, 100, 200, 150], 0.5, 640, 480) == (50, 75, 250, 175)
    c = crop_and_zoom(img, [100, 100, 120, 110], 0.0, 448)
    assert max(c.size) == 448
    c = crop_and_zoom(img, [630, 470, 640, 480], 0.2, 448)  # clipped at border
    assert min(c.size) >= 28


def test_text_helpers():
    """Phrase mapping, articles, open-answer matching, Woodpecker prompt and parser."""
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
    """AUROC on a textbook example and POPE accuracy / yes-rate."""
    assert np.isclose(auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]), 0.75)
    m = pope_metrics(["yes", "yes", "no", "no"], ["yes", "no", "no", "yes"])
    assert m["accuracy"] == 0.5 and m["yes_rate"] == 0.5


def test_pipeline_events():
    """The online pipeline emits ordered progress events and gives the same answer with or without a callback."""
    from dvd.mock_agents import MockDetector, MockVerifier
    from dvd.pipeline import DetectVerifyDecide

    cfg = {"verifier": {"crop_margin": 0.15, "crop_target_size": 448, "min_crop_side": 28, "mode": "existence",
                        "prompt_existence": "Is there {article} {label} in the image?", "prompt_open": "",
                        "open_max_new_tokens": 5}}
    pipe = DetectVerifyDecide(MockDetector(), MockVerifier(), __import__("dvd.agents.decision", fromlist=["x"])
                              .RuleDecisionAgent(0.3, 0.7), cfg)
    img, events = Image.new("RGB", (320, 240)), []
    res = pipe(img, on_event=events.append)
    # Progress never goes backwards and ends at 100 % with a "done" event.
    prog = [e["progress"] for e in events]
    assert prog == sorted(prog) and prog[-1] == 1.0 and events[-1]["stage"] == "done"
    assert {e["stage"] for e in events} >= {"detector", "gate", "decision", "done"}
    c = np.array(res["scores"])
    # Only candidates inside the uncertain band are verified.
    band = (c >= 0.3) & (c < 0.7)
    assert res["verified"] == band.tolist() and res["verify_invocations"] == int(band.sum())
    assert (np.array(res["accept"]) == rule_accept(c, res["verdict"], 0.3, 0.7)).all()
    assert pipe(img)["accept"] == res["accept"]  # works without a callback, same answer


class _FakeVLM:
    """Stands in for VLMVerifier: answers 'yes' for dog, 'no' otherwise; records how it was called."""
    batch_size = 8

    def __init__(self):
        # What the pipeline asked: the labels of every verify call and the size of every generate call.
        self.verify_calls, self.generate_calls = [], []

    def verify(self, views, labels, mode="existence", *a, **k):
        """Record the labels and answer P(yes)=0.9 for 'dog', 0.1 for anything else."""
        self.verify_calls.append(list(labels))
        p = [0.9 if l == "dog" else 0.1 for l in labels]
        return {"p_yes": p, "verdict": [x > 0.5 for x in p], "answers": None}

    def generate(self, images, prompts, max_new_tokens=16):
        """Record the batch size and answer 'k: yes' for odd items, 'k: no' for even ones."""
        self.generate_calls.append(len(prompts))
        return ["\n".join(f"{k}: {'yes' if k % 2 else 'no'}" for k in range(1, p.count("\n") + 1)) for p in prompts]


# Minimal config with only the keys the stages under test read.
_CFG = {"verifier": {"crop_margin": 0.15, "crop_target_size": 336, "min_crop_side": 28, "mode": "existence",
                     "view": "crop", "full_image_max_side": 448, "prompt_existence": "Is there {article} {label}?",
                     "prompt_open": "", "open_max_new_tokens": 5},
        "woodpecker": {"max_candidates": 20, "batch_size": 3}}


def test_global_view_asks_each_label_once(tmp_path=None):
    """Global view: one question per DISTINCT label, mapped back to all candidates; crop view: one per candidate."""
    import tempfile
    from pathlib import Path

    from dvd.pipeline import run_verification_stage

    d = Path(tempfile.mkdtemp())
    det = {1: {"scores": [.9, .5, .4, .3], "labels": ["dog", "dog", "cat", "dog"], "boxes": [[0, 0, 9, 9]] * 4}}
    vlm = _FakeVLM()
    out = run_verification_stage(vlm, det, [1], d / "g.jsonl", _CFG, lambda i: Image.new("RGB", (64, 64)), view="full")
    assert vlm.verify_calls == [["cat", "dog"]]                      # 2 distinct labels, not 4 candidates
    assert out[1]["verdict"] == [True, True, False, True] and out[1]["verify_invocations"] == 4
    vlm2 = _FakeVLM()
    run_verification_stage(vlm2, det, [1], d / "c.jsonl", _CFG, lambda i: Image.new("RGB", (64, 64)), view="crop")
    assert vlm2.verify_calls == [["dog", "dog", "cat", "dog"]]       # crops: one per candidate


def test_woodpecker_batching_is_equivalent():
    """Batching several images per generation call must not change any verdict."""
    import tempfile
    from pathlib import Path

    from dvd.baselines.woodpecker_style import run_woodpecker_stage

    d = Path(tempfile.mkdtemp())
    # 7 images with 1, 2 or 3 candidates each.
    det = {i: {"scores": [.9, .5, .4][: 1 + i % 3], "labels": ["dog", "cat", "cup"][: 1 + i % 3],
               "boxes": [[0, 0, 9, 9]] * (1 + i % 3)} for i in range(7)}
    img = lambda i: Image.new("RGB", (64, 64))  # noqa: E731
    a = run_woodpecker_stage(_FakeVLM(), det, list(det), d / "a.jsonl", {**_CFG, "woodpecker": {"max_candidates": 20, "batch_size": 1}}, img)
    vlm = _FakeVLM()
    b = run_woodpecker_stage(vlm, det, list(det), d / "b.jsonl", _CFG, img)
    assert vlm.generate_calls == [3, 3, 1]                           # 7 images in chunks of 3
    assert all(a[i]["verdict"] == b[i]["verdict"] for i in det)


def test_paired_bootstrap():
    """A method with fewer FPs on the same images gets a positive, tight FP-reduction interval; identical methods give 0."""
    from dvd.metrics import paired_bootstrap

    n = 50
    base = {"tp": np.full(n, 3), "fp": np.full(n, 4), "n_gt": np.full(n, 4)}
    better = {"tp": np.full(n, 3), "fp": np.full(n, 1), "n_gt": np.full(n, 4)}
    r = paired_bootstrap(base, better, 200)
    assert np.isclose(r["fp_reduction"]["point"], 0.75) and r["fp_reduction"]["ci"][0] > 0.7
    assert r["d_recall"]["point"] == 0 and r["d_precision"]["point"] > 0 and r["d_f1"]["p_gt0"] == 1.0
    same = paired_bootstrap(base, base, 50)
    assert same["fp_reduction"]["point"] == 0 and same["d_f1"]["ci"] == (0.0, 0.0)


def test_main_comparison_on_synthetic_data():
    """End-to-end check of the results table on synthetic data where a perfect verifier must remove all FPs."""
    from dvd.experiments import DET_ONLY, PROPOSED, SplitData, main_comparison

    rng = np.random.default_rng(0)
    gts, cands, ver = {}, {}, {}
    for i in range(1, 41):
        gts[i] = {"boxes": np.array([[0, 0, 10, 10]], np.float32), "category_ids": np.array([1]),
                  "iscrowd": np.array([False])}
        tp_s = float(rng.choice([0.9, 0.35]))                       # sometimes the real object has a low score
        cands[i] = {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "scores": [tp_s, float(rng.choice([0.4, 0.3]))],
                    "category_ids": [1, 1], "det_time_s": 1.0, "det_invocations": 4}
        ver[i] = np.array([True, False])                            # a perfect verifier
    # Build a SplitData by hand (its normal constructor needs cache files and a COCO object).
    sd = object.__new__(SplitData)
    sd.cfg = {"decision": {"tau_low_grid": [0.25, 0.3], "tau_high_grid": [0.5, 0.8, 1.01],
                           "detector_only_grid": [0.25, 0.3, 0.5]}}
    sd.split, sd.ids, sd.det, sd.gts = "t", list(gts), cands, gts
    sd.ev = DetectionEvaluator(cands, gts, 0.5)
    sd.labels = sd.ev.candidate_labels()
    methods = {DET_ONLY: {"kind": "none"}, PROPOSED: {"kind": "rule", "verdicts": ver, "t_call": 0.1}}
    df, sweeps_h, sweeps_t, base = main_comparison(sd.cfg, sd, sd, methods, methods, n_boot=100)
    m = df[(df.criterion == "matched_recall") & (df.method == PROPOSED)].iloc[0]
    assert m.recall >= base["recall"] - 1e-9 and m.fp_reduction > 0.5 and m.fp_reduction_ci[0] > 0
    assert set(df.criterion) == {"best_f1", "matched_recall"} and len(df) == 4
    from dvd.experiments import paired_vs_proposed
    vs = paired_vs_proposed(sd.cfg, sd, methods, sweeps_h, base, "matched_recall", 100)
    assert list(vs.versus) == [DET_ONLY] and vs.iloc[0].fp_reduction > 0.5


def test_config_profiles():
    """Both profiles load, are internally consistent, and the fast one is really smaller."""
    import tempfile

    from dvd.config import load_config

    # Point all folders to a temp dir so the test never touches the project's data/results.
    tmp = tempfile.mkdtemp()
    ov = {"paths": {"data_root": tmp + "/d", "download_cache": tmp + "/c", "results_root": tmp + "/r"}}
    fast = load_config(overrides=ov, use_selected=False)
    os.environ["DVD_CONFIG"] = "full"
    try:
        full = load_config(overrides=ov, use_selected=False)
    finally:
        del os.environ["DVD_CONFIG"]
    assert fast["coco"]["test_size"] < full["coco"]["test_size"]
    # The candidate threshold must not exceed the smallest tau_low, or the sweep would use candidates that do not exist.
    assert fast["detector"]["box_threshold"] <= min(fast["decision"]["tau_low_grid"])
    assert full["detector"]["box_threshold"] <= min(full["decision"]["tau_low_grid"])
    # Woodpecker must be able to list every candidate the detector keeps.
    assert fast["detector"]["max_detections"] <= fast["woodpecker"]["max_candidates"]
    # The crop zoom size must be a whole number of 28-pixel VLM patches.
    assert fast["verifier"]["crop_target_size"] % fast["verifier"]["min_crop_side"] == 0


def test_kaggle_paths():
    """On Kaggle the dataset and the zips go to scratch space (not saved as notebook output)."""
    import tempfile

    import dvd.config as C

    tmp = tempfile.mkdtemp()
    old = (C.in_kaggle, C._kaggle_tmp)
    # Pretend to be on Kaggle, with a temp folder standing in for /kaggle/temp.
    C.in_kaggle, C._kaggle_tmp = (lambda: True), (lambda: tmp)
    try:
        cfg = C.load_config(overrides={"paths": {"results_root": tmp + "/r"}}, use_selected=False)
    finally:
        C.in_kaggle, C._kaggle_tmp = old
    assert cfg["paths"]["data_root"] == tmp + "/dvd_data"
    assert cfg["paths"]["download_cache"] == tmp + "/dvd_downloads"
    assert cfg["paths"]["results_root"] == tmp + "/r"


def test_download_retries_after_503():
    """HTTP 503 'Slow Down' (COCO's rate limit) is retried with a pause, and https is tried as the second route."""
    import io
    import tempfile
    import urllib.error
    import urllib.request
    from pathlib import Path

    import dvd.data.download as DL

    calls = []

    class Resp(io.BytesIO):
        headers = {"Content-Length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 503, "Slow Down", {}, None)
        return Resp(b"abc")

    old = (urllib.request.urlopen, DL.time.sleep)
    urllib.request.urlopen, DL.time.sleep = fake_urlopen, (lambda s: None)    # no real network, no real waiting
    try:
        dest = DL.download("http://example.org/y.bin", Path(tempfile.mkdtemp()) / "y.bin", quiet=True)
    finally:
        urllib.request.urlopen, DL.time.sleep = old
    assert dest.read_bytes() == b"abc" and len(calls) == 3 and calls[1].startswith("https://")


def test_coco_from_attached_kaggle_input():
    """On Kaggle, annotations and the needed images are taken from an attached COCO dataset instead of the network."""
    import tempfile
    from pathlib import Path

    import dvd.data.coco as DC

    tmp = Path(tempfile.mkdtemp())
    src = tmp / "input" / "coco-2017-dataset" / "coco2017"
    (src / "annotations").mkdir(parents=True)
    (src / "val2017").mkdir()
    (src / "annotations" / "instances_val2017.json").write_text("{}")
    (src / "val2017" / "000000000139.jpg").write_bytes(b"img")
    assert DC.find_in_kaggle_inputs("instances_val2017.json", root=str(tmp / "input")) == src / "annotations" / "instances_val2017.json"
    assert DC.find_in_kaggle_inputs("val2017", want_dir=True, root=str(tmp / "input")) == src / "val2017"
    assert DC.find_in_kaggle_inputs("nothing.json", root=str(tmp / "input")) is None

    cfg = {"coco": {"split": "val2017"}, "paths": {"data_root": str(tmp / "data"), "download_cache": str(tmp / "dl")}}
    old = (DC.in_kaggle, DC.KAGGLE_INPUT)
    DC.in_kaggle, DC.KAGGLE_INPUT = (lambda: True), str(tmp / "input")        # pretend to be on Kaggle
    try:
        root = DC.prepare_annotations(cfg)
        DC.fetch_images(cfg, [139])
    finally:
        DC.in_kaggle, DC.KAGGLE_INPUT = old
    assert (root / "annotations" / "instances_val2017.json").exists()
    assert (root / "val2017" / "000000000139.jpg").read_bytes() == b"img"


if __name__ == "__main__":
    # Allow `python tests/test_core.py` without pytest: run every test_* function and print its name.
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
