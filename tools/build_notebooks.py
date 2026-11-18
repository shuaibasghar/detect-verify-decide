"""Generates notebooks/*.ipynb for the Detect-Verify-Decide project (nbformat 4, no dependency)."""
import json  # write the .ipynb files (they are JSON)
import sys  # command-line arguments of this script
import textwrap  # strip the indentation of the cell source strings
from pathlib import Path  # portable file paths

OUT = Path(sys.argv[1])
ONLY = set(sys.argv[2:])  # optional: regenerate only these notebooks, e.g. 12_streamlit_demo
OUT.mkdir(parents=True, exist_ok=True)


def md(s):
    """Build a Markdown cell from text (indentation stripped, split into lines as nbformat expects)."""
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def code(s):
    """Build a code cell from source text (no outputs, never executed)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def notebook(cells, gpu=True):
    """Wrap cells into a complete notebook dict (nbformat 4); `gpu` asks Colab for a T4 accelerator by default."""
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


BOOT = code(r'''
# ===================== Bootstrap (identical in every notebook) =====================
import os, sys, subprocess, shutil  # paths, sys.path, shell commands (git clone, pip), folder copy
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = (not IN_COLAB) and os.path.isdir("/kaggle/working")    # Kaggle notebook (needs Internet ON, see docs/KAGGLE_GUIDE.md)

# Where the project lives on Google Drive. Upload the whole repo folder there (see docs/COLAB_GUIDE.md),
# or set REPO_URL and it will be cloned into DRIVE_PROJECT on first run.
DRIVE_PROJECT = "/content/drive/MyDrive/detect-verify-decide"
REPO_URL = ""  # e.g. "https://github.com/<you>/detect-verify-decide.git"
BRANCH = "kaggle-version"  # git branch to take the latest code from (change it if you push to another branch)

if IN_COLAB:
    from google.colab import drive  # Colab only: mount Google Drive
    drive.mount("/content/drive")
    if not os.path.isdir(DRIVE_PROJECT) and REPO_URL:
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, DRIVE_PROJECT], check=True)
    elif os.path.isdir(os.path.join(DRIVE_PROJECT, ".git")):   # pick up the latest pushed code (never touches your notebooks/results)
        subprocess.run(["git", "-C", DRIVE_PROJECT, "fetch", "origin", BRANCH], check=False)
        subprocess.run(["git", "-C", DRIVE_PROJECT, "checkout", "FETCH_HEAD", "--", "src", "configs", "requirements.txt", "tools", "app"], check=False)
    PROJECT_ROOT = DRIVE_PROJECT
elif IN_KAGGLE:
    PROJECT_ROOT = "/kaggle/working/detect-verify-decide"
    if not os.path.isdir(os.path.join(PROJECT_ROOT, "src", "dvd")):
        import glob
        attached = glob.glob("/kaggle/input/*/src/dvd") + glob.glob("/kaggle/input/*/*/src/dvd")   # project attached as a Kaggle Dataset
        if attached:
            shutil.copytree(os.path.dirname(os.path.dirname(attached[0])), PROJECT_ROOT)
        elif REPO_URL:
            url = REPO_URL
            try:   # private repo: add a Kaggle secret named GITHUB_TOKEN (Add-ons -> Secrets)
                from kaggle_secrets import UserSecretsClient
                url = REPO_URL.replace("https://", f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
            except Exception:
                pass
            r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, PROJECT_ROOT], capture_output=True)
            if r.returncode:   # (the message deliberately does not echo the command, which could contain the token)
                raise RuntimeError("git clone failed: is Internet ON, REPO_URL / BRANCH right, and for a private repo the GITHUB_TOKEN secret set?")
            subprocess.run(["git", "-C", PROJECT_ROOT, "remote", "set-url", "origin", REPO_URL])   # never keep the token in .git/config
        else:
            raise RuntimeError("Set REPO_URL (and BRANCH) above, or attach the project as a Kaggle Dataset.")
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

assert os.path.isdir(os.path.join(PROJECT_ROOT, "src", "dvd")), f"src/dvd not found under {PROJECT_ROOT}"
os.environ["DVD_ROOT"] = PROJECT_ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
print("PROJECT_ROOT =", PROJECT_ROOT)
''')

INSTALL = code(r'''
# Install dependencies (Colab / Kaggle already have torch/torchvision with CUDA). ~1-2 min per fresh runtime.
if IN_COLAB or IN_KAGGLE:
    %pip install -q -r "{PROJECT_ROOT}/requirements.txt"
''')

CFG = code(r'''
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt  # json files, numpy arrays, pandas tables, matplotlib plots
from dvd.config import load_config, save_selected, results_dir  # load configs/default.yaml, save tuned values, results folders
from dvd.io_utils import set_seed, save_json, load_json, JsonlCache  # seeding, JSON helpers and the resumable JSONL cache
from dvd import paths  # cache-file path helpers

cfg = load_config()
set_seed(cfg["seed"])
pd.set_option("display.precision", 4)
try:
    import torch  # PyTorch: GPU check and memory statistics
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only - expect very slow runs)")
except ImportError:
    print("torch not installed")
print("data_root   :", cfg["paths"]["data_root"])
print("results_root:", cfg["paths"]["results_root"])
print("tuned params:", cfg["selected"] or "(none yet)")
''')

COCO = code(r'''
from dvd.data.coco import prepare_annotations, fetch_images, CocoData, load_splits  # COCO annotations, image download, splits
prepare_annotations(cfg)   # annotation json only (the 241 MB zip is cached on Drive)
coco = CocoData(cfg)
splits = load_splits(cfg)
HELDOUT, TEST = splits["heldout"], splits["test"]
fetch_images(cfg, HELDOUT + TEST)   # only these images (a few hundred files), not the whole 778 MB val2017 zip
print(f"held-out: {len(HELDOUT)} images | test: {len(TEST)} images")
''')


def header(title, purpose, inputs, outputs, runtime, objective):
    """First Markdown cell of a notebook: title, purpose and a table of objective / inputs / outputs / runtime."""
    return md(f"""
# {title}

**Purpose.** {purpose}

| | |
|---|---|
| **Proposal objective** | {objective} |
| **Inputs** | {inputs} |
| **Outputs** | {outputs} |
| **Runtime (T4)** | {runtime} |

> Run cells top to bottom. Every expensive stage is **resumable**: if Colab disconnects, reconnect,
> re-run from the top and it continues where it stopped (results are cached on Drive under `results/`).
""")


NB = {}

# ------------------------------------------------------------------------------------------------ 00
NB["00_setup_and_smoke_test"] = [
    header("00 · Setup & smoke test",
           "Verify the Colab environment, install dependencies, and run the three agents end-to-end on one image "
           "before committing GPU hours. Nothing here is used as a result.",
           "one COCO image downloaded from the web", "none (printouts + figures only)", "~10 min (model downloads)",
           "Prerequisite for all objectives"),
    md("## 1. Environment"),
    BOOT, INSTALL, CFG,
    code(r'''
!nvidia-smi
import transformers, PIL  # version printout of the two core libraries
print("transformers", transformers.__version__, "| torch", torch.__version__, "| PIL", PIL.__version__)
'''),
    md("## 2. A test image\nCOCO `000000039769` (two cats on a couch) is fetched directly; no dataset download needed yet."),
    code(r'''
import io, urllib.request  # download the test image straight into memory
from PIL import Image  # Pillow image type
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
plt.imshow(image); plt.axis("off"); plt.show()
'''),
    md("## 3. Detector Agent\nA small vocabulary is used here; the real runs use all 80 COCO classes."),
    code(r'''
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.viz import show_detections  # plotting helpers (detections overlay, PR frontier, crop gallery)
vocab = ["cat", "dog", "couch", "remote", "bed", "person", "tv"]
detector = build_detector(cfg, vocab, {n: k for k, n in enumerate(vocab)})
det = detector.detect(image)
print(f"{len(det['scores'])} candidates in {det['det_time_s']:.2f}s using {det['det_invocations']} prompt(s)")
show_detections(image, det["boxes"], det["labels"], det["scores"], title="Detector candidates"); plt.show()
'''),
    md("## 4. Verifier Agent (crop-and-zoom, label-blind to confidence)"),
    code(r'''
from dvd.agents.verifier import build_verifier, make_views  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
verifier = build_verifier(cfg)
views = make_views(image, det["boxes"], cfg, "crop")
out = verifier.verify(views, det["labels"])
pd.DataFrame({"label": det["labels"], "det_conf": det["scores"], "verifier_p_yes": out["p_yes"],
              "verdict": out["verdict"]})
'''),
    code(r'''
fig, axes = plt.subplots(1, len(views), figsize=(3 * len(views), 3), squeeze=False)
for ax, v, l, p in zip(axes[0], views, det["labels"], out["p_yes"]):
    ax.imshow(v); ax.set_title(f"{l}  P(yes)={p:.2f}", fontsize=9); ax.axis("off")
plt.show()
'''),
    md("## 5. Hallucination probe\nAsk about objects that are **not** in the image. A well-behaved verifier should give low P(yes)."),
    code(r'''
from dvd.cropping import fit_max_side  # crop-and-zoom / image resizing helpers
full = fit_max_side(image, cfg["verifier"]["full_image_max_side"])
qs = [f"Is there a {o} in the image? Answer yes or no." for o in ["cat", "remote", "dog", "car", "pizza"]]
for q, p in zip(qs, verifier.yes_no([full] * len(qs), qs)):
    print(f"{p:.3f}  {q}")
'''),
    md("## 6. Full online pipeline (Detect -> Verify uncertain band -> Decide)"),
    code(r'''
from dvd.agents.decision import RuleDecisionAgent  # Decision Agent: the accept/reject rule and the optional LLM aggregator
from dvd.pipeline import DetectVerifyDecide  # cached offline stages and the online Detect-Verify-Decide pipeline
pipe = DetectVerifyDecide(detector, verifier, RuleDecisionAgent(tau_low=0.25, tau_high=0.6), cfg)
res = pipe(image)
status = ["accepted" if a else "rejected" for a in res["accept"]]
print(f"detector {res['det_time_s']:.2f}s | verifier {res['verify_time_s']:.2f}s on {res['verify_invocations']} crops | total {res['total_time_s']:.2f}s")
show_detections(image, res["boxes"], res["labels"], res["scores"], status, title="green = accepted, red dashed = rejected"); plt.show()
if torch.cuda.is_available():
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
'''),
    md("""
## ✅ Smoke-test checklist
- [ ] GPU detected (T4 or better)
- [ ] Detector finds the two cats / couch / remotes
- [ ] Verifier P(yes) is high for true crops and low for absent objects in the probe
- [ ] Peak GPU memory comfortably below the card limit (both models fit together)

If all pass, continue with **01_data_preparation**.
"""),
]

# ------------------------------------------------------------------------------------------------ 01
NB["01_data_preparation"] = [
    header("01 · Data preparation",
           "Download COCO val2017, create the seeded held-out (tuning) / test split, inspect POPE, and pre-compute "
           "category co-occurrence statistics for the error analysis.",
           "internet", "`results_fast/splits.json`, `results_fast/data/*.json|csv`", "≈3 min (annotation zip 241 MB + a few hundred images)",
           "Objective 2 (datasets for evaluation)"),
    BOOT, INSTALL, CFG,
    md("## 1. COCO val2017\nThe annotation zip is cached in `data/downloads/` on Drive (persistent). Only the images of the splits are downloaded, "
       "to the fast local disk (set `DVD_CONFIG=full` for the original 1000-image protocol)."),
    code(r'''
from dvd.data.coco import prepare_annotations, fetch_images, CocoData, make_splits  # COCO annotations, image download, splits
prepare_annotations(cfg)
coco = CocoData(cfg)
n_inst = len(coco.coco.getAnnIds(iscrowd=False))
print(f"{len(coco.image_ids)} images, {n_inst} non-crowd instances, {len(coco.cat_ids)} categories")
'''),
    code(r'''
cat_counts = pd.Series({coco.cat_id_to_name[c]: len(coco.coco.getAnnIds(catIds=[c], iscrowd=False)) for c in coco.cat_ids})
cat_counts.sort_values(ascending=False).plot.bar(figsize=(16, 4), title="Instances per category (val2017)"); plt.show()
results_dir(cfg, "data")
cat_counts.sort_values(ascending=False).to_csv(results_dir(cfg, "data") / "category_counts.csv", header=["instances"])
'''),
    md("## 2. Held-out / test split\nThe held-out split is used **only** for choosing the crop margin and the decision thresholds. "
       "All reported numbers come from the test split. The split is created once and frozen."),
    code(r'''
splits = make_splits(cfg, coco.image_ids)
HELDOUT, TEST = splits["heldout"], splits["test"]
assert not set(HELDOUT) & set(TEST)
fetch_images(cfg, HELDOUT + TEST)
def split_stats(ids):
    # Summary of a split: number of images, instances, instances per image and images without objects.
    g = coco.gts(ids)
    n = [int((~v["iscrowd"]).sum()) for v in g.values()]
    return {"images": len(ids), "instances": int(np.sum(n)), "inst/img": float(np.mean(n)), "empty images": int(np.sum(np.array(n) == 0))}
pd.DataFrame({"held-out": split_stats(HELDOUT), "test": split_stats(TEST)})
'''),
    md("## 3. POPE (verifier evaluated in isolation)\nThree negative-sampling settings, 3000 questions each, 50 % yes."),
    code(r'''
from dvd.data.pope import iter_pope  # POPE benchmark loader
rows = []
for s in cfg["pope"]["settings"]:
    items = list(iter_pope(cfg, s, max_n=cfg["pope"]["max_per_setting"]))
    rows.append({"setting": s, "questions": len(items), "yes_fraction": np.mean([i["label"] == "yes" for i in items]),
                 "unique_objects": len({i["object"] for i in items}), "example": items[0]["question"]})
    del items
pope_stats = pd.DataFrame(rows); pope_stats
'''),
    code(r'''
pope_stats.to_csv(results_dir(cfg, "data") / "pope_stats.csv", index=False)
'''),
    md("## 4. Category co-occurrence (for hallucination-style FP analysis in notebook 10)\n"
       "Li et al. report that hallucinated objects concentrate on classes that co-occur with objects truly present. "
       "We store, for each class, its 10 most frequent co-occurring classes."),
    code(r'''
from dvd.analysis import cooccurrence_topk  # per-candidate error-analysis tables and co-occurrence statistics
cooc = cooccurrence_topk(coco, k=10)
save_json({str(k): sorted(v) for k, v in cooc.items()}, results_dir(cfg, "data") / "cooccurrence_top10.json")
for c in [1, 3, 62]:
    print(coco.cat_id_to_name[c], "->", [coco.cat_id_to_name[x] for x in sorted(cooc[c])])
'''),
    md("**Next:** `02_detector_agent_baseline`."),
]

# ------------------------------------------------------------------------------------------------ 02
NB["02_detector_agent_baseline"] = [
    header("02 · Detector Agent & detector-only baseline (baseline i)",
           "Run the open-vocabulary detector over the held-out and test splits once, cache every candidate "
           "(c ≥ box_threshold), and evaluate the single-pass detector across confidence thresholds. "
           "Includes the go / no-go checkpoint from the work plan.",
           "COCO + `results/splits.json`", "`results/detections/*.jsonl`, `results/detector/*`",
           "Grounding DINO-T: ≈0.5–1.5 s/img → fast profile (150 + 300 images) ≈8 min",
           "Objective 1 (Detector Agent), Objective 2 (baseline i)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.pipeline import run_detection_stage  # cached offline stages and the online Detect-Verify-Decide pipeline
detector = build_detector(cfg, coco.cat_names, coco.name_to_cat_id)
print("backend:", cfg["detector"]["backend"], "| prompts per image:", getattr(detector, "prompts", ["(single pass)"]))
'''),
    md("## 1. Go / no-go checkpoint (50 held-out images)\nThe same cache file is used, so this work is not repeated later."),
    code(r'''
from dvd.metrics import DetectionEvaluator, sweep_detector_only, select_best  # matching, metrics, sweeps, AUROC, POPE scores
probe = HELDOUT[:50]
det_probe = run_detection_stage(detector, probe, paths.det_cache(cfg, "heldout"), coco.image, "detect probe")
ev = DetectionEvaluator(det_probe, coco.gts(probe), cfg["coco"]["iou_threshold"])
probe_rows = pd.DataFrame(sweep_detector_only(ev, cfg["decision"]["detector_only_grid"]))
best = probe_rows.loc[probe_rows.f1.idxmax()]
print(f"best F1 {best.f1:.3f} at tau={best.tau}: precision {best.precision:.3f}, recall {best.recall:.3f}")
print("GO" if best.precision > 0.3 and best.recall > 0.3 else "NO-GO: check detector config / label mapping before continuing")
probe_rows[["tau", "precision", "recall", "f1", "fp_per_image"]]
'''),
    md("## 2. Full detection runs (resumable)"),
    code(r'''
det_h = run_detection_stage(detector, HELDOUT, paths.det_cache(cfg, "heldout"), coco.image, "detect held-out")
det_t = run_detection_stage(detector, TEST, paths.det_cache(cfg, "test"), coco.image, "detect test")
n_c = [len(r["scores"]) for r in det_t.values()]
print(f"test: {np.mean(n_c):.1f} candidates / image (max {np.max(n_c)}), c >= {cfg['detector']['box_threshold']}")
'''),
    md("## 3. Detector-only operating curve\nThreshold chosen on held-out (best F1), then applied unchanged to test."),
    code(r'''
ev_h = DetectionEvaluator(det_h, coco.gts(HELDOUT), cfg["coco"]["iou_threshold"])
ev_t = DetectionEvaluator(det_t, coco.gts(TEST), cfg["coco"]["iou_threshold"])
rows_h = pd.DataFrame(sweep_detector_only(ev_h, cfg["decision"]["detector_only_grid"]))
rows_t = pd.DataFrame(sweep_detector_only(ev_t, cfg["decision"]["detector_only_grid"]))
tau_star = float(rows_h.loc[rows_h.f1.idxmax(), "tau"])
print("held-out best-F1 threshold tau* =", tau_star)
rows_t[["tau", "tp", "fp", "precision", "recall", "f1", "fp_per_image"]]
'''),
    code(r'''
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for df, name in [(rows_h, "held-out"), (rows_t, "test")]:
    ax[0].plot(df.recall, df.precision, "-o", label=name)
    ax[1].plot(df.tau, df.fp_per_image, "-o", label=name)
ax[0].set(xlabel="recall", ylabel="precision", title="Detector-only PR"); ax[1].set(xlabel="tau", ylabel="FP / image", title="False positives vs threshold")
for a in ax: a.grid(alpha=.3); a.legend()
plt.show()
'''),
    md("## 4. Sanity check with official COCO mAP\nCandidates below `box_threshold` were discarded, so mAP is a lower bound of the detector's published value. "
       "Use it only to confirm the label mapping is right (mAP should be clearly non-trivial)."),
    code(r'''
from pycocotools.cocoeval import COCOeval  # official COCO mAP evaluator (sanity check)
dets = [{"image_id": i, "category_id": c, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
        for i, r in det_t.items() for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"])]
E = COCOeval(coco.coco, coco.coco.loadRes(dets), "bbox"); E.params.imgIds = TEST
E.evaluate(); E.accumulate(); E.summarize()
'''),
    md("## 5. Cost of the detector and save"),
    code(r'''
from dvd.cost import detector_cost  # latency timer and cost model
dcost = detector_cost(det_t); print(dcost, f"-> {1 / dcost['det_latency_s']:.2f} FPS")
out = results_dir(cfg, "detector")
rows_h.to_csv(out / "detector_only_sweep_heldout.csv", index=False)
rows_t.to_csv(out / "detector_only_sweep_test.csv", index=False)
save_json({"tau_star": tau_star, "coco_map": float(E.stats[0]), "coco_map50": float(E.stats[1]), **dcost}, out / "summary.json")
save_selected(cfg, detector_only_tau=tau_star)
'''),
    md("## 6. Examples"),
    code(r'''
from dvd.viz import show_detections  # plotting helpers (detections overlay, PR frontier, crop gallery)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, iid in zip(axes.ravel(), TEST[:6]):
    r = det_t[iid]; keep = [k for k, s in enumerate(r["scores"]) if s >= tau_star]
    show_detections(coco.image(iid), [r["boxes"][k] for k in keep], [r["labels"][k] for k in keep],
                    [r["scores"][k] for k in keep], ax=ax, title=f"{iid} (c >= {tau_star})")
plt.tight_layout(); plt.show()
'''),
    md("**Next:** `03_verifier_agent_crop_verification`."),
]

# ------------------------------------------------------------------------------------------------ 03
NB["03_verifier_agent_crop_verification"] = [
    header("03 · Verifier Agent: crop-and-zoom verification + crop-margin tuning",
           "Tune the crop expansion margin on held-out, then run the frozen VLM verifier on every cached candidate of "
           "held-out and test. Also runs the global-view ablation (verifier sees the full image) and, optionally, the fully "
           "label-blind open-answer mode.",
           "`results/detections/*.jsonl`", "`results/verifier/*.jsonl`, `results/tuning/selected.json` (crop_margin), `results/verifier/*.csv`",
           "fast profile: margin search ≈5 min; held-out + test ≈10–15 min; global ablation ≈5–10 min",
           "Objective 1 (Verifier Agent), Objective 3 (ablation of evidence separation)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.pipeline import run_verification_stage  # cached offline stages and the online Detect-Verify-Decide pipeline
from dvd.experiments import SplitData  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.metrics import auroc, TP, FP  # matching, metrics, sweeps, AUROC, POPE scores
H = SplitData(cfg, coco, "heldout", HELDOUT)
T = SplitData(cfg, coco, "test", TEST)
verifier = build_verifier(cfg)
'''),
    md("## 1. Crop-margin search (held-out subset)\nCriterion: AUROC of the verifier's P(yes) for separating true from false candidates. "
       "Also reported: how many FPs it would reject and how many TPs it would wrongly reject at P(yes)=0.5."),
    code(r'''
tune_ids = HELDOUT[: cfg["margin_tuning"]["images"]]
rows = []
for m in cfg["margin_tuning"]["margins"]:
    ver = run_verification_stage(verifier, H.det, tune_ids, paths.ver_cache(cfg, "heldout", "crop", margin=m), cfg,
                                 coco.image, view="crop", margin=m, desc=f"margin {m}")
    y = np.concatenate([H.labels[i] for i in tune_ids]); p = np.concatenate([np.asarray(ver[i]["p_yes"]) for i in tune_ids])
    keep = y != -1; y, p = y[keep], p[keep]
    rows.append({"margin": m, "auroc": auroc(p, y == TP), "fp_rejected": float((p[y == FP] <= .5).mean()),
                 "tp_over_rejected": float((p[y == TP] <= .5).mean()), "yes_rate": float((p > .5).mean()),
                 "n_candidates": int(len(y))})
margin_df = pd.DataFrame(rows); margin_df.to_csv(results_dir(cfg, "verifier") / "margin_tuning.csv", index=False)
margin_df
'''),
    code(r'''
best_margin = float(margin_df.loc[margin_df.auroc.idxmax(), "margin"])
save_selected(cfg, crop_margin=best_margin)
print("selected crop margin:", best_margin)
margin_df.plot(x="margin", y=["auroc", "fp_rejected", "tp_over_rejected"], marker="o", grid=True, title="Crop margin search"); plt.show()
'''),
    md("## 2. Crop verification on all candidates (proposed verifier)"),
    code(r'''
ver_h = run_verification_stage(verifier, H.det, HELDOUT, paths.ver_cache(cfg, "heldout", "crop"), cfg, coco.image, view="crop", desc="verify held-out")
ver_t = run_verification_stage(verifier, T.det, TEST, paths.ver_cache(cfg, "test", "crop"), cfg, coco.image, view="crop", desc="verify test")
'''),
    md("## 3. Global-view ablation\nSame verifier, same question, but it sees the **whole image** instead of the crop. "
       "Isolates the contribution of evidence separation (crop-and-zoom)."),
    code(r'''
RUN_GLOBAL_ABLATION = True
if RUN_GLOBAL_ABLATION:
    gl_h = run_verification_stage(verifier, H.det, HELDOUT, paths.ver_cache(cfg, "heldout", "full"), cfg, coco.image, view="full", desc="global held-out")
    gl_t = run_verification_stage(verifier, T.det, TEST, paths.ver_cache(cfg, "test", "full"), cfg, coco.image, view="full", desc="global test")
'''),
    md("## 4. Verifier diagnostics on test"),
    code(r'''
def diag(ver, data, name):
    # Verifier diagnostics on a split: AUROC, share of FPs rejected, share of TPs wrongly rejected, yes-rate, time per call.
    y = np.concatenate([data.labels[i] for i in data.ids]); p = np.concatenate([np.asarray(ver[i]["p_yes"]) for i in data.ids])
    k = y != -1; y, p = y[k], p[k]
    t = sum(r["verify_time_s"] for r in ver.values()) / max(1, sum(r["verify_invocations"] for r in ver.values()))
    return {"view": name, "auroc": auroc(p, y == TP), "fp_rejected@0.5": float((p[y == FP] <= .5).mean()),
            "tp_over_rejected@0.5": float((p[y == TP] <= .5).mean()), "yes_rate": float((p > .5).mean()), "s_per_call": t}, (y, p)
d_crop, (y, p_crop) = diag(ver_t, T, "crop")
rows = [d_crop]
if RUN_GLOBAL_ABLATION:
    d_full, (_, p_full) = diag(gl_t, T, "full image"); rows.append(d_full)
diag_df = pd.DataFrame(rows); diag_df.to_csv(results_dir(cfg, "verifier") / "verifier_diagnostics_test.csv", index=False)
diag_df
'''),
    code(r'''
fig, axes = plt.subplots(1, 2 if RUN_GLOBAL_ABLATION else 1, figsize=(12, 4), squeeze=False)
for ax, (p, name) in zip(axes[0], [(p_crop, "crop"), (p_full if RUN_GLOBAL_ABLATION else None, "full image")]):
    ax.hist(p[y == TP], bins=40, alpha=.6, label="true detections", density=True)
    ax.hist(p[y == FP], bins=40, alpha=.6, label="false positives", density=True)
    ax.set(title=f"Verifier P(yes) - {name} view", xlabel="P(yes)"); ax.legend()
plt.show()
'''),
    md("## 5. (Optional) fully label-blind open-answer mode\nThe VLM is asked *what* the object is, without being told the label; the answer is matched to the label afterwards. "
       "Slower (generation instead of one forward pass) — run on the margin-tuning subset only."),
    code(r'''
RUN_OPEN_MODE = False
if RUN_OPEN_MODE:
    op = run_verification_stage(verifier, H.det, tune_ids, paths.ver_cache(cfg, "heldout", "crop", mode="open"), cfg,
                                coco.image, view="crop", mode="open", desc="open mode")
    y = np.concatenate([H.labels[i] for i in tune_ids]); v = np.concatenate([np.asarray(op[i]["verdict"], bool) for i in tune_ids])
    k = y != -1
    print(f"open mode: FP rejected {(~v[k][y[k] == FP]).mean():.3f} | TP over-rejected {(~v[k][y[k] == TP]).mean():.3f}")
    print("sample answers:", [a for i in tune_ids[:5] for a in (op[i]["answers"] or [])][:15])
'''),
    md("**Next:** `04_verifier_pope_eval` (verifier in isolation) and `05_decision_agent_and_pipeline`."),
]

# ------------------------------------------------------------------------------------------------ 04
NB["04_verifier_pope_eval"] = [
    header("04 · Verifier Agent in isolation on POPE (+ VCD on POPE)",
           "Evaluate the verifier's yes/no existence judgement with the standard POPE protocol in all three sampling "
           "settings (random, popular, adversarial). Report accuracy, precision, recall, F1 and Yes-rate. Visual Contrastive "
           "Decoding (α=1, β=0.1, noise step 500) is evaluated on the same questions as a decoding-time reference.",
           "`lmms-lab/POPE` (Hugging Face)", "`results/pope/*.jsonl`, `results/pope/pope_results.csv`",
           "fast profile (3 × 300 questions): plain ≈2 min, VCD ≈5 min",
           "Objective 2 (verifier on POPE), baseline iii at the verifier level"),
    BOOT, INSTALL, CFG,
    code(r'''
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.data.pope import iter_pope  # POPE benchmark loader
from dvd.metrics import pope_metrics  # matching, metrics, sweeps, AUROC, POPE scores
from dvd.cropping import fit_max_side  # crop-and-zoom / image resizing helpers
from dvd.baselines.vcd import vcd_yes_no  # baseline (iii): Visual Contrastive Decoding
from dvd.cost import Timer  # latency timer and cost model
from tqdm.auto import tqdm  # progress bars
verifier = build_verifier(cfg)
SIDE = cfg["verifier"]["full_image_max_side"]
SUFFIX = " Answer yes or no."
MAX_N = cfg["pope"]["max_per_setting"]   # set e.g. 500 in configs/default.yaml for a quick run
RUN_VCD = True
'''),
    code(r'''
def run_pope(setting, method):
    # Run (or resume) one POPE setting with plain or VCD yes/no and return its metrics.
    cache = JsonlCache(paths.pope_cache(cfg, setting, method), key="idx")
    batch = []
    def flush():
        # Score the collected batch, write one cache line per question, and empty the batch.
        imgs = [fit_max_side(b["image"], SIDE) for b in batch]; qs = [b["question"].strip() + SUFFIX for b in batch]
        with Timer() as t:
            p = verifier.yes_no(imgs, qs) if method == "plain" else vcd_yes_no(verifier, imgs, qs, cfg["vcd"]["alpha"],
                                                                                  cfg["vcd"]["beta"], cfg["vcd"]["noise_step"])[0]
        for b, pi in zip(batch, p):
            cache.add({"idx": b["idx"], "question": b["question"], "label": b["label"], "p_yes": float(pi),
                       "pred": "yes" if pi > .5 else "no", "time_s": t.elapsed / len(batch)})
        batch.clear()
    for item in tqdm(iter_pope(cfg, setting, MAX_N), desc=f"POPE {setting} [{method}]", total=MAX_N or 3000):
        if item["idx"] in cache: continue
        batch.append(item)
        if len(batch) == verifier.batch_size: flush()
    if batch: flush()
    recs = list(cache.records().values())
    return {"setting": setting, "method": method, **pope_metrics([r["pred"] for r in recs], [r["label"] for r in recs]),
            "s_per_question": float(np.mean([r["time_s"] for r in recs]))}
'''),
    code(r'''
rows = []
for s in cfg["pope"]["settings"]:
    rows.append(run_pope(s, "plain"))
    if RUN_VCD:
        rows.append(run_pope(s, "vcd"))
pope_df = pd.DataFrame(rows)
pope_df.to_csv(results_dir(cfg, "pope") / "pope_results.csv", index=False)
pope_df[["setting", "method", "accuracy", "precision", "recall", "f1", "yes_rate", "s_per_question"]]
'''),
    md("### Reading the table\n- **Yes-rate** far above 0.5 signals over-assertion (POPE is balanced 50/50).\n"
       "- The gap between *random* and *adversarial* measures how much co-occurrence drives false 'yes' answers.\n"
       "- Compare with published POPE numbers of other LVLMs only as context; cite them, do not mix them into our tables."),
    code(r'''
ax = pope_df.pivot(index="setting", columns="method", values="f1").loc[cfg["pope"]["settings"]].plot.bar(rot=0, title="POPE F1")
ax.grid(alpha=.3); plt.show()
pope_df.pivot(index="setting", columns="method", values="yes_rate").loc[cfg["pope"]["settings"]].plot.bar(rot=0, title="POPE Yes-rate").axhline(.5, color="k", ls="--"); plt.show()
'''),
]

# ------------------------------------------------------------------------------------------------ 05
NB["05_decision_agent_and_pipeline"] = [
    header("05 · Decision Agent & the full Detect–Verify–Decide pipeline",
           "Fuse detector confidence with the verifier verdict via accept = (c ≥ τ_high) ∨ (v = yes ∧ c ≥ τ_low). "
           "Sweep (τ_low, τ_high) on held-out to trace the precision–recall frontier, select the operating point, and apply "
           "it unchanged to test. Optionally compares an LLM aggregator against the rule.",
           "detection + verifier caches", "`results/tuning/selected.json` (τ_low, τ_high), `results/pipeline/*`",
           "seconds (offline from caches); optional LLM aggregator ≈10 min; demo ≈2 min",
           "Objective 1 (Decision Agent), Objective 2 (pipeline vs single-pass detector)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.experiments import SplitData, available_methods, sweep, operating_point, evaluate_row, PROPOSED, DET_ONLY  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.metrics import decision_breakdown  # matching, metrics, sweeps, AUROC, POPE scores
from dvd.viz import plot_pr  # plotting helpers (detections overlay, PR frontier, crop gallery)
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
MH, MT = available_methods(cfg, H), available_methods(cfg, T)
assert PROPOSED in MH and PROPOSED in MT, "crop verifier caches missing - run notebook 03"
'''),
    md("## 1. Threshold sweep on held-out"),
    code(r'''
base_rows = sweep(DET_ONLY, MH[DET_ONLY], H, cfg)
dvd_rows = sweep(PROPOSED, MH[PROPOSED], H, cfg)
sw = pd.DataFrame(dvd_rows)
heat = sw.pivot(index="tau_low", columns="tau_high", values="f1")
plt.figure(figsize=(9, 4)); plt.imshow(heat, cmap="viridis", aspect="auto")
plt.xticks(range(len(heat.columns)), heat.columns); plt.yticks(range(len(heat.index)), heat.index)
plt.xlabel("tau_high"); plt.ylabel("tau_low"); plt.colorbar(label="F1"); plt.title("Held-out F1 over the decision grid"); plt.show()
'''),
    md("## 2. Select operating points (held-out only)\n- **best-F1**: maximise F1.\n"
       "- **matched-recall**: fewest false positives whose recall ≥ the detector-only best-F1 recall (the 'same recall, fewer FPs' view)."),
    code(r'''
base_best = operating_point(base_rows)
dvd_best = operating_point(dvd_rows)
dvd_match = operating_point(dvd_rows, "matched_recall", base_best["recall"])
save_selected(cfg, tau_low=dvd_best["tau_low"], tau_high=dvd_best["tau_high"],
              matched_tau_low=dvd_match["tau_low"], matched_tau_high=dvd_match["tau_high"],
              detector_only_tau=base_best["tau"])
cols = ["method", "tau", "tau_low", "tau_high", "precision", "recall", "f1", "fp_per_image", "n_verified"]
pd.DataFrame([base_best, dvd_best, dvd_match]).reindex(columns=cols)
'''),
    md("## 3. Apply unchanged to test"),
    code(r'''
res = {"Detector only (tau*)": evaluate_row(MT[DET_ONLY], T, base_best, n_boot=1000),
       "DVD best-F1": evaluate_row(MT[PROPOSED], T, dvd_best, n_boot=1000),
       "DVD matched-recall": evaluate_row(MT[PROPOSED], T, dvd_match, n_boot=1000)}
fp0 = res["Detector only (tau*)"]["fp"]
tab = pd.DataFrame([{"config": k, **{c: v.get(c) for c in ["tau", "tau_low", "tau_high", "precision", "recall", "f1", "fp", "fp_per_image", "n_verified"]},
                     "FP reduction vs detector": 1 - v["fp"] / fp0, "F1 95% CI": v["ci"]["f1"]} for k, v in res.items()])
tab.to_csv(results_dir(cfg, "pipeline") / "test_operating_points.csv", index=False); tab
'''),
    md("## 4. Decision-level breakdown on test\nReference pool = candidates the detector alone outputs at τ_low (what the pipeline starts from)."),
    code(r'''
br = []
for name in ["DVD best-F1", "DVD matched-recall"]:
    r = res[name]
    br.append({"config": name, **decision_breakdown(T.labels, {i: T.ev.scores(i) for i in T.ids}, r["_masks"], r["tau_low"])})
br_df = pd.DataFrame(br); br_df.to_csv(results_dir(cfg, "pipeline") / "decision_breakdown_test.csv", index=False); br_df
'''),
    md("## 5. Precision–recall frontier (test)"),
    code(r'''
rows_t = {DET_ONLY: sweep(DET_ONLY, MT[DET_ONLY], T, cfg), PROPOSED: sweep(PROPOSED, MT[PROPOSED], T, cfg)}
plot_pr(rows_t, highlight={"DVD best-F1": res["DVD best-F1"], "Detector tau*": res["Detector only (tau*)"]},
        title="Test PR frontier (selection made on held-out)")
plt.savefig(results_dir(cfg, "figures") / "pr_frontier_pipeline_test.png", dpi=150, bbox_inches="tight"); plt.show()
pd.DataFrame(rows_t[PROPOSED]).to_csv(results_dir(cfg, "pipeline") / "dvd_sweep_test.csv", index=False)
'''),
    md("## 6. (Optional) LLM aggregator instead of the rule\nThe verifier's own language model (text-only) receives c, the verdict and P(yes) and answers accept/reject. "
       "Compared with the rule on the same held-out subset."),
    code(r'''
RUN_LLM_AGGREGATOR = False
if RUN_LLM_AGGREGATOR:
    from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
    from dvd.agents.decision import LLMAggregatorAgent, rule_accept  # Decision Agent: the accept/reject rule and the optional LLM aggregator
    from dvd.metrics import DetectionEvaluator  # matching, metrics, sweeps, AUROC, POPE scores
    from tqdm.auto import tqdm  # progress bars
    agg = LLMAggregatorAgent(build_verifier(cfg))
    sub = HELDOUT[: cfg["decision"]["llm_aggregator_images"]]
    ev = DetectionEvaluator({i: H.det[i] for i in sub}, {i: H.gts[i] for i in sub}, cfg["coco"]["iou_threshold"])
    v, p = MH[PROPOSED]["verdicts"], MH[PROPOSED]["p_yes"]
    tl, th = dvd_best["tau_low"], dvd_best["tau_high"]
    m_llm = {}
    for i in tqdm(sub):
        c = ev.scores(i); m = c >= th; band = np.where((c >= tl) & (c < th))[0]
        if len(band):
            m[band] = agg.decide([H.det[i]["labels"][k] for k in band], c[band], v[i][band], p[i][band])
        m_llm[i] = m
    m_rule = {i: rule_accept(ev.scores(i), v[i], tl, th) for i in sub}
    display(pd.DataFrame([{"decision": "rule", **ev.evaluate(m_rule)}, {"decision": "LLM aggregator", **ev.evaluate(m_llm)}]))
'''),
    md("## 7. Online pipeline demo (gated: only the uncertain band is verified)"),
    code(r'''
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.agents.decision import RuleDecisionAgent  # Decision Agent: the accept/reject rule and the optional LLM aggregator
from dvd.pipeline import DetectVerifyDecide  # cached offline stages and the online Detect-Verify-Decide pipeline
from dvd.viz import show_detections  # plotting helpers (detections overlay, PR frontier, crop gallery)
verifier = agg.verifier if RUN_LLM_AGGREGATOR else build_verifier(cfg)   # never load two VLMs on a T4
pipe = DetectVerifyDecide(build_detector(cfg, coco.cat_names, coco.name_to_cat_id), verifier,
                          RuleDecisionAgent(dvd_best["tau_low"], dvd_best["tau_high"]), cfg)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, iid in zip(axes.ravel(), TEST[10:14]):
    img = coco.image(iid); r = pipe(img)
    keep = [k for k, s in enumerate(r["scores"]) if s >= dvd_best["tau_low"]]
    st = ["accepted" if r["accept"][k] else "rejected" for k in keep]
    show_detections(img, [r["boxes"][k] for k in keep], [r["labels"][k] for k in keep], [r["scores"][k] for k in keep], st, ax=ax,
                    title=f"{iid}: {r['verify_invocations']} crops verified, {r['total_time_s']:.2f}s")
plt.tight_layout(); plt.show()
'''),
]

# ------------------------------------------------------------------------------------------------ 06
NB["06_baseline_woodpecker_style"] = [
    header("06 · Baseline ii — Woodpecker-style single-agent self-verification",
           "One VLM (the same frozen model as the verifier) sees the full image plus the detector's numbered candidate list and "
           "decides keep/reject for every item in one call. Validation and decision are fused, evidence is global — "
           "the structural opposite of the proposed pipeline.",
           "detection caches", "`results/baselines/woodpecker_*.jsonl`, `results/baselines/woodpecker_*.csv`",
           "fast profile (batched generation): ≈10 min for 450 images",
           "Objective 3 (isolate the benefit of role separation)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.baselines.woodpecker_style import run_woodpecker_stage, build_prompt  # baseline (ii): Woodpecker-style single agent
from dvd.experiments import SplitData, available_methods, sweep, operating_point, evaluate_row  # methods over identical candidates: sweeps, operating points, evaluation, costs
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
verifier = build_verifier(cfg)
iid = TEST[0]; im = coco.image(iid); d = T.det[iid]
print(build_prompt(d["labels"][:5], d["boxes"][:5], *im.size))
'''),
    code(r'''
wp_h = run_woodpecker_stage(verifier, H.det, HELDOUT, paths.woodpecker_cache(cfg, "heldout"), cfg, coco.image, "woodpecker held-out")
wp_t = run_woodpecker_stage(verifier, T.det, TEST, paths.woodpecker_cache(cfg, "test"), cfg, coco.image, "woodpecker test")
print("answer parse rate (test):", np.mean([r["parse_ok"] for r in wp_t.values()]))
print("example raw answer:\n", next(r["raw"] for r in wp_t.values() if r["raw"]))
'''),
    code(r'''
NAME = "Woodpecker-style single agent"
mh, mt = available_methods(cfg, H)[NAME], available_methods(cfg, T)[NAME]
rows_h = sweep(NAME, mh, H, cfg)
best = operating_point(rows_h)
r_t = evaluate_row(mt, T, best, n_boot=1000); r_t.pop("_masks")
pd.DataFrame(rows_h).to_csv(results_dir(cfg, "baselines") / "woodpecker_sweep_heldout.csv", index=False)
save_json({"selected_on_heldout": best, "test": r_t}, results_dir(cfg, "baselines") / "woodpecker_test.json")
pd.DataFrame([r_t])[["tau", "precision", "recall", "f1", "fp", "fp_per_image"]]
'''),
]

# ------------------------------------------------------------------------------------------------ 07
NB["07_baseline_vcd"] = [
    header("07 · Baseline iii — Visual Contrastive Decoding (VCD) on the same detections",
           "For each image and each distinct candidate label, ask the VLM on the full image whether the object exists, using "
           "VCD (logits contrasted against a diffusion-noised copy, α=1, β=0.1, noise step 500). A candidate is kept iff the "
           "answer for its label is yes. The same passes also give the plain (non-contrastive) global answer.",
           "detection caches", "`results/baselines/vcd_*.jsonl`, `results/baselines/vcd_*.json`",
           "2 forward passes per (image, label) → fast profile ≈10 min for 450 images",
           "Baseline iii (decoding-time mitigation)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.baselines.vcd import run_vcd_stage, diffusion_noise  # baseline (iii): Visual Contrastive Decoding
from dvd.cropping import fit_max_side  # crop-and-zoom / image resizing helpers
from dvd.experiments import SplitData, available_methods, sweep, operating_point, evaluate_row  # methods over identical candidates: sweeps, operating points, evaluation, costs
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
verifier = build_verifier(cfg)
im = fit_max_side(coco.image(TEST[0]), 448)
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].imshow(im); ax[0].set_title("original")
ax[1].imshow(diffusion_noise(im, cfg["vcd"]["noise_step"])); ax[1].set_title(f"noised (step {cfg['vcd']['noise_step']})")
for a in ax: a.axis("off")
plt.show()
'''),
    code(r'''
vcd_h = run_vcd_stage(verifier, H.det, HELDOUT, paths.vcd_cache(cfg, "heldout"), cfg, coco.image, "vcd held-out")
vcd_t = run_vcd_stage(verifier, T.det, TEST, paths.vcd_cache(cfg, "test"), cfg, coco.image, "vcd test")
'''),
    code(r'''
out = []
for NAME in ["VCD global yes/no", "Plain global yes/no (no contrast)"]:
    mh, mt = available_methods(cfg, H)[NAME], available_methods(cfg, T)[NAME]
    best = operating_point(sweep(NAME, mh, H, cfg))
    r = evaluate_row(mt, T, best, n_boot=1000); r.pop("_masks")
    out.append({"method": NAME, **r})
save_json(out, results_dir(cfg, "baselines") / "vcd_test.json")
pd.DataFrame(out)[["method", "tau", "precision", "recall", "f1", "fp", "fp_per_image"]]
'''),
]

# ------------------------------------------------------------------------------------------------ 08
NB["08_main_comparison_coco"] = [
    header("08 · Main comparison on COCO (all methods, identical models)",
           "Every method works on the same detector candidates. Its operating point is selected on held-out (best-F1 and "
           "matched-recall), then evaluated unchanged on test, with image-level bootstrap 95 % CIs. Produces the main results "
           "table and the precision–recall frontier figure.",
           "all caches from notebooks 02, 03, 06, 07", "`results/tables/main_comparison.{csv,md}`, `results/figures/*.png`",
           "≈2–5 min (offline)", "Objectives 2 and 3"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.experiments import SplitData, available_methods, main_comparison, DET_ONLY, PROPOSED  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.viz import plot_pr  # plotting helpers (detections overlay, PR frontier, crop gallery)
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
MH, MT = available_methods(cfg, H), available_methods(cfg, T)
print("methods with complete caches:", list(MT))
'''),
    code(r'''
# selection on held-out, evaluation on test, bootstrap CIs + paired bootstrap against the detector-only baseline
main, sweeps_h, sweeps_t, base_h = main_comparison(cfg, H, T, MH, MT, n_boot=1000)
main[main.criterion == "best_f1"][["method", "precision", "recall", "f1", "fp_per_image", "fp_reduction", "fp_reduction_ci", "over_rejection_rate", "total_latency_s"]]
'''),
    code(r'''
main[main.criterion == "matched_recall"][["method", "precision", "recall", "d_recall_ci", "f1", "fp_per_image", "fp_reduction", "fp_reduction_ci", "over_rejection_rate", "total_latency_s"]]
'''),
    md("## Save the tables"),
    code(r'''
tdir = results_dir(cfg, "tables")
main.to_csv(tdir / "main_comparison.csv", index=False)
fmt = main.copy()
for c in ["precision", "recall", "f1", "fp_reduction", "fp_removed_rate", "over_rejection_rate"]:
    fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
for c in ["f1_ci", "fp_reduction_ci", "d_recall_ci", "d_f1_ci"]:
    fmt[c] = fmt[c].map(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
cols = ["criterion", "method", "precision", "recall", "f1", "f1_ci", "fp_per_image", "fp_reduction", "fp_reduction_ci", "d_recall_ci", "over_rejection_rate", "total_latency_s", "total_invocations"]
(tdir / "main_comparison.md").write_text(fmt[cols].to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
print((tdir / "main_comparison.md").read_text())
'''),
    md("## Precision–recall frontiers (test)"),
    code(r'''
sel = {n: main[(main.criterion == "best_f1") & (main.method == n)].iloc[0] for n in [DET_ONLY, PROPOSED] if n in MT}
plot_pr(sweeps_t, highlight={k: v for k, v in sel.items()}, title="All methods - test split")
plt.savefig(results_dir(cfg, "figures") / "pr_frontier_all_methods_test.png", dpi=150, bbox_inches="tight"); plt.show()
'''),
    md("""
## Interpretation guide
- **Proposed vs Detector only** → does a verification stage reduce FPs at equal (matched) recall? (Objective 2)
- **Proposed vs Woodpecker-style** → same VLM, same candidates; difference = role separation + crop evidence. (Objective 3)
- **Proposed vs Global-view verifier + rule** → difference = crop-and-zoom evidence only.
- **Proposed vs Crop verifier decides alone** → difference = fusing detector confidence in a separate decision agent.
- **VCD / plain global yes/no** → a decoding-time remedy on global evidence.
- Overlapping bootstrap CIs mean the difference is not established on this test size; say so in the report.
"""),
]

# ------------------------------------------------------------------------------------------------ 09
NB["09_cost_analysis"] = [
    header("09 · Accuracy-versus-cost trade-off",
           "Quantify the added latency and model invocations of every configuration, validate the cost model against a real "
           "online run of the gated pipeline, and characterise FP reduction per unit of added cost.",
           "all caches", "`results/cost/*.csv`, `results/figures/cost_*.png`",
           "offline part: seconds; online measurement ≈2 min for 30 images", "Objective 4"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.experiments import SplitData, available_methods, sweep, operating_point, evaluate_row, method_cost, DET_ONLY, PROPOSED  # methods over identical candidates: sweeps, operating points, evaluation, costs
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
MT = available_methods(cfg, T)
fp_base = evaluate_row(MT[DET_ONLY], T, {"tau": cfg["selected"]["detector_only_tau"]})["fp"]
'''),
    md("## 1. Cost of every (τ_low, τ_high) configuration of the pipeline"),
    code(r'''
rows = []
for r in sweep(PROPOSED, MT[PROPOSED], T, cfg):
    c = method_cost(MT[PROPOSED], T, r["n_verified"])
    rows.append({**r, **c, "fp_reduction": 1 - r["fp"] / fp_base})
cost_df = pd.DataFrame(rows)
cost_df.to_csv(results_dir(cfg, "cost") / "pipeline_config_costs_test.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sc = ax[0].scatter(cost_df.added_latency_s, cost_df.fp_reduction, c=cost_df.recall, cmap="viridis", s=18)
plt.colorbar(sc, ax=ax[0], label="recall"); ax[0].set(xlabel="added latency per image (s)", ylabel="FP reduction vs detector tau*", title="FP removed vs added latency")
ax[1].scatter(cost_df.verify_invocations, cost_df.f1, s=18); ax[1].set(xlabel="verifier calls per image", ylabel="F1", title="F1 vs verifier calls")
for a in ax: a.grid(alpha=.3)
plt.savefig(results_dir(cfg, "figures") / "cost_pipeline_tradeoff.png", dpi=150, bbox_inches="tight"); plt.show()
'''),
    md("## 2. Method-level cost at each method's selected operating point"),
    code(r'''
main = pd.read_csv(results_dir(cfg, "tables") / "main_comparison.csv")
m = main[main.criterion == "best_f1"][["method", "f1", "fp_reduction", "det_latency_s", "added_latency_s", "total_latency_s",
                                         "det_invocations", "verify_invocations", "total_invocations"]].copy()
m["fps"] = 1 / m.total_latency_s
m["fp_reduction_per_added_s"] = m.fp_reduction / m.added_latency_s.replace(0, np.nan)
m.to_csv(results_dir(cfg, "cost") / "method_costs_test.csv", index=False); m
'''),
    md("## 3. Validate the cost model with a real online run\nThe deployable pipeline verifies only the uncertain band; its measured latency should match the cost-model estimate."),
    code(r'''
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.agents.decision import RuleDecisionAgent  # Decision Agent: the accept/reject rule and the optional LLM aggregator
from dvd.pipeline import DetectVerifyDecide  # cached offline stages and the online Detect-Verify-Decide pipeline
from tqdm.auto import tqdm  # progress bars
tl, th = cfg["selected"]["tau_low"], cfg["selected"]["tau_high"]
pipe = DetectVerifyDecide(build_detector(cfg, coco.cat_names, coco.name_to_cat_id), build_verifier(cfg), RuleDecisionAgent(tl, th), cfg)
ids = TEST[: cfg["cost"]["online_latency_images"]]
pipe(coco.image(ids[0]))  # warm-up
online = []
for iid in tqdm(ids):
    r = pipe(coco.image(iid))
    online.append({"det_s": r["det_time_s"], "verify_s": r["verify_time_s"], "decide_s": r["decision_time_s"],
                   "total_s": r["total_time_s"], "verify_calls": r["verify_invocations"]})
online = pd.DataFrame(online)
est = cost_df[(cost_df.tau_low == tl) & (cost_df.tau_high == th)].iloc[0]
summary = pd.DataFrame({"measured (online, gated)": [online.det_s.mean(), online.verify_s.mean(), online.total_s.mean(), online.verify_calls.mean()],
                        "estimated (cost model)": [est.det_latency_s, est.added_latency_s, est.total_latency_s, est.verify_invocations]},
                       index=["detector s/img", "verifier s/img", "total s/img", "verifier calls/img"])
summary.to_csv(results_dir(cfg, "cost") / "online_vs_estimated.csv"); summary
'''),
    code(r'''
print(f"Online pipeline: {1 / online.total_s.mean():.2f} FPS vs detector alone {1 / online.det_s.mean():.2f} FPS "
      f"(x{online.total_s.mean() / online.det_s.mean():.2f} latency); decision agent {online.decide_s.mean() * 1e3:.3f} ms/img")
online.total_s.hist(bins=30); plt.xlabel("end-to-end seconds / image"); plt.title("Online latency distribution (bounded: single fixed pass)"); plt.show()
'''),
]

# ------------------------------------------------------------------------------------------------ 10
NB["10_error_analysis"] = [
    header("10 · Error analysis — where verification helps, and where it hurts",
           "Break the selected pipeline's decisions down by category, object size, detector confidence, image brightness "
           "(low-light proxy) and co-occurrence-driven false positives; show galleries of correct rejections, omissions and "
           "over-rejections.",
           "detection + verifier caches, `results/data/cooccurrence_top10.json`", "`results/error_analysis/*.csv`, figures",
           "≈5 min", "Expected contribution 4"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.experiments import SplitData, available_methods, masks_for, PROPOSED  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.analysis import candidate_table, mean_brightness  # per-candidate error-analysis tables and co-occurrence statistics
from dvd.viz import crop_gallery  # plotting helpers (detections overlay, PR frontier, crop gallery)
from tqdm.auto import tqdm  # progress bars
T = SplitData(cfg, coco, "test", TEST)
M = available_methods(cfg, T)[PROPOSED]
tl, th = cfg["selected"]["tau_low"], cfg["selected"]["tau_high"]
accept = masks_for(M, T, {"tau_low": tl, "tau_high": th})
cooc = {int(k): set(v) for k, v in load_json(results_dir(cfg, "data") / "cooccurrence_top10.json").items()}
bright = {i: mean_brightness(coco.image(i)) for i in tqdm(TEST, desc="brightness")}
df = pd.DataFrame(candidate_table(T.det, T.labels, accept, T.gts, M["verdicts"], M["p_yes"], cooc, bright))
df = df[(df.score >= tl) & (df.outcome != "ignore")]          # reference pool: what the detector outputs at tau_low
df["band"] = np.where(df.score >= th, "auto-accept (c >= tau_high)", "verified band")
print(df.outcome.value_counts())
'''),
    code(r'''
def rates(g):
    # Per group: candidate count, FP/TP counts, share of FPs removed and share of TPs lost.
    fp, tp = g[g.is_fp], g[g.is_tp]
    return pd.Series({"candidates": len(g), "FP": len(fp), "TP": len(tp),
                      "FP removed (correct rejection)": (~fp.accepted).mean() if len(fp) else np.nan,
                      "TP lost (over-rejection)": (~tp.accepted).mean() if len(tp) else np.nan})
out = results_dir(cfg, "error_analysis")
by_cat = df.groupby("label").apply(rates).sort_values("FP", ascending=False); by_cat.to_csv(out / "by_category.csv")
by_cat.head(20)
'''),
    code(r'''
df["conf_bin"] = pd.cut(df.score, [0, .3, .4, .5, .6, .8, 1.0])
df["brightness_bin"] = pd.qcut(df.brightness, 3, labels=["dark", "medium", "bright"])
for col in ["size", "conf_bin", "brightness_bin", "cooccurrence_fp", "band"]:
    t = df.groupby(col, observed=True).apply(rates); t.to_csv(out / f"by_{col}.csv")
    print(f"\n=== by {col} ==="); display(t)
'''),
    md("### Where does the verifier help most / hurt most?"),
    code(r'''
k = by_cat[by_cat.FP >= 20]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
k.sort_values("FP removed (correct rejection)").tail(15)["FP removed (correct rejection)"].plot.barh(ax=ax[0], title="Highest FP-removal categories (>=20 FPs)")
by_cat[by_cat.TP >= 20].sort_values("TP lost (over-rejection)").tail(15)["TP lost (over-rejection)"].plot.barh(ax=ax[1], color="C3", title="Highest over-rejection categories (>=20 TPs)")
plt.tight_layout(); plt.savefig(results_dir(cfg, "figures") / "error_by_category.png", dpi=150, bbox_inches="tight"); plt.show()
'''),
    md("### Galleries (crops exactly as the verifier saw them)"),
    code(r'''
def gallery_items(outcome, n=12):
    # Random sample of candidates with a given outcome, formatted for crop_gallery.
    pool = df[(df.outcome == outcome) & (df.band == "verified band")]
    s = pool.sample(min(n, len(pool)), random_state=0)
    return [{"image_id": r.image_id, "box": r.box, "label": r.label, "score": r.score, "p_yes": r.p_yes} for r in s.itertuples()]
for oc in ["correct_rejection", "omission", "over_rejection"]:
    crop_gallery(gallery_items(oc), coco.image, cfg["verifier"]["crop_margin"], title=oc.replace("_", " "))
    plt.savefig(results_dir(cfg, "figures") / f"gallery_{oc}.png", dpi=120, bbox_inches="tight"); plt.show()
'''),
]

# ------------------------------------------------------------------------------------------------ 11
NB["11_optional_robustness_monitoring"] = [
    header("11 · (Optional) Monitoring conditions: degraded footage + your own CCTV images",
           "Proxy for surveillance / traffic footage: apply low-light, motion-blur, low-resolution and JPEG degradations to a "
           "test subset (COCO labels stay valid) and compare detector-only vs the pipeline using the thresholds selected on clean "
           "held-out data. A second section runs the online pipeline on a folder of your own images (qualitative).",
           "COCO test subset, optional image folder", "`results/robustness/*.csv`", "fast profile: ≈2 min per condition (40 images)",
           "Optional extension (proposal §6.1)"),
    BOOT, INSTALL, CFG, COCO,
    code(r'''
from dvd.degrade import degrade  # synthetic low-light / blur / low-res / JPEG degradations
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.agents.verifier import build_verifier  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.pipeline import run_detection_stage, run_verification_stage  # cached offline stages and the online Detect-Verify-Decide pipeline
from dvd.experiments import SplitData, evaluate_row, cached  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.metrics import decision_breakdown  # matching, metrics, sweeps, AUROC, POPE scores
SUB = TEST[: cfg["robustness"]["images"]]
detector = build_detector(cfg, coco.cat_names, coco.name_to_cat_id); verifier = build_verifier(cfg)
iid = SUB[0]
fig, ax = plt.subplots(1, 5, figsize=(22, 4))
for a, c in zip(ax, ["clean"] + cfg["robustness"]["conditions"]):
    a.imshow(coco.image(iid) if c == "clean" else degrade(coco.image(iid), c, iid)); a.set_title(c); a.axis("off")
plt.show()
'''),
    code(r'''
sel = cfg["selected"]; rows = []
for cond in ["clean"] + cfg["robustness"]["conditions"]:
    # clean images reuse the test-split caches from notebooks 02/03 (already computed -> no-op)
    split, tag = ("test", "") if cond == "clean" else ("robust", f"_{cond}")
    fn = coco.image if cond == "clean" else (lambda i, c=cond: degrade(coco.image(i), c, i))
    det = run_detection_stage(detector, SUB, paths.det_cache(cfg, split, tag), fn, f"detect {cond}")
    run_verification_stage(verifier, det, SUB, paths.ver_cache(cfg, split, "crop", tag=tag), cfg, fn, view="crop", desc=f"verify {cond}")
    D = SplitData(cfg, coco, split, SUB, det_tag=tag)
    v = cached(paths.ver_cache(cfg, split, "crop", tag=tag), SUB)
    meth = {"kind": "rule", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in v.items()}}
    base = evaluate_row({"kind": "none"}, D, {"tau": sel["detector_only_tau"]})
    dvd = evaluate_row(meth, D, {"tau_low": sel["tau_low"], "tau_high": sel["tau_high"]})
    b = decision_breakdown(D.labels, {i: D.ev.scores(i) for i in SUB}, dvd.pop("_masks"), sel["tau_low"])
    base.pop("_masks")
    rows.append({"condition": cond, "det_precision": base["precision"], "det_recall": base["recall"], "det_fp_per_img": base["fp_per_image"],
                 "dvd_precision": dvd["precision"], "dvd_recall": dvd["recall"], "dvd_fp_per_img": dvd["fp_per_image"],
                 "fp_reduction": 1 - dvd["fp"] / max(base["fp"], 1), "over_rejection_rate": b["over_rejection_rate"]})
rob = pd.DataFrame(rows); rob.to_csv(results_dir(cfg, "robustness") / "degradations.csv", index=False); rob
'''),
    md("## Your own monitoring images (qualitative)\nPut `.jpg/.png` frames in `data/custom_images/` on Drive (no labels needed)."),
    code(r'''
from pathlib import Path  # portable file paths
from PIL import Image  # Pillow image type
from dvd.agents.decision import RuleDecisionAgent  # Decision Agent: the accept/reject rule and the optional LLM aggregator
from dvd.pipeline import DetectVerifyDecide  # cached offline stages and the online Detect-Verify-Decide pipeline
from dvd.viz import show_detections  # plotting helpers (detections overlay, PR frontier, crop gallery)
CUSTOM_DIR = Path(cfg["root"]) / "data" / "custom_images"
files = sorted([p for p in CUSTOM_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:8] if CUSTOM_DIR.exists() else []
print(f"{len(files)} custom images found in {CUSTOM_DIR}")
pipe = DetectVerifyDecide(detector, verifier, RuleDecisionAgent(sel["tau_low"], sel["tau_high"]), cfg)
for p in files:
    img = Image.open(p).convert("RGB"); r = pipe(img)
    keep = [k for k, s in enumerate(r["scores"]) if s >= sel["tau_low"]]
    show_detections(img, [r["boxes"][k] for k in keep], [r["labels"][k] for k in keep], [r["scores"][k] for k in keep],
                    ["accepted" if r["accept"][k] else "rejected" for k in keep], title=f"{p.name} - {r['total_time_s']:.2f}s"); plt.show()
'''),
]

# ------------------------------------------------------------------------------------------------ 12
NB["12_streamlit_demo"] = [
    header("12 · Streamlit demo UI (live agent view)",
           "Launch the Streamlit app (`app/streamlit_app.py`) from Colab and open it through a temporary public URL. "
           "The app shows which agent is running, the current step, a progress percentage and a live feedback log, "
           "then the final detections with the reason for every accept / reject.",
           "uploaded image or the COCO sample; optional tuned thresholds from `results/tuning/selected.json`",
           "interactive UI only (nothing is written to `results/`)", "≈3 min to start (models load once)",
           "Demo of Objective 1 (the three-agent pipeline)"),
    BOOT, INSTALL,
    md("""
> **Heads-up.** Colab cannot expose a port directly, so this notebook starts a **Cloudflare quick tunnel**.
> It downloads the `cloudflared` binary from Cloudflare's GitHub releases and prints a temporary
> `https://<random>.trycloudflare.com` URL. **Anyone who has that URL can use the app on your GPU session**
> until you run the last cell (or the runtime stops). Do not share the URL.
"""),
    code(r'''
# 1. cloudflared (tunnel client)
if not os.path.exists("/usr/local/bin/cloudflared"):
    subprocess.run(["wget", "-q", "-O", "/usr/local/bin/cloudflared",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)
print("cloudflared ready")
'''),
    code(r'''
# 2. start the Streamlit app in the background (log: /content/streamlit.log)
import time, re  # stopwatch and regular expressions (find the tunnel URL)
os.environ["DVD_ROOT"] = PROJECT_ROOT
app_proc = subprocess.Popen(
    ["streamlit", "run", f"{PROJECT_ROOT}/app/streamlit_app.py", "--server.port", "8501", "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=open("/content/streamlit.log", "w"), stderr=subprocess.STDOUT)
time.sleep(8)
print("streamlit running:", app_proc.poll() is None)
if app_proc.poll() is not None:
    print(open("/content/streamlit.log").read()[-2000:])
'''),
    code(r'''
# 3. open the tunnel and print the URL
tunnel_log = open("/content/cloudflared.log", "w")
tunnel_proc = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8501"], stdout=tunnel_log, stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open("/content/cloudflared.log").read())
    if m:
        url = m.group(0); break
print("Open this URL in a new tab:" if url else "No URL yet - see /content/cloudflared.log", url or "")
'''),
    md("""
## Using the app
1. Wait for **Loading models** to finish (first start downloads the weights).
2. Upload an image, or press **Use a COCO sample image**.
3. Adjust the uncertain band (τ_low, τ_high) and the crop margin if you like, then press **Run pipeline**.
4. Watch the five agent cards, the progress percentage and the log update live; then open the tabs
   (**Result**, **Verifier crops**, **All candidates**, **Run log**).

If the page shows errors, check `/content/streamlit.log`.
"""),
    code(r'''
# 4. stop everything (also closes the public URL)
for p in (globals().get("tunnel_proc"), globals().get("app_proc")):
    if p is not None and p.poll() is None:
        p.terminate()
print("stopped")
'''),
]

exec(compile((Path(__file__).parent / "nb_run_all_fast.py").read_text(encoding="utf-8"), "nb_run_all_fast.py", "exec"))

for name, cells in NB.items():
    if ONLY and name not in ONLY:
        continue
    gpu = name not in ("08_main_comparison_coco",)
    (OUT / f"{name}.ipynb").write_text(json.dumps(notebook(cells, gpu), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)
