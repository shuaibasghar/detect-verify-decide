# Cells of notebooks/run_all_fast.ipynb. Executed from tools/build_notebooks.py (uses md / code / header / NB defined there).
# One notebook, one Colab session, one model load: the whole proposal (all baselines, ablations, POPE, cost, error
# analysis, robustness) on the fast profile (configs/default.yaml), written for ~2 h on a T4.

NB["run_all_fast"] = [
    md("""
# Detect–Verify–Decide — the whole project in one run (fast profile)

**Purpose.** Produce every result the proposal asks for — the three-agent pipeline against the detector-only, Woodpecker-style
and VCD baselines, the ablations, POPE, the cost trade-off, the error analysis and the monitoring-conditions test — in **one
Colab session of about 2 hours** (T4). Run all cells top to bottom (*Runtime → Run all*).

| Step | What | Rough time on a T4 |
|---|---|---|
| 0 | install, data (annotation file + 450 images), load both models | ≈ 8–12 min |
| 1 | Detector Agent on 150 held-out + 300 test images | ≈ 8 min |
| 2 | crop-margin search (3 margins × 60 held-out images) | ≈ 5 min |
| 3 | Verifier Agent on all candidates, crop view | ≈ 10–15 min |
| 4 | global-view ablation, Decision Agent sweep, LLM aggregator | ≈ 10 min |
| 5–7 | Woodpecker-style baseline, VCD baseline, POPE | ≈ 25 min |
| 8–11 | comparison table with paired CIs, cost + online run, error analysis, robustness | ≈ 15 min |

Cell 7 **measures** the real speed on your GPU and prints a projection, so you know the total before the long stages start.

**Nothing is trained.** All models are frozen; only the crop margin and (τ_low, τ_high) are chosen, on the held-out split.
Results are written to `results_fast/` on Drive (and copied there after every step, so a disconnect loses nothing:
re-run from the top and finished steps are skipped). `results_fast/SUMMARY.md` collects the headline numbers.
"""),
    md("## 0 · Setup"),
    code(r'''
# ---- bootstrap: Drive (Colab) or /kaggle/working (Kaggle), project folder, latest code ---------------------------------
import os, sys, subprocess, time, shutil, contextlib  # paths, sys.path, shell commands, clock, file copy, context-manager helper
T0 = time.time()
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = (not IN_COLAB) and os.path.isdir("/kaggle/working")    # Kaggle notebook (needs Internet ON, see docs/KAGGLE_GUIDE.md)
DRIVE_PROJECT = "/content/drive/MyDrive/detect-verify-decide"
REPO_URL = ""  # only needed if the project is not yet in Drive, e.g. "https://github.com/<you>/detect-verify-decide.git"
BRANCH = "kaggle-version"  # git branch to take the latest code from (change it if you push to another branch)

if IN_COLAB:
    from google.colab import drive  # Colab only: mount Google Drive
    drive.mount("/content/drive")
    if not os.path.isdir(DRIVE_PROJECT) and REPO_URL:
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, DRIVE_PROJECT], check=True)
    elif os.path.isdir(os.path.join(DRIVE_PROJECT, ".git")):      # pick up the latest pushed code (never touches your notebooks/results)
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
'''),
    code(r'''
# ---- dependencies (Colab / Kaggle keep their own torch/CUDA build; the optional YOLO-World and Streamlit packages are skipped) ----
if IN_COLAB or IN_KAGGLE:
    reqs = [l.split("#")[0].strip() for l in open(f"{PROJECT_ROOT}/requirements.txt", encoding="utf-8")]
    reqs = [r for r in reqs if r and not r.startswith(("ultralytics", "streamlit"))]
    open("/tmp/requirements_fast.txt", "w").write("\n".join(reqs))
    %pip install -q -r /tmp/requirements_fast.txt
'''),
    code(r'''
# ---- config, results folder (Colab: fast local disk, copied to Drive after every step; Kaggle: /kaggle/working), timing helper ----
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt  # json files, numpy arrays, pandas tables, matplotlib plots
pd.set_option("display.precision", 4); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

RESULTS_DRIVE = f"{PROJECT_ROOT}/results_fast"
RESULTS_LOCAL = "/content/dvd_results" if IN_COLAB else RESULTS_DRIVE
if IN_COLAB:
    os.makedirs(RESULTS_DRIVE, exist_ok=True)
    shutil.copytree(RESULTS_DRIVE, RESULTS_LOCAL, dirs_exist_ok=True)       # resume from a previous session
if IN_KAGGLE:   # resume: results of an earlier version, attached via Add Input -> Notebook output (or as a Dataset)
    import glob
    os.makedirs(RESULTS_LOCAL, exist_ok=True)
    for prev in glob.glob("/kaggle/input/**/results_fast", recursive=True):
        shutil.copytree(prev, RESULTS_LOCAL, dirs_exist_ok=True)
        print("restored earlier results from", prev)

from dvd.config import load_config, save_selected, results_dir  # load configs/default.yaml, save tuned values, results folders
from dvd.io_utils import set_seed, save_json, load_json, JsonlCache  # seeding, JSON helpers and the resumable JSONL cache
from dvd import paths  # cache-file path helpers
cfg = load_config(overrides={"paths": {"results_root": RESULTS_LOCAL}})
set_seed(cfg["seed"])

def sync():
    # Copy the local results folder to Drive (no-op outside Colab).
    if IN_COLAB:
        shutil.copytree(RESULTS_LOCAL, RESULTS_DRIVE, dirs_exist_ok=True)

TIMES = {}
@contextlib.contextmanager
def stage(name):
    # Context manager: prints the step name, measures its duration, syncs results to Drive afterwards.
    t = time.time(); print(f"\n>>> {name}")
    yield
    TIMES[name] = time.time() - t; sync()
    print(f"<<< {name}: {TIMES[name] / 60:.1f} min   (session total {(time.time() - T0) / 60:.1f} min)")

import torch  # PyTorch: GPU check and memory statistics
assert torch.cuda.is_available(), "No GPU: Colab: Runtime -> Change runtime type -> T4 GPU | Kaggle: Settings -> Accelerator -> GPU T4 x2"
print("GPU:", torch.cuda.get_device_name(0), "| platform:", "Colab" if IN_COLAB else ("Kaggle" if IN_KAGGLE else "local"))
if IN_KAGGLE and "P100" in torch.cuda.get_device_name(0):
    print("WARNING: the Kaggle P100 is often not supported by the current PyTorch build; choose GPU T4 x2 instead.")
print("results:", cfg["paths"]["results_root"], "| selected so far:", cfg["selected"] or "(none)")
print({k: cfg[k] for k in ("coco", "margin_tuning")}, "| crop", cfg["verifier"]["crop_target_size"], "px | max candidates/img", cfg["detector"]["max_detections"])
'''),
    code(r'''
# ---- data: annotation file + only the 450 images we need ---------------------------------------------------------------
from dvd.data.coco import prepare_annotations, fetch_images, CocoData, make_splits  # COCO annotations, image download, splits
with stage("0 data"):
    prepare_annotations(cfg)
    coco = CocoData(cfg)
    splits = make_splits(cfg, coco.image_ids)                 # seeded, created once, frozen in results_fast/splits.json
    HELDOUT, TEST = splits["heldout"], splits["test"]
    assert len(HELDOUT) == cfg["coco"]["heldout_size"] and len(TEST) == (cfg["coco"]["test_size"] or len(TEST)), \
        "results_fast/splits.json was created with other sizes - delete results_fast/ on Drive and re-run"
    assert not set(HELDOUT) & set(TEST)
    fetch_images(cfg, HELDOUT + TEST)
print(f"held-out (tuning only): {len(HELDOUT)} images | test (all reported numbers): {len(TEST)} images")
'''),
    code(r'''
# ---- both frozen models, loaded once for the whole session ---------------------------------------------------------------
from dvd.agents.detector import build_detector  # Detector Agent (Grounding DINO): factory
from dvd.agents.verifier import build_verifier, make_views  # Verifier Agent (frozen Qwen2-VL): factory and crop builder
from dvd.pipeline import run_detection_stage, run_verification_stage, verify_global, DetectVerifyDecide  # cached offline stages and the online Detect-Verify-Decide pipeline
from dvd.agents.decision import RuleDecisionAgent  # Decision Agent: the accept/reject rule and the optional LLM aggregator
from dvd.cost import Timer  # latency timer and cost model
from dvd.experiments import (SplitData, available_methods, sweep, operating_point, evaluate_row, method_cost,  # methods over identical candidates: sweeps, operating points, evaluation, costs
                             main_comparison, paired_vs_proposed, DET_ONLY, PROPOSED)
from dvd.metrics import auroc, TP, FP, DetectionEvaluator, decision_breakdown  # matching, metrics, sweeps, AUROC, POPE scores
with stage("load models"):
    detector = build_detector(cfg, coco.cat_names, coco.name_to_cat_id)     # Grounding DINO-T
    verifier = build_verifier(cfg)                                          # Qwen2-VL-2B
print(f"peak GPU memory after loading: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
'''),
    md("## 1 · Detector Agent (baseline i)\nEvery candidate with c ≥ 0.25 (at most 20 per image) is cached once. All methods work on these **same** candidates."),
    code(r'''
with stage("1 detector on held-out + test"):
    run_detection_stage(detector, HELDOUT, paths.det_cache(cfg, "heldout"), coco.image, "detect held-out")
    run_detection_stage(detector, TEST, paths.det_cache(cfg, "test"), coco.image, "detect test")
H = SplitData(cfg, coco, "heldout", HELDOUT); T = SplitData(cfg, coco, "test", TEST)
n_h = np.array([len(H.det[i]["scores"]) for i in HELDOUT]); n_t = np.array([len(T.det[i]["scores"]) for i in TEST])
print(f"candidates per image: held-out {n_h.mean():.1f} | test {n_t.mean():.1f} (max {n_t.max()}) "
      f"| total {n_h.sum() + n_t.sum()} | detector {np.mean([T.det[i]['det_time_s'] for i in TEST]):.2f} s/img")
rows = pd.DataFrame(sweep(DET_ONLY, {"kind": "none"}, H, cfg)); b = rows.loc[rows.f1.idxmax()]
print(f"go/no-go (held-out, detector only): best F1 {b.f1:.3f} at tau={b.tau} | precision {b.precision:.3f} recall {b.recall:.3f}",
      "-> GO" if b.precision > .3 and b.recall > .3 else "-> NO-GO: check the detector configuration")
'''),
    md("### Speed check (so the long stages are not a surprise)\nTimes the verifier on a few real crops of **your** GPU and projects the stages that dominate the runtime."),
    code(r'''
crops, labs, n_uniq = [], [], 0
for iid in HELDOUT:
    d = H.det[iid]
    if d["scores"]:
        crops += make_views(coco.image(iid), d["boxes"], cfg, "crop"); labs += d["labels"]
    if len(crops) >= 48: break
verifier.verify(crops[:8], labs[:8])                                    # warm-up
with Timer() as t: verifier.verify(crops[:48], labs[:48])
s_crop = t.elapsed / len(labs[:48])
g_imgs = [i for i in HELDOUT if H.det[i]["scores"]][:4]
with Timer() as t:
    for i in g_imgs: verify_global(verifier, coco.image(i), H.det[i]["labels"], cfg)
n_pairs = sum(len(set(H.det[i]["labels"])) for i in g_imgs)
s_glob = t.elapsed / max(n_pairs, 1)
uniq_all = sum(len(set(r["labels"])) for r in list(H.det.values()) + list(T.det.values()))
tune_c = int(sum(len(H.det[i]["scores"]) for i in HELDOUT[: cfg["margin_tuning"]["images"]]))
proj = {"crop-margin search": tune_c * len(cfg["margin_tuning"]["margins"]) * s_crop,
        "crop verifier, all candidates": (n_h.sum() + n_t.sum()) * s_crop,
        "global-view ablation": uniq_all * s_glob,
        "VCD baseline (2 passes / label)": 2 * uniq_all * s_glob}
print(f"verifier: {s_crop * 1000:.0f} ms per crop (batch {cfg['verifier']['batch_size']}) | {s_glob * 1000:.0f} ms per (image, label) on the full image")
print(pd.Series({k: f"{v / 60:.1f} min" for k, v in proj.items()}, name="projected GPU time").to_string())
print(f"-> these four stages: ~{sum(proj.values()) / 60:.0f} min. If that is more than you want, lower coco.test_size / "
      "detector.max_detections in configs/default.yaml and delete results_fast/ (see docs/FAST_RUN.md).")
'''),
    md("## 2 · Verifier Agent — crop-margin search (held-out only)\nCriterion: AUROC of P(yes) for separating true from false candidates. The margin is the only Verifier hyper-parameter."),
    code(r'''
tune_ids = HELDOUT[: cfg["margin_tuning"]["images"]]
rows = []
with stage("2 crop-margin search"):
    for m in cfg["margin_tuning"]["margins"]:
        ver = run_verification_stage(verifier, H.det, tune_ids, paths.ver_cache(cfg, "heldout", "crop", margin=m), cfg,
                                     coco.image, view="crop", margin=m, desc=f"margin {m}")
        y = np.concatenate([H.labels[i] for i in tune_ids]); p = np.concatenate([np.asarray(ver[i]["p_yes"]) for i in tune_ids])
        keep = y != -1; y, p = y[keep], p[keep]
        rows.append({"margin": m, "auroc": auroc(p, y == TP), "fp_rejected": float((p[y == FP] <= .5).mean()),
                     "tp_over_rejected": float((p[y == TP] <= .5).mean()), "yes_rate": float((p > .5).mean()), "n_candidates": int(len(y))})
margin_df = pd.DataFrame(rows); margin_df.to_csv(results_dir(cfg, "verifier") / "margin_tuning.csv", index=False)
best_margin = float(margin_df.loc[margin_df.auroc.idxmax(), "margin"])
save_selected(cfg, crop_margin=best_margin)
print("selected crop margin (held-out):", best_margin)
margin_df.plot(x="margin", y=["auroc", "fp_rejected", "tp_over_rejected"], marker="o", grid=True, title="Crop margin search (held-out)")
plt.savefig(results_dir(cfg, "figures") / "margin_search.png", dpi=130, bbox_inches="tight"); plt.show()
margin_df
'''),
    md("## 3 · Verifier Agent on every candidate (crop-and-zoom) and the global-view ablation"),
    code(r'''
with stage("3 verifier, crop view, held-out + test"):
    ver_h = run_verification_stage(verifier, H.det, HELDOUT, paths.ver_cache(cfg, "heldout", "crop"), cfg, coco.image, view="crop", desc="verify held-out")
    ver_t = run_verification_stage(verifier, T.det, TEST, paths.ver_cache(cfg, "test", "crop"), cfg, coco.image, view="crop", desc="verify test")
'''),
    code(r'''
# Same verifier, same question, but it sees the WHOLE image: isolates the contribution of evidence separation (the crop).
with stage("3b global-view ablation"):
    gl_h = run_verification_stage(verifier, H.det, HELDOUT, paths.ver_cache(cfg, "heldout", "full"), cfg, coco.image, view="full", desc="global held-out")
    gl_t = run_verification_stage(verifier, T.det, TEST, paths.ver_cache(cfg, "test", "full"), cfg, coco.image, view="full", desc="global test")

def diag(ver, data, name):
    # Verifier diagnostics on a split: AUROC, share of FPs rejected, share of TPs wrongly rejected, yes-rate.
    y = np.concatenate([data.labels[i] for i in data.ids]); p = np.concatenate([np.asarray(ver[i]["p_yes"]) for i in data.ids])
    k = y != -1; y, p = y[k], p[k]
    return {"view": name, "auroc": auroc(p, y == TP), "fp_rejected@0.5": float((p[y == FP] <= .5).mean()),
            "tp_over_rejected@0.5": float((p[y == TP] <= .5).mean()), "yes_rate": float((p > .5).mean())}, (y, p)
d_crop, (y, p_crop) = diag(ver_t, T, "crop (proposed)"); d_full, (_, p_full) = diag(gl_t, T, "full image")
diag_df = pd.DataFrame([d_crop, d_full]); diag_df.to_csv(results_dir(cfg, "verifier") / "verifier_diagnostics_test.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, (p, name) in zip(axes, [(p_crop, "crop view"), (p_full, "full-image view")]):
    ax.hist(p[y == TP], bins=30, alpha=.6, label="true detections", density=True); ax.hist(p[y == FP], bins=30, alpha=.6, label="false positives", density=True)
    ax.set(title=f"Verifier P(yes), {name}", xlabel="P(yes)"); ax.legend()
plt.savefig(results_dir(cfg, "figures") / "pyes_hist_test.png", dpi=130, bbox_inches="tight"); plt.show()
diag_df
'''),
    md("## 4 · Decision Agent (held-out → test)\n`accept = (c ≥ τ_high) ∨ (v = yes ∧ c ≥ τ_low)`. The grid is swept on held-out; the selected configuration is applied **unchanged** to test."),
    code(r'''
with stage("4 decision agent"):
    MH, MT = available_methods(cfg, H), available_methods(cfg, T)
    base_rows, dvd_rows = sweep(DET_ONLY, MH[DET_ONLY], H, cfg), sweep(PROPOSED, MH[PROPOSED], H, cfg)
    base_best, dvd_best = operating_point(base_rows), operating_point(dvd_rows)
    dvd_match = operating_point(dvd_rows, "matched_recall", base_best["recall"])
    save_selected(cfg, tau_low=dvd_best["tau_low"], tau_high=dvd_best["tau_high"], matched_tau_low=dvd_match["tau_low"],
                  matched_tau_high=dvd_match["tau_high"], detector_only_tau=base_best["tau"])
    res = {"Detector only (tau*)": evaluate_row(MT[DET_ONLY], T, base_best, n_boot=1000),
           "DVD best-F1": evaluate_row(MT[PROPOSED], T, dvd_best, n_boot=1000),
           "DVD matched-recall": evaluate_row(MT[PROPOSED], T, dvd_match, n_boot=1000)}
fp0 = res["Detector only (tau*)"]["fp"]
tab = pd.DataFrame([{"config": k, **{c: v.get(c) for c in ["tau", "tau_low", "tau_high", "precision", "recall", "f1", "fp", "fp_per_image", "n_verified"]},
                     "FP reduction vs detector": 1 - v["fp"] / fp0, "F1 95% CI": v["ci"]["f1"]} for k, v in res.items()])
out = results_dir(cfg, "pipeline"); tab.to_csv(out / "test_operating_points.csv", index=False)
br = pd.DataFrame([{"config": n, **decision_breakdown(T.labels, {i: T.ev.scores(i) for i in T.ids}, res[n]["_masks"], res[n]["tau_low"])}
                   for n in ["DVD best-F1", "DVD matched-recall"]]); br.to_csv(out / "decision_breakdown_test.csv", index=False)
print("selected on held-out: DVD", {k: dvd_best[k] for k in ("tau_low", "tau_high")}, "| matched-recall", {k: dvd_match[k] for k in ("tau_low", "tau_high")}, "| detector tau*", base_best["tau"])
display(tab); display(br)
'''),
    code(r'''
sw = pd.DataFrame(dvd_rows); heat = sw.pivot(index="tau_low", columns="tau_high", values="f1")
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
im = ax[0].imshow(heat, cmap="viridis", aspect="auto"); ax[0].set_xticks(range(len(heat.columns)), heat.columns); ax[0].set_yticks(range(len(heat.index)), heat.index)
ax[0].set(xlabel="tau_high", ylabel="tau_low", title="Held-out F1 over the decision grid"); plt.colorbar(im, ax=ax[0])
from dvd.viz import plot_pr  # plotting helpers (detections overlay, PR frontier, crop gallery)
rows_t = {DET_ONLY: sweep(DET_ONLY, MT[DET_ONLY], T, cfg), PROPOSED: sweep(PROPOSED, MT[PROPOSED], T, cfg)}
plot_pr(rows_t, ax=ax[1], highlight={"DVD best-F1": res["DVD best-F1"], "Detector tau*": res["Detector only (tau*)"]}, title="Test PR frontier (selected on held-out)")
plt.savefig(results_dir(cfg, "figures") / "pr_frontier_pipeline_test.png", dpi=130, bbox_inches="tight"); plt.show()
'''),
    md("### Ablation: LLM aggregator instead of the rule\nThe verifier's own language model (text only) receives the detector confidence, the verdict and P(yes) and answers accept/reject; compared with the rule on a held-out subset."),
    code(r'''
from dvd.agents.decision import LLMAggregatorAgent, rule_accept  # Decision Agent: the accept/reject rule and the optional LLM aggregator
with stage("4b LLM aggregator ablation"):
    agg = LLMAggregatorAgent(verifier)
    sub = HELDOUT[: cfg["decision"]["llm_aggregator_images"]]
    ev_s = DetectionEvaluator({i: H.det[i] for i in sub}, {i: H.gts[i] for i in sub}, cfg["coco"]["iou_threshold"])
    v, p = MH[PROPOSED]["verdicts"], MH[PROPOSED]["p_yes"]; tl, th = dvd_best["tau_low"], dvd_best["tau_high"]
    m_llm = {}
    for i in sub:
        c = ev_s.scores(i); m = c >= th; band = np.where((c >= tl) & (c < th))[0]
        if len(band): m[band] = agg.decide([H.det[i]["labels"][k] for k in band], c[band], v[i][band], p[i][band])
        m_llm[i] = m
    m_rule = {i: rule_accept(ev_s.scores(i), v[i], tl, th) for i in sub}
llm_df = pd.DataFrame([{"decision": "rule (proposed)", **ev_s.evaluate(m_rule)}, {"decision": "LLM aggregator", **ev_s.evaluate(m_llm)}])
llm_df.to_csv(results_dir(cfg, "pipeline") / "llm_vs_rule_heldout_subset.csv", index=False)
llm_df[["decision", "precision", "recall", "f1", "fp", "tp"]]
'''),
    md("## 5 · Baseline ii — Woodpecker-style single agent\nThe same VLM sees the **full image** plus the numbered candidate list and decides keep/reject for every item in one call: validation and decision are fused, evidence is global. (A re-implementation with the same models, not the official GPT-3.5/BLIP-2 code — see `docs/DATASETS_AND_BASELINES.md`.)"),
    code(r'''
from dvd.baselines.woodpecker_style import run_woodpecker_stage  # baseline (ii): Woodpecker-style single agent
with stage("5 Woodpecker-style baseline"):
    wp_h = run_woodpecker_stage(verifier, H.det, HELDOUT, paths.woodpecker_cache(cfg, "heldout"), cfg, coco.image, "woodpecker held-out")
    wp_t = run_woodpecker_stage(verifier, T.det, TEST, paths.woodpecker_cache(cfg, "test"), cfg, coco.image, "woodpecker test")
print("answer parse rate (test):", round(float(np.mean([r["parse_ok"] for r in wp_t.values()])), 3))
print("example raw answer:", repr(next(r["raw"] for r in wp_t.values() if r["raw"])[:120]))
'''),
    md("## 6 · Baseline iii — Visual Contrastive Decoding (VCD)\nFor every (image, candidate label) the VLM is asked on the full image, contrasting its logits with those from a diffusion-noised copy (α = 1, β = 0.1, noise step 500). The same passes give the plain (non-contrastive) global answer."),
    code(r'''
from dvd.baselines.vcd import run_vcd_stage, vcd_yes_no  # baseline (iii): Visual Contrastive Decoding
with stage("6 VCD baseline"):
    run_vcd_stage(verifier, H.det, HELDOUT, paths.vcd_cache(cfg, "heldout"), cfg, coco.image, "vcd held-out")
    run_vcd_stage(verifier, T.det, TEST, paths.vcd_cache(cfg, "test"), cfg, coco.image, "vcd test")
'''),
    md("## 7 · The Verifier Agent in isolation on POPE\nPOPE (random / popular / adversarial; 50 % yes). Plain yes/no and VCD yes/no on the first 300 questions of each setting. Reported: accuracy, precision, recall, F1 and the yes-rate (over-assertion)."),
    code(r'''
from dvd.data.pope import iter_pope  # POPE benchmark loader
from dvd.metrics import pope_metrics  # matching, metrics, sweeps, AUROC, POPE scores
from dvd.cropping import fit_max_side  # crop-and-zoom / image resizing helpers
from tqdm.auto import tqdm  # progress bars
SIDE, MAX_N = cfg["verifier"]["full_image_max_side"], cfg["pope"]["max_per_setting"]

def run_pope(setting, method):
    # Run (or resume) one POPE setting with plain or VCD yes/no and return its metrics.
    cache = JsonlCache(paths.pope_cache(cfg, setting, method), key="idx"); batch = []
    def flush():
        # Score the collected batch, write one cache line per question, and empty the batch.
        imgs = [fit_max_side(b["image"], SIDE) for b in batch]; qs = [b["question"].strip() + " Answer yes or no." for b in batch]
        with Timer() as t:
            pp = verifier.yes_no(imgs, qs) if method == "plain" else vcd_yes_no(verifier, imgs, qs, cfg["vcd"]["alpha"], cfg["vcd"]["beta"], cfg["vcd"]["noise_step"])[0]
        for b, pi in zip(batch, pp):
            cache.add({"idx": b["idx"], "question": b["question"], "label": b["label"], "p_yes": float(pi), "pred": "yes" if pi > .5 else "no", "time_s": t.elapsed / len(batch)})
        batch.clear()
    for item in tqdm(iter_pope(cfg, setting, MAX_N), desc=f"POPE {setting} [{method}]", total=MAX_N or 3000):
        if item["idx"] in cache: continue
        batch.append(item)
        if len(batch) == 8: flush()
    if batch: flush()
    recs = list(cache.records().values())
    return {"setting": setting, "method": method, **pope_metrics([r["pred"] for r in recs], [r["label"] for r in recs]), "s_per_question": float(np.mean([r["time_s"] for r in recs]))}

with stage("7 POPE"):
    pope_df = pd.DataFrame([run_pope(s, m) for s in cfg["pope"]["settings"] for m in ("plain", "vcd")])
pope_df.to_csv(results_dir(cfg, "pope") / "pope_results.csv", index=False)
pope_df[["setting", "method", "n", "accuracy", "precision", "recall", "f1", "yes_rate", "s_per_question"]]
'''),
    md("## 8 · Main comparison on COCO (all methods, identical models and candidates)\nOperating point of every method is selected on held-out (best-F1, and matched-recall = fewest FPs at the detector's recall), then evaluated unchanged on test. Intervals are image-level bootstrap; the *paired* bootstrap tests every method against the detector-only baseline and the proposed pipeline against every other method on the same resampled images."),
    code(r'''
with stage("8 main comparison"):
    MH, MT = available_methods(cfg, H), available_methods(cfg, T)
    print("methods compared:", list(MT))
    main, sweeps_h, sweeps_t, base_h = main_comparison(cfg, H, T, MH, MT, n_boot=1000)
    vs = pd.concat([paired_vs_proposed(cfg, T, MT, sweeps_h, base_h, c, 1000) for c in ("best_f1", "matched_recall")], ignore_index=True)
tdir = results_dir(cfg, "tables"); main.to_csv(tdir / "main_comparison.csv", index=False); vs.to_csv(tdir / "proposed_vs_each_method.csv", index=False)
fmt = main.copy()
for c in ["precision", "recall", "f1", "fp_reduction", "over_rejection_rate", "d_recall", "d_f1"]: fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
for c in ["f1_ci", "fp_reduction_ci", "d_recall_ci", "d_f1_ci"]: fmt[c] = fmt[c].map(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
(tdir / "main_comparison.md").write_text(fmt[["criterion", "method", "precision", "recall", "f1", "f1_ci", "fp_per_image", "fp_reduction", "fp_reduction_ci", "d_recall_ci", "over_rejection_rate", "total_latency_s", "total_invocations"]].to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
cols = ["method", "precision", "recall", "f1", "fp_per_image", "fp_reduction", "fp_reduction_ci", "d_recall_ci", "over_rejection_rate", "total_latency_s"]
print("=== operating point: best F1 (selected on held-out, applied to test) ==="); display(main[main.criterion == "best_f1"][cols])
print("=== operating point: matched recall (fewest FPs at the detector's recall) ==="); display(main[main.criterion == "matched_recall"][cols])
'''),
    code(r'''
sel = {n: main[(main.criterion == "best_f1") & (main.method == n)].iloc[0] for n in MT}
fig, ax = plt.subplots(figsize=(8, 6)); plot_pr(sweeps_t, ax=ax, highlight={n: sel[n] for n in (DET_ONLY, PROPOSED)}, title="All methods, test split")
plt.savefig(results_dir(cfg, "figures") / "pr_frontier_all_methods_test.png", dpi=130, bbox_inches="tight"); plt.show()
print("=== proposed pipeline vs each method (paired; + = proposed is better) ==="); display(vs[["criterion", "versus", "d_f1", "d_f1_ci", "fp_reduction", "fp_reduction_ci", "d_recall", "p_f1_gt0"]])
'''),
    md("## 9 · Accuracy versus cost\nCost of every (τ_low, τ_high) configuration, cost of each method at its operating point, and a **measured** online run of the gated pipeline (only the uncertain band is verified) to validate the cost model."),
    code(r'''
fp_base = evaluate_row(MT[DET_ONLY], T, {"tau": cfg["selected"]["detector_only_tau"]})["fp"]
cost_df = pd.DataFrame([{**r, **method_cost(MT[PROPOSED], T, r["n_verified"]), "fp_reduction": 1 - r["fp"] / fp_base} for r in sweep(PROPOSED, MT[PROPOSED], T, cfg)])
cdir = results_dir(cfg, "cost"); cost_df.to_csv(cdir / "pipeline_config_costs_test.csv", index=False)
mc = main[main.criterion == "best_f1"][["method", "f1", "fp_reduction", "det_latency_s", "added_latency_s", "total_latency_s", "det_invocations", "verify_invocations", "total_invocations"]].copy()
mc["fps"] = 1 / mc.total_latency_s; mc["fp_reduction_per_added_s"] = mc.fp_reduction / mc.added_latency_s.replace(0, np.nan); mc.to_csv(cdir / "method_costs_test.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
sc = ax[0].scatter(cost_df.added_latency_s, cost_df.fp_reduction, c=cost_df.recall, cmap="viridis", s=22); plt.colorbar(sc, ax=ax[0], label="recall")
ax[0].set(xlabel="added latency per image (s)", ylabel="FP reduction vs detector tau*", title="FP removed vs added latency (all configurations)")
ax[1].scatter(cost_df.verify_invocations, cost_df.f1, s=22); ax[1].set(xlabel="verifier calls per image", ylabel="F1", title="F1 vs verifier calls")
for a in ax: a.grid(alpha=.3)
plt.savefig(results_dir(cfg, "figures") / "cost_pipeline_tradeoff.png", dpi=130, bbox_inches="tight"); plt.show()
mc
'''),
    code(r'''
tl, th = cfg["selected"]["tau_low"], cfg["selected"]["tau_high"]
pipe = DetectVerifyDecide(detector, verifier, RuleDecisionAgent(tl, th), cfg)
ids = TEST[: cfg["cost"]["online_latency_images"]]; pipe(coco.image(ids[0]))      # warm-up
online = []
with stage("9 online latency measurement"):
    for iid in tqdm(ids):
        r = pipe(coco.image(iid))
        online.append({"det_s": r["det_time_s"], "verify_s": r["verify_time_s"], "decide_s": r["decision_time_s"], "total_s": r["total_time_s"],
                       "verify_calls": r["verify_invocations"], "candidates": len(r["scores"])})
online = pd.DataFrame(online)
est = cost_df[(cost_df.tau_low == tl) & (cost_df.tau_high == th)].iloc[0]
cost_summary = pd.DataFrame({"measured (online, gated)": [online.det_s.mean(), online.verify_s.mean(), online.total_s.mean(), online.verify_calls.mean()],
                             "estimated (cost model)": [est.det_latency_s, est.added_latency_s, est.total_latency_s, est.verify_invocations]},
                            index=["detector s/img", "verifier s/img", "total s/img", "verifier calls/img"])
cost_summary.to_csv(cdir / "online_vs_estimated.csv")
print(f"gated pipeline verifies {online.verify_calls.mean():.1f} of {online.candidates.mean():.1f} candidates per image "
      f"({100 * online.verify_calls.mean() / online.candidates.mean():.0f} %) | {1 / online.total_s.mean():.2f} FPS vs detector alone {1 / online.det_s.mean():.2f} FPS "
      f"(x{online.total_s.mean() / online.det_s.mean():.2f} latency) | decision agent {online.decide_s.mean() * 1e3:.3f} ms/img | worst case {online.total_s.max():.2f} s (bounded: single fixed pass)")
cost_summary
'''),
    md("## 10 · Error analysis — where verification helps and where it hurts\nPer category, object size, detector-confidence bin, image brightness, co-occurrence-driven false positives, and galleries of the crops exactly as the verifier saw them."),
    code(r'''
from dvd.analysis import candidate_table, mean_brightness, cooccurrence_topk  # per-candidate error-analysis tables and co-occurrence statistics
from dvd.experiments import masks_for  # methods over identical candidates: sweeps, operating points, evaluation, costs
from dvd.viz import crop_gallery  # plotting helpers (detections overlay, PR frontier, crop gallery)
with stage("10 error analysis"):
    M = MT[PROPOSED]; tl, th = cfg["selected"]["tau_low"], cfg["selected"]["tau_high"]
    accept = masks_for(M, T, {"tau_low": tl, "tau_high": th})
    cooc = {int(k): set(v) for k, v in cooccurrence_topk(coco, k=10).items()}
    bright = {i: mean_brightness(coco.image(i)) for i in TEST}
    df = pd.DataFrame(candidate_table(T.det, T.labels, accept, T.gts, M["verdicts"], M["p_yes"], cooc, bright))
    df = df[(df.score >= tl) & (df.outcome != "ignore")]
    df["band"] = np.where(df.score >= th, "auto-accept (c >= tau_high)", "verified band")
    df["conf_bin"] = pd.cut(df.score, [0, .3, .4, .5, .6, .8, 1.0]); df["brightness_bin"] = pd.qcut(df.brightness, 3, labels=["dark", "medium", "bright"])
def rates(g):
    # Per group: candidate count, FP/TP counts, share of FPs removed and share of TPs lost.
    fp, tp = g[g.is_fp], g[g.is_tp]
    return pd.Series({"candidates": len(g), "FP": len(fp), "TP": len(tp), "FP removed (correct rejection)": (~fp.accepted).mean() if len(fp) else np.nan,
                      "TP lost (over-rejection)": (~tp.accepted).mean() if len(tp) else np.nan})
edir = results_dir(cfg, "error_analysis"); print(df.outcome.value_counts().to_dict())
for col in ["size", "conf_bin", "brightness_bin", "cooccurrence_fp", "band"]:
    t_ = df.groupby(col, observed=True).apply(rates); t_.to_csv(edir / f"by_{col}.csv"); print(f"\n=== by {col} ==="); display(t_)
by_cat = df.groupby("label").apply(rates).sort_values("FP", ascending=False); by_cat.to_csv(edir / "by_category.csv"); display(by_cat.head(12))
'''),
    code(r'''
for oc in ["correct_rejection", "omission", "over_rejection"]:
    pool = df[(df.outcome == oc) & (df.band == "verified band")]
    if not len(pool): print("no", oc, "in this sample"); continue
    s = pool.sample(min(12, len(pool)), random_state=0)
    crop_gallery([{"image_id": r.image_id, "box": r.box, "label": r.label, "score": r.score, "p_yes": r.p_yes} for r in s.itertuples()],
                 coco.image, cfg["verifier"]["crop_margin"], title=oc.replace("_", " "))
    plt.savefig(results_dir(cfg, "figures") / f"gallery_{oc}.png", dpi=110, bbox_inches="tight"); plt.show()
'''),
    md("## 11 · Monitoring conditions (low light, motion blur, low resolution, JPEG)\nThe thresholds chosen on **clean** held-out data are applied unchanged to degraded test images (COCO labels stay valid) — a proxy for surveillance / traffic footage."),
    code(r'''
from dvd.degrade import degrade  # synthetic low-light / blur / low-res / JPEG degradations
from dvd.experiments import cached  # methods over identical candidates: sweeps, operating points, evaluation, costs
SUB = TEST[: cfg["robustness"]["images"]]; sel_p = cfg["selected"]; rows = []
with stage("11 robustness"):
    for cond in ["clean"] + cfg["robustness"]["conditions"]:
        split, tag = ("test", "") if cond == "clean" else ("robust", f"_{cond}")
        fn = coco.image if cond == "clean" else (lambda i, c=cond: degrade(coco.image(i), c, i))
        det = run_detection_stage(detector, SUB, paths.det_cache(cfg, split, tag), fn, f"detect {cond}")
        run_verification_stage(verifier, det, SUB, paths.ver_cache(cfg, split, "crop", tag=tag), cfg, fn, view="crop", desc=f"verify {cond}")
        D = SplitData(cfg, coco, split, SUB, det_tag=tag); v = cached(paths.ver_cache(cfg, split, "crop", tag=tag), SUB)
        base = evaluate_row({"kind": "none"}, D, {"tau": sel_p["detector_only_tau"]})
        dvd = evaluate_row({"kind": "rule", "verdicts": {i: np.asarray(r["verdict"], bool) for i, r in v.items()}}, D, {"tau_low": sel_p["tau_low"], "tau_high": sel_p["tau_high"]})
        b = decision_breakdown(D.labels, {i: D.ev.scores(i) for i in SUB}, dvd.pop("_masks"), sel_p["tau_low"]); base.pop("_masks")
        rows.append({"condition": cond, "det_precision": base["precision"], "det_recall": base["recall"], "det_fp_per_img": base["fp_per_image"],
                     "dvd_precision": dvd["precision"], "dvd_recall": dvd["recall"], "dvd_fp_per_img": dvd["fp_per_image"],
                     "fp_reduction": 1 - dvd["fp"] / max(base["fp"], 1), "over_rejection_rate": b["over_rejection_rate"]})
rob = pd.DataFrame(rows); rob.to_csv(results_dir(cfg, "robustness") / "degradations.csv", index=False); rob
'''),
    md("## 12 · Summary: what the results show (and what they do not)\nEvery statement is computed from the tables above; a claim is only marked *supported* when its 95 % interval excludes zero. With 300 test images some differences will be too small to establish — the notebook says so instead of hiding it."),
    code(r'''
def verdict(ci, point):
    # Plain-language verdict from a 95 % interval: supported / opposite direction / not established.
    return "supported (95% CI excludes 0)" if ci[0] > 0 else ("opposite direction (CI below 0)" if ci[1] < 0 else "not established at this sample size (CI includes 0)")
f2 = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"
L = ["# Detect-Verify-Decide - results summary (fast profile)", "",
     f"{len(HELDOUT)} held-out images (tuning only), {len(TEST)} test images (all reported numbers), seed {cfg['seed']}. Selected on held-out: "
     f"crop margin {cfg['selected']['crop_margin']}, tau_low {cfg['selected']['tau_low']}, tau_high {cfg['selected']['tau_high']}. No model was trained.", ""]
mm = main[main.criterion == "matched_recall"].set_index("method"); bb = main[main.criterion == "best_f1"].set_index("method")
p, d = mm.loc[PROPOSED], mm.loc[DET_ONLY]
L += ["## Objective 2 - does verification reduce false positives at the same recall?",
      f"- Matched recall: detector-only {d.fp_per_image:.2f} FP/img (precision {d.precision:.3f}, recall {d.recall:.3f}) -> pipeline {p.fp_per_image:.2f} FP/img "
      f"(precision {p.precision:.3f}, recall {p.recall:.3f}).",
      f"- False positives removed: {p.fp_reduction:.1%} {f2(p.fp_reduction_ci)}; recall change {p.d_recall:+.3f} {f2(p.d_recall_ci)}"
      + ("" if p.recall_matched else "  (NOTE: no configuration reached the detector's recall on held-out; best-F1 point shown)") + f" -> {verdict(p.fp_reduction_ci, p.fp_reduction)}.",
      f"- Best-F1: detector-only F1 {bb.loc[DET_ONLY].f1:.3f} -> pipeline {bb.loc[PROPOSED].f1:.3f} (paired dF1 {bb.loc[PROPOSED].d_f1:+.3f} {f2(bb.loc[PROPOSED].d_f1_ci)}) -> {verdict(bb.loc[PROPOSED].d_f1_ci, bb.loc[PROPOSED].d_f1)}.",
      f"- Of the detector's false positives (reference pool c >= tau_low) the decision stage rejects {bb.loc[PROPOSED].fp_removed_rate:.1%}; it wrongly rejects {bb.loc[PROPOSED].over_rejection_rate:.1%} of true detections.", ""]
L += ["## Objective 3 - is the pipeline structure better than the baselines and ablations? (same models, same candidates)"]
for crit in ("best_f1", "matched_recall"):
    for r in vs[vs.criterion == crit].itertuples():
        L.append(f"- [{crit}] vs {r.versus}: dF1 {r.d_f1:+.3f} {f2(r.d_f1_ci)} -> {verdict(r.d_f1_ci, r.d_f1)}; recall change {r.d_recall:+.3f}; false positives of the pipeline relative to it {-r.fp_reduction:+.1%} {f2((-r.fp_reduction_ci[1], -r.fp_reduction_ci[0]))}")
L += ["", "## Objective 2 - the verifier in isolation (POPE, first 300 questions per setting)", pope_df[["setting", "method", "accuracy", "precision", "recall", "f1", "yes_rate"]].round(3).to_markdown(index=False), ""]
L += ["## Objective 4 - cost",
      f"- Online gated pipeline: {1 / online.total_s.mean():.2f} FPS vs detector alone {1 / online.det_s.mean():.2f} FPS; verifier called on {online.verify_calls.mean():.1f} of {online.candidates.mean():.1f} candidates per image; "
      f"worst-case {online.total_s.max():.2f} s per image (single fixed pass). Cost model {est.total_latency_s:.2f} s/img vs measured {online.total_s.mean():.2f} s/img.", ""]
L += ["## Monitoring conditions (thresholds from clean held-out data)", rob.round(3).to_markdown(index=False), "",
      "## Caveats", f"- Small samples ({len(TEST)} test images): read the intervals, not just the point estimates. Run `DVD_CONFIG=full` (docs/FAST_RUN.md) for the full protocol.",
      "- Woodpecker-style is a re-implementation with the same VLM/detector, not the official GPT-3.5 + BLIP-2 system. VCD is applied to yes/no existence queries per (image, label).",
      f"- Crops are zoomed to {cfg['verifier']['crop_target_size']} px and global views are {cfg['verifier']['full_image_max_side']} px in this profile (speed).", ""]
(results_dir(cfg, "") / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
'''),
    code(r'''
sync()
print("stage times (min):", {k: round(v / 60, 1) for k, v in TIMES.items()})
print(f"TOTAL this session: {(time.time() - T0) / 60:.0f} min | results in {RESULTS_DRIVE}")
if IN_KAGGLE:
    print("Kaggle keeps /kaggle/working only if you press Save Version -> Save & Run All (Commit); then download SUMMARY.md and the tables from the Output tab.")
'''),
]
