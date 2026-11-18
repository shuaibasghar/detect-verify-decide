# Notebook Reference

### run_all_fast.ipynb (start here)
- **Goal:** every experiment of the proposal in **one session (≈2 h target on a T4)**, models loaded once.
- **Does:** setup and data → detector → speed check (measures your GPU and projects the long stages) → crop-margin search → verifier on all candidates (crop view) → global-view ablation → Decision Agent sweep and test evaluation → LLM-aggregator ablation → Woodpecker-style baseline → VCD baseline → POPE → main comparison with paired-bootstrap CIs → cost and online latency → error analysis → monitoring conditions → `SUMMARY.md`.
- **Writes:** everything under `results_fast/` (Drive copy refreshed after every step; resumable).
- **Details and the claim-to-evidence map:** [FAST_RUN.md](FAST_RUN.md). The cells are written in `tools/nb_run_all_fast.py`.

The numbered notebooks below are the same experiments split into separate, individually resumable steps. They use the same
`configs/default.yaml` (fast profile) unless `DVD_CONFIG=full` is set.

---

All numbered notebooks start with the same three cells:

- **Bootstrap**: mounts Drive, finds the project, fetches the latest `src/` and `configs/` from GitHub (if the Drive copy is a git clone), adds `src/` to the path.
- **Install**: installs `requirements.txt`.
- **Config**: loads `configs/default.yaml` and `results/tuning/selected.json`.

Notebooks that use COCO add a fourth cell that loads the COCO data and the frozen splits. All heavy logic lives in `src/dvd/`, so the notebooks read as experiment scripts.

> `tools/build_notebooks.py` regenerates every notebook from one source file (`python tools/build_notebooks.py notebooks`).
> Regenerating **overwrites** any edits made directly in the `.ipynb` files. Either edit the builder, or stop using it once you start editing notebooks by hand.

---

### 00_setup_and_smoke_test.ipynb
- **Goal:** prove the environment works before spending GPU hours.
- **Does:** checks GPU and library versions, then runs the detector, the crop verifier, a hallucination probe (questions about absent objects) and the online pipeline on one COCO image fetched by URL.
- **Writes:** nothing.
- **Pass criteria:** the checklist at the bottom of the notebook.

### 01_data_preparation.ipynb
- **Goal:** datasets ready, splits frozen.
- **Does:**
  - downloads COCO val2017 and its annotations;
  - creates the seeded held-out / test split;
  - computes category and split statistics;
  - inspects all three POPE settings;
  - computes the top-10 co-occurrence partners of each class.
- **Writes:** `splits.json`, `data/category_counts.csv`, `pope_stats.csv`, `cooccurrence_top10.json` (under `results_fast/`, or `results/` with `DVD_CONFIG=full`).
- **Download:** the annotation file and only the images of the two splits (not the 778 MB zip).

### 02_detector_agent_baseline.ipynb — baseline (i)
- **Goal:** cache the detector candidates and evaluate the single-pass detector.
- **Does:**
  - go / no-go on 50 images;
  - full held-out and test detection;
  - τ sweep (τ\* chosen on held-out);
  - COCO mAP sanity check;
  - detector latency and FPS;
  - example figures.
- **Writes:** `results/detections/*.jsonl`, `results/detector/detector_only_sweep_*.csv`, `summary.json`, plus `detector_only_tau` in `selected.json`.

### 03_verifier_agent_crop_verification.ipynb
- **Goal:** the Verifier Agent, plus its crop-margin hyper-parameter.
- **Does:**
  - margin search on 200 held-out images (AUROC criterion), saving `crop_margin`;
  - crop verification of all held-out and test candidates;
  - global-view ablation;
  - optional open-answer (label-blind) mode;
  - P(yes) histograms and diagnostics.
- **Writes:** `results/verifier/*.jsonl`, `margin_tuning.csv`, `verifier_diagnostics_test.csv`.

### 04_verifier_pope_eval.ipynb
- **Goal:** evaluate the verifier in isolation on the POPE protocol (objective 2).
- **Does:** plain yes/no and VCD yes/no on random, popular and adversarial, reporting accuracy, precision, recall, F1 and yes-rate.
- **Writes:** `results/pope/*.jsonl`, `pope_results.csv`.
- **Note:** independent of NB02/03, so it can run in a parallel session.

### 05_decision_agent_and_pipeline.ipynb
- **Goal:** the Decision Agent and the full pipeline.
- **Does:**
  - (τ_low, τ_high) sweep and F1 heat-map on held-out;
  - best-F1 and matched-recall selection, saved to `selected.json`;
  - test evaluation with bootstrap CIs;
  - decision breakdown (correct rejection, omission, over-rejection);
  - PR frontier;
  - optional LLM aggregator;
  - online demo.
- **Writes:** `results/pipeline/*.csv`, `results/figures/pr_frontier_pipeline_test.png`.

### 06_baseline_woodpecker_style.ipynb — baseline (ii)
- **Goal:** single-agent self-verification (no role separation, global evidence).
- **Does:** one VLM call per image with the candidate list, answer parsing (parse rate logged), τ sweep, test evaluation.
- **Writes:** `results/baselines/woodpecker_*.{jsonl,csv,json}`.

### 07_baseline_vcd.ipynb — baseline (iii)
- **Goal:** decoding-time mitigation on the same detections.
- **Does:** shows the noised image, then runs VCD and plain yes/no per (image, label), with a τ sweep and test evaluation.
- **Writes:** `results/baselines/vcd_*.{jsonl,json}`.

### 08_main_comparison_coco.ipynb
- **Goal:** the main results table (objectives 2 and 3).
- **Does:** every method with a complete cache, selected on held-out (best-F1 and matched-recall), evaluated on test. Reports CIs, FP reduction against detector-only, the breakdown and the cost of each method.
- **Writes:** `results/tables/main_comparison.{csv,md}`, `results/figures/pr_frontier_all_methods_test.png`.

### 09_cost_analysis.ipynb
- **Goal:** the accuracy-versus-cost trade-off (objective 4).
- **Does:**
  - cost of every pipeline configuration;
  - cost of each method at its operating point (FPS, FP reduction per added second);
  - **online measurement** on 100 images, compared with the cost-model estimate.
- **Writes:** `results/cost/*.csv`, `results/figures/cost_pipeline_tradeoff.png`.

### 10_error_analysis.ipynb
- **Goal:** where verification helps or hurts (contribution 4).
- **Does:** breakdowns by category, size, confidence bin, brightness tercile, co-occurrence FP and band, plus crop galleries for each outcome.
- **Writes:** `results/error_analysis/*.csv`, `results/figures/error_by_category.png`, `gallery_*.png`.

### 11_optional_robustness_monitoring.ipynb
- **Goal:** the monitoring-conditions extension.
- **Does:**
  - four synthetic degradations on 200 test images, using thresholds from clean held-out data;
  - qualitative run on your own frames in `data/custom_images/`.
- **Writes:** `results/robustness/degradations.csv`.

### 12_streamlit_demo.ipynb
- **Goal:** launch the Streamlit demo UI from Colab (see [UI_GUIDE.md](UI_GUIDE.md)).
- **Does:** starts `app/streamlit_app.py` in the background and opens a temporary Cloudflare tunnel URL. The last cell stops both.
- **Writes:** nothing in `results/` (logs go to `/content/streamlit.log` and `/content/cloudflared.log`).
- **Note:** the tunnel URL is public until you stop the notebook's last cell.

---

## Flags you may want to change inside notebooks

| Notebook | Flag | Default | Effect |
|---|---|---|---|
| 03 | `RUN_GLOBAL_ABLATION` | `True` | Global-view verifier ablation (doubles NB03 time) |
| 03 | `RUN_OPEN_MODE` | `False` | Label-blind open-answer verifier on the tuning subset |
| 04 | `RUN_VCD` | `True` | VCD on POPE |
| 05 | `RUN_LLM_AGGREGATOR` | `False` | LLM aggregator vs rule |
