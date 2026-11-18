# Fast run: the whole proposal in one Colab session (~2 h target)

`notebooks/run_all_fast.ipynb` runs every experiment of the proposal in **one session, with the models loaded once**:
detector-only baseline, the three-agent pipeline, the Woodpecker-style and VCD baselines, the ablations, POPE, the cost
trade-off with an online latency measurement, the error analysis and the monitoring-conditions test. It writes
`results_fast/SUMMARY.md` with the headline numbers and the confidence intervals.

> The 2-hour figure is a **target built from per-step estimates, not a measured result.** Cell 7 of the notebook times the
> verifier on your GPU and prints a projection for the dominant stages before they start. If it is too long, use the knobs below.

## How to run

(Kaggle instead of Colab: see [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md); the notebook detects the platform itself.)

1. Push the repo. In Colab: *File → Open notebook → GitHub → `notebooks/run_all_fast.ipynb`*; *Runtime → T4 GPU*.
2. *Runtime → Run all*. Every step is resumable: after a disconnect run again from the top, and finished steps are skipped
   (the results are copied to `MyDrive/detect-verify-decide/results_fast/` after every step).
3. Read `results_fast/SUMMARY.md` (also printed by the second-to-last cell) and the tables/figures in `results_fast/`.

The first cell fetches the latest `src/`, `configs/` and `tools/` from GitHub into the Drive copy of the project (your own notebooks and results are not touched).
If the project is on Drive without a `.git` folder, upload the new files by hand instead.

## What makes it fast (and what it costs)

The Verifier Agent runs on **every cached candidate** (that is what lets all thresholds be swept without re-running a model), so its cost is
images × candidates per image × seconds per crop. The previous protocol (500 + 1000 images, up to 100 candidates per image with c ≥ 0.20, 448 px crops)
needed ≈10 h for the verifier alone. The fast profile (`configs/default.yaml`) changes only sample size and input resolution:

| Setting | Full profile (`configs/full.yaml`) | Fast profile (default) | Effect |
|---|---|---|---|
| held-out / test images | 500 / 1000 | **150 / 300** | ~3× fewer images; wider confidence intervals |
| candidate threshold, cap per image | c ≥ 0.20, 100 | **c ≥ 0.25, 20** | the verifier sees the 20 most confident candidates; every method uses the same ones |
| crop zoom (longer side) | 448 px | **336 px** | 144 instead of 256 visual tokens per crop (~1.7× faster) |
| full-image views (global ablation, baselines, POPE) | 896 px | **448 px** | ~4× fewer tokens |
| verifier batch | 8 | **16** | better GPU use |
| margin search | 5 margins × 200 images | **3 margins × 60 images** | |
| Woodpecker-style | 1 image per call | **4 images per call** | per-image time is amortised over the batch (the DVD verifier timing is amortised over a batch too) |
| global-view verifier | one call per candidate | **one call per distinct (image, label)** | identical inputs give identical answers, so nothing changes except time |
| POPE | 3 × 3000 questions | **3 × 300** (first 300 of each file, balanced yes/no; only ~50 images are downloaded) | |
| COCO download | 778 MB zip | **annotation file + 450 images** | |
| robustness / online latency | 200 / 100 images | **40 / 30** | |
| LLM aggregator | 100 held-out images | **60** | |
| results folder | `results/` | **`results_fast/`** (profiles never mix) | |

Nothing is trained and no model is changed in either profile. What the smaller setup cannot do is establish **small** differences:
the summary marks a claim as *supported* only if its 95 % interval excludes zero and otherwise says *not established at this sample size*.

## What the notebook produces, and which claim it supports

| Claim in the proposal | Evidence (in `results_fast/`) |
|---|---|
| The pipeline reduces false positives at the same recall (Obj. 2) | `tables/main_comparison.*`: matched-recall rows, FP reduction with paired-bootstrap CI and recall change with CI; `figures/pr_frontier_*` |
| The verifier in isolation (Obj. 2) | `pope/pope_results.csv` (accuracy, precision, recall, F1, yes-rate; plain and VCD) |
| Role separation + crop evidence beat single-agent and decoding-time baselines (Obj. 3) | `tables/proposed_vs_each_method.csv` (paired ΔF1, ΔFP, Δrecall against Woodpecker-style, VCD, plain global, global-view ablation, crop-alone ablation) |
| Aggregation: rule vs LLM aggregator | `pipeline/llm_vs_rule_heldout_subset.csv` |
| Accuracy-versus-cost trade-off, bounded latency (Obj. 4) | `cost/*.csv`, `figures/cost_pipeline_tradeoff.png`, online measurement `cost/online_vs_estimated.csv` |
| Where verification helps or hurts (contribution 4) | `error_analysis/by_*.csv`, `figures/gallery_*.png`, `pipeline/decision_breakdown_test.csv` |
| Monitoring conditions (§6.1 extension) | `robustness/degradations.csv` |
| Margin choice, verifier separability | `verifier/margin_tuning.csv`, `figures/margin_search.png`, `figures/pyes_hist_test.png`, `verifier/verifier_diagnostics_test.csv` |

## If it is still too slow, or you want more

Edit `configs/default.yaml`, then **delete `results_fast/` on Drive** (the split and the caches depend on these values) and run again.

| To … | Change |
|---|---|
| halve the time | `coco.test_size: 150`, `coco.heldout_size: 100`, `detector.max_detections: 12`, `margin_tuning.margins: [0.15]` |
| get tighter intervals | `coco.test_size: 600` (time grows roughly linearly) |
| run the original protocol | set the environment variable `DVD_CONFIG=full` (see [COLAB_GUIDE.md](COLAB_GUIDE.md) §4) and use notebooks 01–11 |
| CUDA out of memory | `verifier.batch_size: 8` and restart the runtime |

## Limits worth knowing

- The proposal's headline claim (substantial FP reduction at roughly the same recall) is a **hypothesis the experiment tests**; the notebook reports whatever comes out, including "not established".
- The Woodpecker-style baseline is a re-implementation with the same VLM and detector (not the official GPT-3.5 + BLIP-2 system) and VCD is applied to per-(image, label) yes/no queries; see [DATASETS_AND_BASELINES.md](DATASETS_AND_BASELINES.md).
- Timings of batched stages are amortised per item, so absolute latencies are indicative, and the comparison between methods uses the same convention.
- The numbers of the fast profile are not directly comparable with the full profile (different candidate cap and resolution). Do not mix them in one table.
