# Experiment Log

Fill this in from the files in `results_fast/` (fast profile, default) or `results/` (`DVD_CONFIG=full`); `results_fast/SUMMARY.md` already collects the headline numbers with their intervals. Paths below say `results/`; substitute the folder you used. **Copy numbers from the CSV/JSON outputs; never type them from memory.**
Every cell below is empty until the corresponding notebook has run. No results are claimed in advance.

## Run metadata

| Field | Value |
|---|---|
| Date(s) | |
| GPU | |
| `transformers` / `torch` versions | |
| Detector checkpoint | |
| Verifier checkpoint | |
| Held-out / test sizes | |
| Seed | 42 |

## Tuned parameters (`results/tuning/selected.json`)

| Parameter | Value | Selected in |
|---|---|---|
| crop_margin | | NB03 |
| detector-only τ\* | | NB02 / NB05 |
| τ_low, τ_high (best-F1) | | NB05 |
| τ_low, τ_high (matched-recall) | | NB05 |

## T1 — Main comparison on COCO test (`results/tables/main_comparison.md`)

| Method | P | R | F1 [95 % CI] | FP/img | FP reduction | Over-rejection | Latency s/img | Calls/img |
|---|---|---|---|---|---|---|---|---|
| Detector only | | | | | — | — | | |
| **DVD (proposed)** | | | | | | | | |
| Crop verifier decides alone | | | | | | | | |
| Global-view verifier + rule | | | | | | | | |
| Woodpecker-style single agent | | | | | | | | |
| VCD global yes/no | | | | | | | | |
| Plain global yes/no | | | | | | | | |

## T2 — Verifier on POPE (`results/pope/pope_results.csv`)

| Setting | Decoding | Acc | P | R | F1 | Yes-rate |
|---|---|---|---|---|---|---|
| random | plain | | | | | |
| random | VCD | | | | | |
| popular | plain | | | | | |
| popular | VCD | | | | | |
| adversarial | plain | | | | | |
| adversarial | VCD | | | | | |

## T3 — Decision breakdown (`results/pipeline/decision_breakdown_test.csv`)

| Config | Correct rejections | Omissions | Over-rejections | FP-removed rate | Over-rejection rate |
|---|---|---|---|---|---|
| best-F1 | | | | | |
| matched-recall | | | | | |

## T4 — Cost (`results/cost/*.csv`)

| | Detector s/img | Verifier s/img | Total s/img | Verifier calls/img | FPS |
|---|---|---|---|---|---|
| Measured (online, gated) | | | | | |
| Estimated (cost model) | | | | | |

## T5 — Robustness (optional, `results/robustness/degradations.csv`)

| Condition | Det P | Det R | DVD P | DVD R | FP reduction | Over-rejection |
|---|---|---|---|---|---|---|
| clean | | | | | | |
| low_light | | | | | | |
| motion_blur | | | | | | |
| low_res | | | | | | |
| jpeg | | | | | | |

## Observations / decisions log

| Date | Observation | Decision |
|---|---|---|
| | | |
