# Evaluation Protocol

The protocol below is fixed before any results are seen. Everything is implemented in `src/dvd/metrics.py` and `src/dvd/experiments.py`.

## 1. Splits
- COCO val2017 is split once, with seed 42, into **held-out** (tuning only) and **test** (all reported numbers). Sizes: 150 / 300 in the fast profile (default), 500 / 1000 in the full profile. The split is stored in `<results folder>/splits.json` (`results_fast/` or `results/`).
- **Held-out is used only for selection:** crop margin, τ\* for detector-only, (τ_low, τ_high) for the pipeline, and τ for every baseline.
- **Test is used only for reporting.** No choice is ever made on test.

## 2. Matching (detection level)
For each image, detections are processed in descending confidence:
1. Among non-crowd ground-truth boxes of the **same class** not yet matched, take the one with the highest IoU. If IoU ≥ 0.5, the detection is a **TP** and that ground-truth box is consumed.
2. Otherwise, if the detection overlaps a same-class **crowd** region with IoA ≥ 0.5, it is **ignored** (as in COCO).
3. Otherwise it is a **FP**.

Every method's *accepted subset* is re-matched from scratch. If the pipeline rejects a duplicate's twin, the remaining box can still match.

## 3. Metrics

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / #non-crowd GT boxes |
| F1 | 2PR / (P + R) |
| FP per image | FP / #images |
| **FP rate** (as used in this report) | FP / (TP + FP) = 1 − precision, i.e. the false-discovery rate. Detection has no true-negative count, so the classical FPR is undefined. We report this quantity together with FP/image. |
| **FP reduction** | 1 − FP_method / FP_detector-only(τ\*), on the same test images |
| 95 % CI | Image-level bootstrap, 1000 resamples (`bootstrap_ci`) |
| Paired comparison | `paired_bootstrap`: both methods are re-evaluated on the *same* resampled images, giving CIs for the FP reduction and the differences in precision, recall and F1, plus the share of resamples in which the first method is better. A claim is called *supported* only if its 95 % CI excludes 0. |
| COCO mAP | `pycocotools` on detector-only candidates; a sanity check only (lower bound because of the candidate threshold: 0.25 in the fast profile, 0.20 in the full one) |

## 4. Operating points
For every method, the configuration is chosen on held-out and applied **unchanged** to test:
- **best-F1:** the configuration with maximum F1. Ties are broken by fewer verifier calls.
- **matched-recall:** the configuration with the fewest FPs among those whose held-out recall is at least the detector-only best-F1 recall. This answers: *at the same recall, how many fewer false positives?* When no configuration reaches that recall, the best-F1 one is used and the row has `recall_matched = False`.
- **Full frontier:** every configuration is also plotted (PR frontier), so conclusions do not hinge on a single point.

Grids (in `configs/default.yaml`):
- Detector-only and gate methods: τ ∈ {0.20, 0.25, …, 0.80} (full profile); {0.25, …, 0.80} (fast profile, candidate threshold 0.25).
- Pipeline: τ_low ∈ {0.20 (full) / 0.25 (fast), …, 0.50} × τ_high ∈ {0.30, …, 0.90, 1.01}, with τ_high ≥ τ_low. τ_high = 1.01 means the verifier decides every candidate with c ≥ τ_low.

## 5. Decision-level breakdown (adapted from Woodpecker)
The reference pool is the candidates that the detector alone would output at τ_ref. NB05 uses τ_ref = τ_low; NB08 uses τ\* for every method, so all methods share one pool.

| Outcome | Candidate truth | Decision |
|---|---|---|
| **Correct rejection** | FP | rejected |
| **Omission** | FP | accepted |
| **Over-rejection** | TP | rejected |
| Correct acceptance | TP | accepted |

The two headline rates are:
- **FP-removed rate** = correct rejections / pool FPs.
- **Over-rejection rate** = over-rejections / pool TPs.

The over-rejection rate is the price paid in recall, and it is always reported next to any precision gain.

## 6. Verifier in isolation (POPE)
- Standard POPE questions for the random, popular and adversarial settings (3000 each, balanced; the fast profile uses the first 300 of each file). The answer is `P(yes) > 0.5` on the full image, with the suffix "Answer yes or no."
- Reported: accuracy, precision, recall, F1 and **yes-rate**. A yes-rate well above 0.5 indicates over-assertion.
- The same protocol is run with VCD decoding as the decoding-time reference.

The verifier's ability to separate TP from FP *candidates* is measured separately on COCO by the AUROC of P(yes) (NB03).

## 7. Cost
- **Model invocation:** one model input. For the detector this is one text prompt, so Grounding DINO uses 4 per image. For the verifier it is one crop or full image. For Woodpecker-style it is one generation per image. For VCD it is 2 forward passes per (image, label).
- **Latency:** wall-clock seconds per image, CUDA-synchronised (`cost.Timer`). It includes cropping and pre-processing, but not image loading from disk.
- **Cost model of the gated pipeline:**
  `latency(τ_low, τ_high) = t_det + N_band · t_crop`, where `N_band` = mean number of candidates with τ_low ≤ c < τ_high and `t_crop` = mean measured seconds per crop. Decision time is measured and negligible.
- **Validation:** NB09 runs the real online pipeline on 100 test images and compares measured with estimated latency.
- Reported per configuration: added latency, total latency, FPS, invocations per image, and FP reduction per added second.

## 8. Fairness rules
- All methods use **the same detector candidates**, the **same VLM checkpoint**, greedy decoding, and the same image-size caps.
- Every method gets its own threshold tuned on the same held-out split.
- No result is reported from the held-out split as a final number.
- Published numbers from the literature are cited as context and are never mixed into our tables.
