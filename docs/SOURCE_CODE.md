# Source Code Reference (`src/dvd/`)

Heavy dependencies (torch, transformers, pycocotools, ultralytics) are imported **inside** the classes that need them. As a result, `metrics`, `decision`, `cropping`, `analysis` and `experiments` can be imported and unit-tested on a laptop without a GPU (`python tests/test_core.py`).

| File | Responsibility | Key API |
|---|---|---|
| `config.py` | Detect Colab / Kaggle (`in_colab`, `in_kaggle`: dataset and zips go to scratch space on Kaggle), load `configs/default.yaml` (fast profile) or the file named by `DVD_CONFIG` (e.g. `full`), resolve paths (`/content/data` on Colab), apply tuned values | `load_config()`, `save_selected(cfg, **kv)`, `results_dir(cfg, *parts)` |
| `paths.py` | Canonical cache file names (encode backend, VLM, view, mode, margin) | `det_cache`, `ver_cache`, `woodpecker_cache`, `vcd_cache`, `pope_cache` |
| `io_utils.py` | Resumable per-image JSONL store, JSON helpers, seeding | `JsonlCache(path).todo(ids)/add(rec)/records(ids)`, `set_seed` |
| `cropping.py` | Crop-and-zoom: expand box by margin, clip, upscale to 448 px | `expand_box`, `crop_and_zoom`, `fit_max_side` |
| `agents/detector.py` | **Detector Agent**: Grounding DINO (chunked prompts → phrase mapping → class-wise NMS) or YOLO-World | `build_detector(cfg, names, name_to_id)`, `.detect(image)`, `map_phrase` |
| `agents/verifier.py` | **Verifier Agent**: frozen Qwen2-VL; P(yes) from one forward pass; open-answer mode; text-only generation | `build_verifier(cfg)`, `.verify(views, labels, mode)`, `.yes_no`, `.generate`, `make_views` |
| `agents/decision.py` | **Decision Agent**: threshold rule and gating; optional LLM aggregator | `rule_accept`, `needs_verification`, `RuleDecisionAgent`, `LLMAggregatorAgent` |
| `pipeline.py` | Cached offline stages (the global-view stage asks each distinct (image, label) once: `verify_global`) and the online gated pipeline. `DetectVerifyDecide(image, on_event=cb)` streams progress events (`detector`, `gate`, `crop`, `verifier`, `decision`, `done`) and excludes callback time from latencies | `run_detection_stage`, `run_verification_stage`, `DetectVerifyDecide` |
| `baselines/woodpecker_style.py` | Baseline (ii): single-agent, full image plus candidate list, one call per image (several images per generation call when `woodpecker.batch_size > 1`; time is amortised per image) | `build_prompt`, `parse_answer`, `run_woodpecker_stage` |
| `baselines/vcd.py` | Baseline (iii): diffusion noise, contrastive yes/no with plausibility cut-off | `diffusion_noise`, `vcd_yes_no`, `run_vcd_stage`, `candidate_verdicts` |
| `metrics.py` | IoU/IoA, NMS, COCO-style greedy matching with crowd handling, P/R/F1, bootstrap CIs, decision breakdown, sweeps, AUROC, POPE metrics | `DetectionEvaluator`, `decision_breakdown`, `sweep_*`, `select_*`, `bootstrap_ci`, `paired_bootstrap`, `pope_metrics` |
| `experiments.py` | Loads all caches of a split as comparable *methods*, sweeps, selects, evaluates and costs them; builds the main table with paired bootstrap | `SplitData`, `available_methods`, `sweep`, `operating_point`, `evaluate_row`, `method_cost`, `main_comparison`, `paired_vs_proposed` |
| `cost.py` | CUDA-synchronised timer, per-call time, cost model | `Timer`, `per_call_seconds`, `detector_cost`, `estimate_config_cost` |
| `analysis.py` | Per-candidate table for error analysis; co-occurrence; brightness | `candidate_table`, `cooccurrence_topk`, `mean_brightness` |
| `degrade.py` | Low-light, motion blur, low-res and JPEG degradations | `degrade(img, condition)` |
| `viz.py` | Detection overlays, PR frontiers, crop galleries | `show_detections`, `plot_pr`, `crop_gallery` |
| `coco_classes.py` | The 80 COCO categories (id, name) without needing the annotation file | `COCO_CATEGORIES`, `COCO_NAMES`, `COCO_NAME_TO_ID` |
| `mock_agents.py` | Instant fake detector / verifier for UI previews and pipeline tests. **Random output, never a result** | `MockDetector`, `MockVerifier`, `synthetic_image` |
| `../../app/streamlit_app.py` | Streamlit demo UI built on the pipeline events (see [UI_GUIDE.md](UI_GUIDE.md)) | `streamlit run app/streamlit_app.py` |
| `data/coco.py` | COCO annotations, download of only the needed images, `CocoData` wrapper (xyxy GT, names↔ids), frozen splits | `prepare_annotations`, `fetch_images`, `prepare_coco`, `CocoData`, `make_splits`, `load_splits` |
| `data/pope.py` | Lazy POPE iterator (`pope.source`: GitHub question files + only the used images, or the HF dataset; the other one is the fallback) | `iter_pope(cfg, setting, max_n)` |
| `data/download.py` | Download with progress; persistent zip cache, local extraction | `download`, `download_and_extract` |

## Record formats

**Detection cache line** (`results/detections/*.jsonl`)
```json
{"image_id": 139, "boxes": [[x1,y1,x2,y2]], "scores": [0.71], "labels": ["person"], "category_ids": [1],
 "det_time_s": 0.84, "det_invocations": 4, "width": 640, "height": 426}
```

**Verifier cache line** (`results/verifier/*.jsonl`), aligned index-by-index with the detection line
```json
{"image_id": 139, "view": "crop", "margin": 0.15, "mode": "existence",
 "p_yes": [0.93], "verdict": [true], "answers": null, "verify_time_s": 0.11, "verify_invocations": 1}
```

**Woodpecker-style line**: `verdict` (per candidate), `raw` (model text), `parse_ok`, `listed`, `time_s`, `invocations`.
**VCD line**: `labels` (distinct candidate labels), `p_vcd`, `p_plain` (per label), `time_s`, `invocations`.
**POPE line**: `idx`, `question`, `label`, `p_yes`, `pred`, `time_s`.

## Extending

- **Another detector:** add a class with `.detect(image)` returning the record above, then register it in `build_detector`.
- **Another VLM:** set `verifier.model`. Any HF image-text-to-text model whose chat template accepts `{"type": "image"}` should work, and the cache names change automatically.
- **Another decision rule:** add a function `scores, verdicts, p_yes -> accept`, plus a `kind` in `experiments.masks_for` and `sweep`.

## Tests
`tests/test_core.py` covers the global-view deduplication, batched Woodpecker equivalence, the paired bootstrap, the main-comparison table on synthetic data, the config profiles, and also IoU, greedy matching (duplicates, crowd regions, re-matching after rejection), the decision rule and gating, the breakdown, sweeps, NMS, cropping at image borders, phrase-to-class mapping, open-answer matching, Woodpecker prompt and parser, AUROC and POPE metrics. Run it with `python tests/test_core.py` or `pytest tests -q`.
