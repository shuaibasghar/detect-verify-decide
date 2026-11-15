# Datasets, Models and Baselines

## Datasets

| Dataset | Use | Source | Size | Licence / terms |
|---|---|---|---|---|
| COCO val2017 | Main detection evaluation (80 classes) | `images.cocodataset.org` | 5000 images, ≈1 GB with annotations | CC BY 4.0 annotations; images under Flickr terms |
| POPE | Verifier in isolation; random, popular and adversarial settings | HF `lmms-lab/POPE`; fallback `github.com/RUCAIBox/POPE` + COCO val2014 images | 3 × 3000 questions over 500 images | Follows COCO terms |
| Monitoring subset (optional) | Realistic conditions | Synthetic degradations of COCO test images; your own frames in `data/custom_images/` | 200 images | — |

The proposal suggests POPE-style questions built from automatic segmentation for unannotated CCTV data. That is listed as future work. NB11 uses degraded COCO images because their ground truth remains valid.

## Models (all frozen, no training)

| Role | Default checkpoint | Alternatives |
|---|---|---|
| Detector Agent | `IDEA-Research/grounding-dino-tiny` (HF transformers) | `grounding-dino-base`; YOLO-World `yolov8s-worldv2.pt` (Ultralytics) |
| Verifier Agent | `Qwen/Qwen2-VL-2B-Instruct` | Any ≤2–3B HF image-text-to-text chat VLM |
| Decision Agent | Threshold rule (no model) | LLM aggregator, using the verifier's language model in text-only mode |

## Baselines

### (i) Detector alone
The cached candidates are thresholded at τ, and τ\* is chosen on held-out by best F1.

### (ii) Woodpecker-style single-agent self-verification — `src/dvd/baselines/woodpecker_style.py`
**What it tests:** whether separating verification from decision (and supplying crop evidence) matters, compared with one model that does both on global evidence.

**Implementation:**
- The *same* VLM as the verifier receives the full image (longer side ≤ 896 px) and a numbered list of the top-30 candidates with boxes on a 0–1000 scale.
- It answers `k: yes|no` for every item in one generation call, and that answer *is* the decision. Detector confidence is not fused in.
- Candidates outside the top 30 are rejected. Listed candidates whose answer cannot be parsed keep the detector's claim, and the parse rate is logged.

**Deviation from the proposal, stated plainly:** the proposal says this baseline is "implemented from the authors' public code" (github.com/BradyFU/Woodpecker). That code chains GPT-3.5 (paid API), Grounding DINO and BLIP-2 to correct *captions*. It cannot score detector boxes directly, and it would use different models from our pipeline. To keep the comparison like-for-like (identical detector, identical VLM, identical candidates), we re-implement Woodpecker's *structure*: one agent that validates and decides on global evidence. The report must state this. Running the official code on captions remains an optional extra if API budget allows.

### (iii) Visual Contrastive Decoding — `src/dvd/baselines/vcd.py`
- For a single-token yes/no answer, VCD is one contrastive step:
  `cd = (1+α)·logits(v) − α·logits(v′)`, where tokens with `logits(v) < log β + max logits(v)` are masked.
- `v′` is the image with diffusion noise from the official schedule: 1000 steps, sigmoid β-schedule, noise step 500, applied in CLIP-normalised space. Qwen2-VL uses the same normalisation.
- Settings: α = 1, β = 0.1, noise step 500. The proposal also lists **γ = 0.1**. Before reporting, check which VCD hyper-parameter γ refers to (the official code exposes `cd_alpha`, `cd_beta` and `noise_step`) and record the mapping.
- Applied to detections: for each image and distinct candidate label, "Is there a {label} in the image?" on the full image. A candidate is kept iff the VCD answer for its label is yes. The plain answer from the same passes is reported as "plain global yes/no".
- The proposal phrases this baseline as "VCD applied to captions generated from the same detections". The per-label existence question is the binary form of that and scores detections directly. State this choice in the report.

### Ablations of the proposed pipeline

| Ablation | What changes | Isolates |
|---|---|---|
| Crop verifier decides alone | τ_high = ∞: the verdict is final for every c ≥ τ | Value of fusing detector confidence in a separate Decision Agent |
| Global-view verifier + rule | Verifier sees the full image instead of the crop | Value of evidence separation (crop-and-zoom) |
| Open-answer mode (optional) | The VLM names the object without being given the label | Effect of naming the label in the query |
| LLM aggregator (optional) | LLM instead of the rule | Rule vs learned-free LLM fusion |

## Acknowledgements to include in the report
Grounding DINO (Liu et al., 2023), Qwen2-VL (Wang et al., 2024), COCO (Lin et al., 2014), POPE [1], Woodpecker [2], VCD [3], Hugging Face `transformers` and `datasets`, `pycocotools`, and Ultralytics YOLO-World if used.
