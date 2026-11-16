# Streamlit Demo UI

`app/streamlit_app.py` is an interactive front-end for the pipeline. It is a demo and inspection tool: it is **not** used to produce any reported result (those come from notebooks 02–10).

## What it shows

While a run is in progress:

| Element | Meaning |
|---|---|
| **Progress bar with a percentage** | Overall progress of the single fixed pass, with the current message |
| **Five agent cards** | ⏳ waiting · 🔄 running · ✅ done · ⏭️ skipped, each with its latest note and its duration |
| **Live feedback log** | Timestamped lines, including the verifier's answer for every crop (`P(yes)` and keep / reject) |

After the run:

| Tab | Content |
|---|---|
| **Result** | Input next to the annotated output. Green = accepted, red = rejected by the verifier, grey = rejected for low confidence |
| **Verifier crops** | The exact crop-and-zoom views the verifier saw, with `c`, `P(yes)` and the outcome |
| **All candidates** | One row per candidate: confidence, route, `P(yes)`, decision and the reason |
| **Run log** | The full log of the run |

A summary line states how many objects the detector alone would have reported and how many the pipeline reports. Metrics show the candidate counts and the detector, verifier, decision and total time.

## Progress events

The pipeline streams events through `DetectVerifyDecide.__call__(image, on_event=callback)`. The UI is only a consumer of those events.

| Stage | Progress | Meaning |
|---|---|---|
| `detector` | 2 % → 52 % | One update per Grounding DINO text prompt (4 prompts for the 80 COCO classes) |
| `gate` | 57 % | Counts of auto-accepted, auto-rejected and to-be-verified candidates |
| `crop` | 58 % → 68 % | One update per uncertain candidate being cropped and zoomed |
| `verifier` | 68 % → 95 % | One update per batch of crops, with the answers |
| `decision` | 96 % | The decision rule is applied |
| `done` | 100 % | Final count of accepted candidates |

If no candidate falls in the uncertain band, the **crop** and **verifier** cards show ⏭️ skipped.

Time spent inside the callback (UI refreshes) is subtracted, so the reported latencies are model time only.

## Sidebar controls

| Control | Effect |
|---|---|
| Upload / COCO sample | Input image (long side capped at 1600 px) |
| Uncertain band (τ_low, τ_high) | `c ≥ τ_high` is accepted, `c < τ_low` is rejected, the rest is verified. Defaults come from `results/tuning/selected.json` when it exists, otherwise 0.25 and 0.60 |
| Crop margin | Context around each box; default from the tuned config value |
| Run pipeline | Starts a run |

## How to launch

**On Colab (real models).** Run `notebooks/12_streamlit_demo.ipynb` on a T4 runtime. It starts Streamlit and a Cloudflare quick tunnel, then prints a temporary URL.

- The tunnel URL is **public**: anyone who has it can use the app on your session until you run the last cell of the notebook.
- The first start downloads the model weights, so it takes a few minutes.

**On a laptop without a GPU (preview only).**
```bash
pip install streamlit pandas pillow pyyaml numpy tqdm
streamlit run app/streamlit_app.py
```
Without torch (or with `DVD_MOCK=1`) the app runs in **MOCK mode**: fake detector and verifier with random outputs (`src/dvd/mock_agents.py`), and a banner says so. It exists only to preview the layout and the live progress. Never use mock output as a result.

## Known limits

- **Tested only in mock mode** (Streamlit's headless `AppTest`, plus the pipeline-event unit test). It has not been run against the real models or on Colab yet.
- Model loading shows a two-step progress bar (detector, then verifier). The weight-download percentage is not shown; Hugging Face prints it in the Colab log.
- Progress is step-based, so the percentage is not proportional to wall-clock time.
- One image at a time, one user at a time.
- Needs `streamlit>=1.52` (the app uses `width="stretch"`, which replaces the deprecated `use_container_width`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Banner says MOCK MODE on Colab | torch was not importable. Re-run the install cell, then restart the app cell |
| No URL printed by notebook 12 | Read `/content/cloudflared.log`; re-run the tunnel cell |
| Blank page or errors | Read `/content/streamlit.log` |
| CUDA out of memory | Lower `verifier.batch_size` in `configs/default.yaml` and restart the app cell |
| `The pipeline failed` with a traceback | Send the traceback; the same code path runs in notebook 00, which is the easier place to debug |
