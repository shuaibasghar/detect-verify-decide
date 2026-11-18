# Running on Google Colab

> **Fastest path:** `notebooks/run_all_fast.ipynb` runs the whole proposal in one session (target ≈2 h on a T4). See [FAST_RUN.md](FAST_RUN.md).
> The rest of this page covers the step-by-step notebooks 00–12.

## 1. One-time setup

**Option A — Google Drive (simplest)**
1. Upload the whole project folder to Google Drive as `MyDrive/detect-verify-decide/`.
   It must contain `src/`, `configs/`, `notebooks/` and `requirements.txt`.
2. On drive.google.com, open `notebooks/00_setup_and_smoke_test.ipynb` → *Open with → Google Colaboratory*.

**Option B — GitHub**
1. Push the repo to GitHub.
2. In the first code cell of each notebook, set `REPO_URL = "https://github.com/<you>/detect-verify-decide.git"`.
   On first run it is cloned into `MyDrive/detect-verify-decide`.
3. Open the notebook from GitHub in Colab (*File → Open notebook → GitHub*).

If you use a different Drive folder, change `DRIVE_PROJECT` in the first code cell. It is identical in every notebook.

## 2. Every session
1. *Runtime → Change runtime type → T4 GPU* (or better).
2. Run all cells top to bottom. The bootstrap cell does the following:
   - mounts Drive;
   - adds `src/` to `sys.path` so `import dvd` works;
   - installs `requirements.txt` (≈1–2 min; Colab's own torch is kept);
   - loads `configs/default.yaml` plus any tuned values from `results/tuning/selected.json`.
3. COCO images are re-extracted to the fast local disk `/content/data` from the zip cached on Drive (≈1–2 min). The download itself happens only once.

## 3. Order and expected time (T4, defaults: held-out 500, test 1000)

These are planning estimates. Real timings depend on the GPU you are given.

| Notebook (full profile) | GPU | Approx. time | Can resume? |
|---|---|---|---|
| 00 smoke test | yes | 10 min | — |
| 01 data prep | no (CPU fine) | 10 min | yes (download cache) |
| 02 detector | yes | 30–45 min | yes |
| 03 verifier + margin + global ablation | yes | 1.5–3 h | yes |
| 04 POPE (plain + VCD) | yes | 45–90 min | yes |
| 05 decision + pipeline | light | 5–15 min | — (offline) |
| 06 Woodpecker-style | yes | 1–2 h | yes |
| 07 VCD | yes | 45–60 min | yes |
| 08 main comparison | no | 5 min | — (offline) |
| 09 cost | yes (online part) | 10 min | — |
| 10 error analysis | no | 5 min | — (offline) |
| 11 robustness (optional) | yes | 1–2 h | yes |
| 12 Streamlit demo (optional) | yes | ≈3 min to start | — |

Free-tier sessions last up to ~12 h but disconnect when idle. Long notebooks (03, 06, 11) may need 2–3 sessions. Just re-run them from the top and they continue where they stopped.

## 4. Fast vs full profile
`configs/default.yaml` is the **fast profile** (150 held-out / 300 test images, ≤20 candidates per image, 336 px crops, results in `results_fast/`).
The original protocol is `configs/full.yaml` (500 / 1000 images, results in `results/`). Select it by setting `DVD_CONFIG=full` in the first cell of a notebook, before the config cell:

```python
os.environ["DVD_CONFIG"] = "full"      # default = fast profile
```

> The split and all caches depend on these values. After changing `heldout_size`, `test_size`, `box_threshold`, `max_detections` or the resolutions, delete the matching results folder (`results_fast/` or `results/`) so nothing stale is reused.
> The durations in the table of section 3 are for the **full** profile; the fast profile needs roughly a quarter of them (see [FAST_RUN.md](FAST_RUN.md)).

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `src/dvd not found under ...` | `DRIVE_PROJECT` points to the wrong folder |
| CUDA out of memory | Lower `verifier.batch_size` (8 → 4 → 2) and restart the runtime. Do not load two VLMs in one session. |
| `... 0/500 images cached - run the notebook that produces it` | Run the producing notebook (see the dependency graph in `PROJECT_PLAN.md`) |
| Very slow verifier on T4 | T4 has no native bf16, so fp16 is chosen automatically. Make sure the runtime really is a GPU. |
| POPE download fails | The loader falls back to the GitHub question files and downloads the 500 COCO val2014 images individually |
| `transformers` import errors after install | *Runtime → Restart session*, then run again from the top (needed only if transformers was imported before installing) |
| Half-written cache line after a crash | Ignored automatically on load (`JsonlCache`) |

## 6. Using a paid / faster GPU
On an A100 or L4 you can raise `verifier.batch_size` to 16–32 and use the full test split. bf16 is selected automatically when the GPU supports it.
