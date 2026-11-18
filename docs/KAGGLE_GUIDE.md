# Running on Kaggle

`notebooks/run_all_fast.ipynb` (the whole proposal in one session, see [FAST_RUN.md](FAST_RUN.md)) runs on Kaggle as well as on Colab.
The numbered notebooks 00-11 also have the Kaggle bootstrap; notebook 12 (Streamlit + Cloudflare tunnel) is Colab-only.

> Not yet run on a real Kaggle machine. The Kaggle code path was exercised in a simulation (fake `/kaggle/...` folders, fake models); the first real run is the real test.

## One-time setup

1. **Phone-verify** the Kaggle account (required for *Internet*), otherwise nothing can be downloaded.
2. **Get the code into the notebook** - pick one:
   - *Public repo or branch:* nothing to do; set `REPO_URL` (and `BRANCH`) in the first cell.
   - *Private repo:* create a GitHub personal access token with read access to the repo, then in the notebook *Add-ons → Secrets* add a secret named **`GITHUB_TOKEN`** and attach it to the notebook. The token is used only for the clone and is removed from `.git/config` afterwards.
   - *No GitHub at all:* upload the project folder (with `src/`, `configs/`, `requirements.txt`) as a Kaggle **Dataset** and *Add Input* it; the first cell finds `src/dvd` in `/kaggle/input` and copies it.
3. **COCO without the internet (recommended on Kaggle):** the COCO server often answers `HTTP Error 503: Slow Down` to Kaggle's shared IPs. Use *Add Input* and search for a **COCO 2017** dataset (for example "COCO 2017 Dataset"); the notebook finds `instances_val2017.json` and the `val2017` folder under `/kaggle/input` by itself, copies only the 450 images it needs and prints `[coco] ... taken from the attached input`. Without such an input it downloads from cocodataset.org (with retries and pauses).
4. **Import the notebook:** *Create → New Notebook → File → Import Notebook*, and upload `notebooks/run_all_fast.ipynb`.
5. **Session options (right side):** *Accelerator* = **GPU T4 x2** (a P100 is often unsupported by the current PyTorch build; the notebook warns about it), *Internet* = **On**. The code uses one GPU; the second one stays idle.

## Run

- **Interactive:** *Run All*. Fine for trying it out.
- **Recommended for the full run:** *Save Version → Save & Run All (Commit)*. It runs in the background (up to 12 h), you can close the browser, and everything under `/kaggle/working` is **kept as the notebook output**.

Plain `Run All` results disappear when the session ends unless you press *Save Version* (or enable *Persistence: Files only* in the session options).

## Where things go

| What | Where | Saved as output? |
|---|---|---|
| project code | `/kaggle/working/detect-verify-decide/` | yes (small) |
| results, tables, figures, `SUMMARY.md` | `/kaggle/working/detect-verify-decide/results_fast/` | **yes** (this is what you download) |
| COCO annotations + images, POPE images | `/kaggle/temp/` (or `/tmp`) | no (scratch, re-downloaded each session, ~1-3 min) |

## Resuming after a stop or a failure

1. In the next notebook version, *Add Input → Your Work / Notebook Output* and choose the previous version.
2. Run again. The setup cell copies every `results_fast/` it finds under `/kaggle/input` into the working folder (it prints `restored earlier results from ...`), and every finished step is skipped.

## Limits and what to expect

- **Quota:** about 30 GPU hours per week, up to 12 h per session. The fast profile is meant to fit comfortably; see [FAST_RUN.md](FAST_RUN.md) for the time per step and the knobs to shrink it.
- **Speed** is that of the T4, i.e. the same as Colab. A failed commit still shows its partial output; fix the error and run again with the previous version attached.
- **Do not** put a token directly into `REPO_URL`: the notebook (and its output) can be public. Use the secret.
- `HTTP Error 503: Slow Down` from `images.cocodataset.org`: the downloader now retries with growing pauses (up to ~2 minutes) and tries https as well; if it still fails, attach a COCO 2017 dataset as an input (setup step 3) or run again later. POPE images come from the same server and use the same retries.
- Code of an earlier run stays in `/kaggle/working` when *Persistence* is on; the first cell updates it from `REPO_URL`/`BRANCH` on every start (a private repo needs the `GITHUB_TOKEN` secret for that too).
- If `git clone failed` appears: check *Internet = On*, that `REPO_URL` and `BRANCH` exist, and (private repo) that the `GITHUB_TOKEN` secret is attached.
- `pip install` may print dependency warnings about packages Kaggle pre-installs (for example `gradio`); they are not used by this project.
