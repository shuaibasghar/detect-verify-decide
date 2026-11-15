# results/

Git-ignored except this file. Everything here is produced by the notebooks and lives on Drive, so it persists across Colab sessions.

| Path | Contents | Created by |
|---|---|---|
| `splits.json` | Frozen held-out / test image ids (seed 42) | NB01 |
| `data/` | Category counts, POPE stats, co-occurrence top-10 | NB01 |
| `detections/` | Detector candidates per image (JSONL, resumable) | NB02, NB11 |
| `detector/` | Detector-only sweeps, mAP sanity check | NB02 |
| `verifier/` | Verifier P(yes) per candidate (crop / full, per margin), margin tuning, diagnostics | NB03, NB11 |
| `tuning/selected.json` | Tuned crop margin and thresholds (read by every notebook) | NB02, NB03, NB05 |
| `pope/` | POPE answers and metrics | NB04 |
| `pipeline/` | Operating points, decision breakdown, sweeps | NB05 |
| `baselines/` | Woodpecker-style and VCD caches and results | NB06, NB07 |
| `tables/` | `main_comparison.{csv,md}` | NB08 |
| `cost/` | Configuration and method costs, online vs estimated | NB09 |
| `error_analysis/` | Breakdowns by category, size, confidence, brightness, co-occurrence | NB10 |
| `robustness/` | Degradation results | NB11 |
| `figures/` | All PNG figures | NB02–NB11 |

To redo a stage, delete its cache file and re-run the notebook. To start over completely, delete `results/` except this README.
