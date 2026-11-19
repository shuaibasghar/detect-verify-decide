# Changelog

## Unreleased - Woodpecker baseline fix, extended test, honest summary wording

Not re-run end to end on a GPU yet; verified with unit tests and a fake-model dry run of `run_all_fast.ipynb`.

1. **Woodpecker-style baseline fixed (v2)** - `src/dvd/baselines/woodpecker_style.py`, `src/dvd/paths.py`, notebook section 5.
   The 2B VLM echoed the numbered list and answered yes to everything (parse rate 0.52). Now: strict JSON answer with a short
   example that includes a `reject`, greedy decoding, `max_new_tokens` sized to the list, JSON-then-regex parser, and unparsed items
   are re-asked one by one as a plain yes/no on the full image (counted as extra invocations / time) instead of defaulting to yes.
   Records add `n_fallback`, `parse_ok_final`; `parse_ok` is the first-pass rate; new `parse_summary()`. New cache name
   `woodpecker_v2_*.jsonl` (the old file is untouched). Section 5 prints first-pass / final parse rate, fallback count, keep rate and
   3 raw answers, and warns loudly if the first-pass rate is < 0.9.
2. **Extended test for the main claim** - `configs/default.yaml` (`coco.test_ext_size: 700`, `full.yaml`: 0), `src/dvd/data/coco.py`
   (`make_splits` adds a disjoint `test_ext` to an existing `splits.json` without touching `heldout`/`test`),
   `src/dvd/experiments.py` (`extended_comparison`: detector-only vs proposed at the already-selected operating points, paired
   bootstrap, test+test_ext / test_ext only / test only, writes `tables/extended_comparison.csv|md`), new notebook section 8b, summary block.
   `paths.det_cache` / `ver_cache` already accept any split name, so no change was needed there.
3. **Summary wording** - notebook section 12: explains that best-F1 gains come from recall (tau_low below the detector tau*), FP reduction is
   claimed only at matched recall, Woodpecker parse statistics in the caveats, and the crop-vs-global statement is computed from the paired CI.
4. **Sanity checks and cleanups** - `src/dvd/data/pope.py`, `src/dvd/paths.py` (`pope_cache(..., sample)`), notebook sections 7, 10, 11.
   POPE: `overlap_report` prints the overlap of the question sets before running; a setting whose questions repeat an earlier one
   (>= 99 %) switches to a seeded balanced sample with its own cache file; prediction agreement between settings is printed.
   On the real POPE files the first 300 questions of random and popular overlap by ~53 % (the positives are shared, the negatives differ),
   so no switch happens and the identical numbers are a coincidence (the same counts: 113 TP and 5 FP in both). Error analysis: pandas FutureWarning removed
   (`groupby(..., observed=True)[cols].apply`). Robustness: all columns displayed, comment on `fp_reduction`.
5. **Tests** - `tests/test_core.py`: Woodpecker parser (JSON, regex fallback, missing indices, degenerate echo), fallback behaviour,
   `make_splits` keeps old ids and adds a disjoint `test_ext`, POPE sampling, `extended_comparison`. `docs/FAST_RUN.md`, `docs/DATASETS_AND_BASELINES.md` updated.
