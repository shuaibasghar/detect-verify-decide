"""Latency measurement and the per-configuration cost model.

Model invocations are counted per model *input*: one text prompt for the detector, one crop
(or full image) for the verifier, regardless of how inputs are batched on the GPU.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# time: high-resolution clock for latency.
import time

# numpy: means over per-image timings.
import numpy as np


def _sync():
    """Wait for all queued GPU work to finish, so timings measure the real compute, not just the launch."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    """`with Timer() as t: ...` -> t.elapsed seconds, CUDA-synchronised on both ends."""

    def __enter__(self):
        # Flush pending GPU work first, then start the clock.
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        # Wait for the block's GPU work, then stop the clock.
        _sync()
        self.elapsed = time.perf_counter() - self._t0
        # False = do not swallow exceptions raised inside the `with` block.
        return False


def per_call_seconds(records: dict, time_key: str, calls_key: str) -> float:
    """Mean seconds per model invocation across a cache of per-image records."""
    t = sum(r[time_key] for r in records.values())
    n = sum(r[calls_key] for r in records.values())
    return t / n if n else 0.0


def detector_cost(det_records: dict) -> dict:
    """Mean detector latency (seconds per image) and mean number of detector forward passes per image."""
    t = np.array([r["det_time_s"] for r in det_records.values()])
    n = np.array([r["det_invocations"] for r in det_records.values()])
    return {"det_latency_s": float(t.mean()), "det_invocations": float(n.mean())}


def estimate_config_cost(det_records: dict, n_verified_per_image: float, t_per_verify: float) -> dict:
    """Cost of a pipeline configuration from cached timings.

    In the online pipeline only candidates in the uncertain band tau_low <= c < tau_high are sent
    to the verifier, so added latency = n_verified x per-crop time (+ negligible decision time).
    """
    d = detector_cost(det_records)
    # Extra time caused by the verifier stage.
    added = n_verified_per_image * t_per_verify
    return {
        **d,
        "verify_invocations": n_verified_per_image,
        "added_latency_s": added,
        "total_latency_s": d["det_latency_s"] + added,
        "total_invocations": d["det_invocations"] + n_verified_per_image,
        # How many times slower than the detector alone.
        "latency_ratio": (d["det_latency_s"] + added) / d["det_latency_s"] if d["det_latency_s"] else np.nan,
    }
