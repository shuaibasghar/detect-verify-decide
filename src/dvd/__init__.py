"""Detect–Verify–Decide: a training-free multi-agent pipeline for reducing false positives.

Heavy dependencies (torch, transformers, pycocotools) are imported lazily inside the
modules that need them, so `dvd.metrics`, `dvd.decision` and `dvd.cropping` can be used
(and unit-tested) on a machine without a GPU.
"""

# Package version string (shown by tools and logs).
__version__ = "0.1.0"
