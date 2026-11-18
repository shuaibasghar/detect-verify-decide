"""Mock agents: instant, deterministic stand-ins for the real models.

Used to preview the Streamlit UI on a machine without a GPU and to unit-test the pipeline
orchestration. Their outputs are random and mean NOTHING - never use them for results.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# time: `sleep` simulates model latency so the UI progress bar can be seen moving.
import time
# zlib: crc32 turns text into a stable number, giving "random" but reproducible outputs.
import zlib

# numpy: random number generation and array maths for the fake boxes and scores.
import numpy as np
# Pillow: Image for the type, ImageDraw to paint rectangles on the synthetic test image.
from PIL import Image, ImageDraw

# The 80 COCO (id, name) pairs, so fake detections carry real class names.
from .coco_classes import COCO_CATEGORIES


class MockDetector:
    """Fake Detector Agent with the same `.detect(image, on_progress)` interface as the real one."""

    def __init__(self, delay: float = 0.0, n_prompts: int = 4):
        # `delay`: seconds to sleep per simulated prompt; `n_prompts`: number of progress steps to report.
        self.delay, self.n_prompts = delay, n_prompts

    def detect(self, image, on_progress=None) -> dict:
        """Return a random-but-deterministic candidate list in the real detector's record format."""
        w, h = image.size
        # Seed from the image size so the same image always yields the same fake detections.
        rng = np.random.default_rng(zlib.crc32(f"{w}x{h}".encode()))
        # Between 6 and 11 fake candidates.
        n = int(rng.integers(6, 12))
        # Random COCO classes for each candidate.
        cats = [COCO_CATEGORIES[int(i)] for i in rng.integers(0, len(COCO_CATEGORIES), n)]
        # Random box sizes (10-40 % of the image) ...
        bw, bh = rng.uniform(0.1, 0.4, n) * w, rng.uniform(0.1, 0.4, n) * h
        # ... and top-left corners placed so the box stays inside the image.
        x1, y1 = rng.uniform(0, 1, n) * (w - bw), rng.uniform(0, 1, n) * (h - bh)
        # Stack into an (n, 4) array of xyxy boxes.
        boxes = np.stack([x1, y1, x1 + bw, y1 + bh], 1)
        # Random confidences in [0.2, 0.95], sorted high to low like the real detector's output.
        scores = np.sort(rng.uniform(0.2, 0.95, n))[::-1]
        # Simulate the detector's per-prompt progress callbacks.
        for k in range(1, self.n_prompts + 1):
            time.sleep(self.delay)
            if on_progress:
                on_progress(k, self.n_prompts, int(round(n * k / self.n_prompts)))
        # Same record layout as GroundingDINODetector.detect (see agents/detector.py).
        return {"boxes": boxes.round(2).tolist(), "scores": scores.round(4).tolist(),
                "labels": [c[1] for c in cats], "category_ids": [c[0] for c in cats],
                "det_time_s": self.delay * self.n_prompts, "det_invocations": self.n_prompts,
                "width": w, "height": h}


class MockVerifier:
    """Fake Verifier Agent with the same `.verify(views, labels, ...)` interface as the real one."""

    # The pipeline reads `verifier.batch_size` to chunk crops; a small value makes UI progress visible.
    batch_size = 4

    def __init__(self, delay: float = 0.0):
        # Seconds of simulated latency per `verify` call.
        self.delay = delay

    def verify(self, views, labels, mode="existence", *args, **kwargs) -> dict:
        """Return a fake P(yes) per crop (derived from the crop size and label, so it is reproducible)."""
        time.sleep(self.delay)
        # Hash (crop size, label) to a number in [0, 1): looks random, but is stable between runs.
        p = [(zlib.crc32(f"{v.size}{l}".encode()) % 1000) / 1000 for v, l in zip(views, labels)]
        return {"p_yes": p, "verdict": [x > 0.5 for x in p], "answers": None}


def synthetic_image(w: int = 640, h: int = 480) -> Image.Image:
    """A plain gradient with a few rectangles, for trying the UI offline."""
    # Horizontal ramp 0..255 (red channel), repeated down all rows.
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
    # Vertical ramp 255..0 (green channel), repeated across all columns.
    y = np.linspace(255, 0, h, dtype=np.uint8)[:, None].repeat(w, 1)
    # Blue channel = average of the two ramps; stack the three channels into an RGB image.
    img = Image.fromarray(np.stack([x, y, (x // 2 + y // 2).astype(np.uint8)], -1))
    d = ImageDraw.Draw(img)
    # Three solid rectangles (red, green, yellow) so the image has something to "detect".
    for box, col in [((60, 80, 220, 260), (230, 60, 60)), ((300, 150, 520, 380), (60, 200, 90)),
                     ((420, 40, 600, 130), (240, 220, 60))]:
        d.rectangle(box, fill=col)
    return img
