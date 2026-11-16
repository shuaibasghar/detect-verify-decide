"""Mock agents: instant, deterministic stand-ins for the real models.

Used to preview the Streamlit UI on a machine without a GPU and to unit-test the pipeline
orchestration. Their outputs are random and mean NOTHING - never use them for results.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
from PIL import Image, ImageDraw

from .coco_classes import COCO_CATEGORIES


class MockDetector:
    def __init__(self, delay: float = 0.0, n_prompts: int = 4):
        self.delay, self.n_prompts = delay, n_prompts

    def detect(self, image, on_progress=None) -> dict:
        w, h = image.size
        rng = np.random.default_rng(zlib.crc32(f"{w}x{h}".encode()))
        n = int(rng.integers(6, 12))
        cats = [COCO_CATEGORIES[int(i)] for i in rng.integers(0, len(COCO_CATEGORIES), n)]
        bw, bh = rng.uniform(0.1, 0.4, n) * w, rng.uniform(0.1, 0.4, n) * h
        x1, y1 = rng.uniform(0, 1, n) * (w - bw), rng.uniform(0, 1, n) * (h - bh)
        boxes = np.stack([x1, y1, x1 + bw, y1 + bh], 1)
        scores = np.sort(rng.uniform(0.2, 0.95, n))[::-1]
        for k in range(1, self.n_prompts + 1):
            time.sleep(self.delay)
            if on_progress:
                on_progress(k, self.n_prompts, int(round(n * k / self.n_prompts)))
        return {"boxes": boxes.round(2).tolist(), "scores": scores.round(4).tolist(),
                "labels": [c[1] for c in cats], "category_ids": [c[0] for c in cats],
                "det_time_s": self.delay * self.n_prompts, "det_invocations": self.n_prompts,
                "width": w, "height": h}


class MockVerifier:
    batch_size = 4

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def verify(self, views, labels, mode="existence", *args, **kwargs) -> dict:
        time.sleep(self.delay)
        p = [(zlib.crc32(f"{v.size}{l}".encode()) % 1000) / 1000 for v, l in zip(views, labels)]
        return {"p_yes": p, "verdict": [x > 0.5 for x in p], "answers": None}


def synthetic_image(w: int = 640, h: int = 480) -> Image.Image:
    """A plain gradient with a few rectangles, for trying the UI offline."""
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
    y = np.linspace(255, 0, h, dtype=np.uint8)[:, None].repeat(w, 1)
    img = Image.fromarray(np.stack([x, y, (x // 2 + y // 2).astype(np.uint8)], -1))
    d = ImageDraw.Draw(img)
    for box, col in [((60, 80, 220, 260), (230, 60, 60)), ((300, 150, 520, 380), (60, 200, 90)),
                     ((420, 40, 600, 130), (240, 220, 60))]:
        d.rectangle(box, fill=col)
    return img
