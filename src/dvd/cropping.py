"""Crop-and-zoom: the step that gives the verifier evidence the detector's global pass never produced."""
from __future__ import annotations

from PIL import Image


def expand_box(box, margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Expand an xyxy box by `margin` x (box w/h) on every side, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_and_zoom(image: Image.Image, box, margin: float = 0.15, target_size: int = 448,
                  min_side: int = 28) -> Image.Image:
    """Crop the expanded box and upscale (or downscale) so the longer side == target_size."""
    w, h = image.size
    crop = image.crop(expand_box(box, margin, w, h))
    cw, ch = crop.size
    scale = target_size / max(cw, ch)
    nw, nh = max(min_side, int(round(cw * scale))), max(min_side, int(round(ch * scale)))
    return crop.resize((nw, nh), Image.BICUBIC)


def fit_max_side(image: Image.Image, max_side: int, min_side: int = 28) -> Image.Image:
    """Downscale a full image so its longer side <= max_side (bounds VLM visual tokens)."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    s = max_side / max(w, h)
    return image.resize((max(min_side, round(w * s)), max(min_side, round(h * s))), Image.BICUBIC)
