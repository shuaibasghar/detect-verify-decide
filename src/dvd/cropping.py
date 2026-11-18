"""Crop-and-zoom: the step that gives the verifier evidence the detector's global pass never produced."""
# Postpone evaluation of type hints so `tuple[int, ...]` etc. also work on older Python versions.
from __future__ import annotations

# Pillow: the image library used to crop and resize (PIL.Image objects are our image type everywhere).
from PIL import Image


def expand_box(box, margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Expand an xyxy box by `margin` x (box w/h) on every side, clipped to the image."""
    # Unpack the box corners as floats (they may arrive as ints, numpy values or JSON numbers).
    x1, y1, x2, y2 = [float(v) for v in box]
    # Box width/height, forced to at least 1 pixel so a degenerate box still gives a usable margin.
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    # Grow the box outwards: the margin is a fraction of the box size, so big objects get more context.
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    # Round to whole pixels and clip the top-left corner to the image (>= 0) ...
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    # ... and the bottom-right corner to the image size.
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    # Guard against an empty box after clipping (e.g. a box fully outside the image): keep at least 1 px.
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_and_zoom(image: Image.Image, box, margin: float = 0.15, target_size: int = 448,
                  min_side: int = 28) -> Image.Image:
    """Crop the expanded box and upscale (or downscale) so the longer side == target_size."""
    w, h = image.size
    # Cut the expanded region out of the full image.
    crop = image.crop(expand_box(box, margin, w, h))
    cw, ch = crop.size
    # Single scale factor so the aspect ratio is preserved and the longer side hits `target_size`.
    scale = target_size / max(cw, ch)
    # New size; never smaller than `min_side` because the VLM needs at least one 28x28 patch.
    nw, nh = max(min_side, int(round(cw * scale))), max(min_side, int(round(ch * scale)))
    # Bicubic resampling gives the smoothest "zoom" for small crops.
    return crop.resize((nw, nh), Image.BICUBIC)


def fit_max_side(image: Image.Image, max_side: int, min_side: int = 28) -> Image.Image:
    """Downscale a full image so its longer side <= max_side (bounds VLM visual tokens)."""
    w, h = image.size
    # Already small enough: return it untouched (no needless resampling).
    if max(w, h) <= max_side:
        return image
    # Shrink factor that brings the longer side down to `max_side`.
    s = max_side / max(w, h)
    # Resize, keeping each side at least `min_side` pixels.
    return image.resize((max(min_side, round(w * s)), max(min_side, round(h * s))), Image.BICUBIC)
