"""Synthetic monitoring-style degradations (low light, motion blur, low resolution, JPEG artefacts).

Used by notebook 11 as a proxy for CCTV / traffic-camera footage when no annotated monitoring
subset is available: COCO ground truth stays valid because the geometry is unchanged.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def low_light(img: Image.Image, factor: float = 0.3, gamma: float = 1.6, noise_std: float = 6.0, seed: int = 0):
    x = np.asarray(img.convert("RGB"), np.float32) / 255.0
    x = (x ** gamma) * factor
    x = x * 255 + np.random.default_rng(seed).normal(0, noise_std, x.shape)
    return Image.fromarray(x.clip(0, 255).astype(np.uint8))


def motion_blur(img: Image.Image, length: int = 15):
    try:
        import cv2
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0 / length
        return Image.fromarray(cv2.filter2D(np.asarray(img.convert("RGB")), -1, k))
    except ImportError:  # fallback: isotropic blur of similar strength
        return img.filter(ImageFilter.GaussianBlur(length / 4))


def low_res(img: Image.Image, scale: float = 0.25):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg(img: Image.Image, quality: int = 12):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


DEGRADATIONS = {"low_light": low_light, "motion_blur": motion_blur, "low_res": low_res, "jpeg": jpeg}


def degrade(img: Image.Image, condition: str, seed: int = 0) -> Image.Image:
    if condition == "low_light":
        return low_light(img, seed=seed)
    return DEGRADATIONS[condition](img)
