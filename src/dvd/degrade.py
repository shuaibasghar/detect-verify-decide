"""Synthetic monitoring-style degradations (low light, motion blur, low resolution, JPEG artefacts).

Used by notebook 11 as a proxy for CCTV / traffic-camera footage when no annotated monitoring
subset is available: COCO ground truth stays valid because the geometry is unchanged.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# io: in-memory byte buffer, so the JPEG round trip never touches the disk.
import io

# numpy: pixel arithmetic (gamma, scaling, noise) on image arrays.
import numpy as np
# Pillow: the image type; ImageFilter provides the blur fallback.
from PIL import Image, ImageFilter


def low_light(img: Image.Image, factor: float = 0.3, gamma: float = 1.6, noise_std: float = 6.0, seed: int = 0):
    """Simulate a dark scene: darken with a gamma curve, scale the brightness down and add sensor noise."""
    # Convert to a float array in [0, 1].
    x = np.asarray(img.convert("RGB"), np.float32) / 255.0
    # Gamma > 1 crushes the shadows, `factor` lowers overall brightness.
    x = (x ** gamma) * factor
    # Back to 0..255 and add Gaussian noise (seeded, so the same image always gets the same noise).
    x = x * 255 + np.random.default_rng(seed).normal(0, noise_std, x.shape)
    # Clip to the valid range and convert to an 8-bit image.
    return Image.fromarray(x.clip(0, 255).astype(np.uint8))


def motion_blur(img: Image.Image, length: int = 15):
    """Simulate camera/object motion: horizontal line kernel of `length` pixels (OpenCV), Gaussian blur as fallback."""
    try:
        # OpenCV is optional (not in requirements.txt), hence the local import.
        import cv2
        # Kernel with a single horizontal line of equal weights = averaging along the motion direction.
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0 / length
        # filter2D convolves the image with the kernel (-1 keeps the input bit depth).
        return Image.fromarray(cv2.filter2D(np.asarray(img.convert("RGB")), -1, k))
    except ImportError:  # fallback: isotropic blur of similar strength
        return img.filter(ImageFilter.GaussianBlur(length / 4))


def low_res(img: Image.Image, scale: float = 0.25):
    """Simulate a low-resolution camera: shrink by `scale`, then enlarge back so the image size is unchanged."""
    w, h = img.size
    # Down-sample (information is lost here) ...
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    # ... and up-sample again so the ground-truth boxes still line up.
    return small.resize((w, h), Image.BILINEAR)


def jpeg(img: Image.Image, quality: int = 12):
    """Simulate heavy compression: encode as a very low-quality JPEG and decode again."""
    buf = io.BytesIO()
    # Encode into the in-memory buffer at low quality (visible blocking artefacts).
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    # Rewind the buffer so it can be read from the start.
    buf.seek(0)
    # Decode the compressed bytes back into an image.
    return Image.open(buf).convert("RGB")


# Name -> function lookup used by `degrade` and by the notebooks' list of conditions.
DEGRADATIONS = {"low_light": low_light, "motion_blur": motion_blur, "low_res": low_res, "jpeg": jpeg}


def degrade(img: Image.Image, condition: str, seed: int = 0) -> Image.Image:
    """Apply the named degradation; only low light is random, so only it uses `seed`."""
    if condition == "low_light":
        return low_light(img, seed=seed)
    return DEGRADATIONS[condition](img)
