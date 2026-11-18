"""Baseline (iii): Visual Contrastive Decoding (Leng et al., CVPR 2024) for yes/no existence answers.

For a single-token yes/no answer, VCD reduces to one contrastive step:
    cd_logits = (1 + alpha) * logits(v) - alpha * logits(v')        v' = diffusion-noised image
    plausibility: tokens with logits(v) < log(beta) + max logits(v) are masked to -inf
    P_vcd(yes) = softmax over the (masked) yes/no token groups
The noise follows the official VCD code: q(x_t | x_0) with a 1000-step sigmoid beta schedule,
applied to the CLIP-normalised image (Qwen2-VL uses the same normalisation), at `noise_step` = 500.

Applied to detections: for each image, every distinct candidate label is queried on the FULL image
("Is there a dog in the image?"), and a candidate is kept iff VCD answers yes for its label.
The same forward passes also give the plain (non-contrastive) global answer for free.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# numpy: the noise schedule and the pixel arithmetic.
import numpy as np
# Pillow: the image type.
from PIL import Image
# tqdm: progress bar over images.
from tqdm.auto import tqdm

# `article` gives "a"/"an" for the question text.
from ..agents.verifier import article
# CUDA-synchronised stopwatch.
from ..cost import Timer
# Shrinks the full image to the configured size.
from ..cropping import fit_max_side
# Resumable per-image JSONL cache.
from ..io_utils import JsonlCache

# CLIP normalisation constants: the diffusion noise is added in this normalised space (as in official VCD).
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def diffusion_noise(image: Image.Image, noise_step: int = 500, seed: int = 0) -> Image.Image:
    """Forward diffusion q(x_t | x_0): the image blended with Gaussian noise at step `noise_step` of 1000."""
    # Sigmoid beta schedule from the official VCD code.
    betas = 1 / (1 + np.exp(-np.linspace(-6, 6, 1000))) * (0.5e-2 - 1e-5) + 1e-5
    # Cumulative signal-retention factor at the chosen step.
    a_bar = np.cumprod(1 - betas)[noise_step]
    # Normalise the pixels like CLIP does.
    x = (np.asarray(image.convert("RGB"), np.float64) / 255.0 - CLIP_MEAN) / CLIP_STD
    # Seeded Gaussian noise (the same image always gets the same noise).
    eps = np.random.default_rng(seed).standard_normal(x.shape)
    # x_t = sqrt(a_bar) * x + sqrt(1 - a_bar) * noise
    xt = np.sqrt(a_bar) * x + np.sqrt(1 - a_bar) * eps
    # Undo the normalisation and convert back to an 8-bit image.
    return Image.fromarray(((xt * CLIP_STD + CLIP_MEAN).clip(0, 1) * 255).astype(np.uint8))


def vcd_yes_no(verifier, images: list, questions: list[str], alpha: float = 1.0, beta: float = 0.1,
               noise_step: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (P_vcd(yes), P_plain(yes)) for each (image, question)."""
    t = verifier.torch
    noisy_cache = {}
    noisy = []
    for im in images:  # the same image object is noised once
        if id(im) not in noisy_cache:
            noisy_cache[id(im)] = diffusion_noise(im, noise_step, seed)
        noisy.append(noisy_cache[id(im)])
    p_vcd, p_plain = [], []
    bs = verifier.batch_size
    for s in range(0, len(images), bs):
        # Logits with the clean image (lo) and with the noised image (ln).
        lo = verifier._last_logits(images[s:s + bs], questions[s:s + bs])
        ln = verifier._last_logits(noisy[s:s + bs], questions[s:s + bs])
        # Contrastive logits: amplify what the clean image supports, subtract what the noised image also "sees".
        cd = (1 + alpha) * lo - alpha * ln
        # Adaptive plausibility cut-off: only tokens reasonably likely under the clean image may survive.
        cutoff = np.log(beta) + lo.max(dim=-1, keepdim=True).values
        cd = cd.masked_fill(lo < cutoff, float("-inf"))
        # Yes/no probability from the contrastive logits (same formula as the plain verifier).
        ly = t.logsumexp(cd[:, verifier.yes_ids], dim=-1)
        ln_ = t.logsumexp(cd[:, verifier.no_ids], dim=-1)
        plain = verifier._yes_no_from_logits(lo)
        p = t.sigmoid(ly - ln_).cpu().numpy()
        p = np.where(np.isnan(p), plain, p)  # both answers implausible -> fall back to plain decoding
        p_vcd.append(p)
        p_plain.append(plain)
    return np.concatenate(p_vcd), np.concatenate(p_plain)


def run_vcd_stage(verifier, det_records: dict, image_ids, cache_path, cfg: dict, image_fn,
                  desc: str = "vcd") -> dict:
    """Per image: query every distinct candidate label with VCD and with plain decoding (resumable)."""
    c, v = cfg["vcd"], cfg["verifier"]
    cache = JsonlCache(cache_path)
    for iid in tqdm(cache.todo(image_ids), desc=desc):
        # One question per distinct label (not per candidate).
        labels = sorted(set(det_records[iid]["labels"]))
        rec = {"image_id": int(iid), "labels": labels, "p_vcd": [], "p_plain": [], "time_s": 0.0,
               "invocations": 2 * len(labels)}
        if labels:
            image = fit_max_side(image_fn(iid), v["full_image_max_side"])
            qs = [v["prompt_existence"].format(article=article(l), label=l) for l in labels]
            with Timer() as tm:
                pv, pp = vcd_yes_no(verifier, [image] * len(labels), qs, c["alpha"], c["beta"], c["noise_step"],
                                    seed=int(iid))
            rec.update(p_vcd=pv.round(4).tolist(), p_plain=pp.round(4).tolist(), time_s=tm.elapsed)
        cache.add(rec)
    return cache.records(image_ids)


def candidate_verdicts(det_records: dict, vcd_records: dict, key: str = "p_vcd") -> dict:
    """Map per-label VCD answers back onto every candidate of the image."""
    out = {}
    for iid, det in det_records.items():
        r = vcd_records[iid]
        # label -> (P(yes) > 0.5) lookup for this image.
        lut = {l: p > 0.5 for l, p in zip(r["labels"], r[key])}
        out[iid] = np.array([lut.get(l, False) for l in det["labels"]], bool)
    return out
