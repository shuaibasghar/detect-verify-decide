"""Canonical cache locations under results/. Names encode everything that changes the output."""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# pathlib.Path: object-style file paths that work the same on Windows, Linux and Colab.
from pathlib import Path

# results_dir creates (if needed) and returns a sub-folder of the active results folder.
from .config import results_dir


def _det_tag(cfg: dict) -> str:
    """Detector backend name (grounding_dino / yolo_world); part of every cache file name."""
    return cfg["detector"]["backend"]


def _vlm_tag(cfg: dict) -> str:
    """Short VLM name, e.g. 'Qwen2-VL-2B-Instruct' (the part of the model id after the '/')."""
    return cfg["verifier"]["model"].split("/")[-1]


def det_cache(cfg: dict, split: str, tag: str = "") -> Path:
    """Detector cache for a split (heldout / test / robust...); `tag` separates degraded variants."""
    return results_dir(cfg, "detections") / f"{_det_tag(cfg)}{tag}_{split}.jsonl"


def ver_cache(cfg: dict, split: str, view: str = "crop", mode: str | None = None, margin: float | None = None,
              tag: str = "") -> Path:
    """Verifier cache. The name contains detector, VLM, view, mode and (for crops) the margin, so changing
    any of them writes a new file instead of silently reusing an old one."""
    # Default to the configured verifier mode and the (possibly tuned) crop margin.
    mode = mode or cfg["verifier"]["mode"]
    margin = cfg["verifier"]["crop_margin"] if margin is None else margin
    # Only the crop view depends on the margin; the full-image view has none.
    m = f"_m{margin:.2f}" if view == "crop" else ""
    return results_dir(cfg, "verifier") / f"{_det_tag(cfg)}{tag}_{_vlm_tag(cfg)}_{view}_{mode}{m}_{split}.jsonl"


def woodpecker_cache(cfg: dict, split: str) -> Path:
    """Cache of the Woodpecker-style baseline (baseline ii)."""
    return results_dir(cfg, "baselines") / f"woodpecker_{_det_tag(cfg)}_{_vlm_tag(cfg)}_{split}.jsonl"


def vcd_cache(cfg: dict, split: str) -> Path:
    """Cache of the Visual Contrastive Decoding baseline (baseline iii)."""
    return results_dir(cfg, "baselines") / f"vcd_{_det_tag(cfg)}_{_vlm_tag(cfg)}_{split}.jsonl"


def pope_cache(cfg: dict, setting: str, method: str = "plain") -> Path:
    """Cache of POPE answers for one setting (random / popular / adversarial) and method (plain / vcd)."""
    return results_dir(cfg, "pope") / f"{_vlm_tag(cfg)}_{method}_{setting}.jsonl"
