"""Canonical cache locations under results/. Names encode everything that changes the output."""
from __future__ import annotations

from pathlib import Path

from .config import results_dir


def _det_tag(cfg: dict) -> str:
    return cfg["detector"]["backend"]


def _vlm_tag(cfg: dict) -> str:
    return cfg["verifier"]["model"].split("/")[-1]


def det_cache(cfg: dict, split: str, tag: str = "") -> Path:
    return results_dir(cfg, "detections") / f"{_det_tag(cfg)}{tag}_{split}.jsonl"


def ver_cache(cfg: dict, split: str, view: str = "crop", mode: str | None = None, margin: float | None = None,
              tag: str = "") -> Path:
    mode = mode or cfg["verifier"]["mode"]
    margin = cfg["verifier"]["crop_margin"] if margin is None else margin
    m = f"_m{margin:.2f}" if view == "crop" else ""
    return results_dir(cfg, "verifier") / f"{_det_tag(cfg)}{tag}_{_vlm_tag(cfg)}_{view}_{mode}{m}_{split}.jsonl"


def woodpecker_cache(cfg: dict, split: str) -> Path:
    return results_dir(cfg, "baselines") / f"woodpecker_{_det_tag(cfg)}_{_vlm_tag(cfg)}_{split}.jsonl"


def vcd_cache(cfg: dict, split: str) -> Path:
    return results_dir(cfg, "baselines") / f"vcd_{_det_tag(cfg)}_{_vlm_tag(cfg)}_{split}.jsonl"


def pope_cache(cfg: dict, setting: str, method: str = "plain") -> Path:
    return results_dir(cfg, "pope") / f"{_vlm_tag(cfg)}_{method}_{setting}.jsonl"
