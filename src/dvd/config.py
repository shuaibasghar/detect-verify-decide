"""Configuration loading, path resolution and tuned-parameter overrides."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import yaml


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def project_root() -> Path:
    """Project root: $DVD_ROOT if set (notebooks set it), else two levels above this package."""
    env = os.environ.get("DVD_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_paths(cfg: dict, root: Path) -> None:
    paths = cfg["paths"]
    for k, v in list(paths.items()):
        if v == "auto" and k == "data_root":
            v = "/content/data" if in_colab() else str(root / "data")
        paths[k] = str(v).replace("${ROOT}", str(root))
    for k in paths:
        Path(paths[k]).mkdir(parents=True, exist_ok=True)


def selected_params_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_root"]) / "tuning" / "selected.json"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None,
                use_selected: bool = True) -> dict:
    """Load configs/default.yaml, apply `overrides`, then held-out-tuned values.

    Tuned values (results/tuning/selected.json, written by notebooks 03 and 05) are applied
    last so every downstream notebook uses the same crop margin and thresholds.
    """
    root = project_root()
    path = Path(path) if path else root / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    _resolve_paths(cfg, root)
    cfg["root"] = str(root)

    cfg["selected"] = {}
    sel = selected_params_path(cfg)
    if use_selected and sel.exists():
        with open(sel, "r", encoding="utf-8") as f:
            cfg["selected"] = json.load(f)
        if "crop_margin" in cfg["selected"]:
            cfg["verifier"]["crop_margin"] = cfg["selected"]["crop_margin"]
    return cfg


def save_selected(cfg: dict, **values) -> dict:
    """Merge tuned values (e.g. crop_margin, tau_low, tau_high) into results/tuning/selected.json."""
    sel = selected_params_path(cfg)
    sel.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(sel.read_text(encoding="utf-8")) if sel.exists() else {}
    current.update(values)
    sel.write_text(json.dumps(current, indent=2), encoding="utf-8")
    cfg["selected"] = current
    if "crop_margin" in values:
        cfg["verifier"]["crop_margin"] = values["crop_margin"]
    return current


def results_dir(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["results_root"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
