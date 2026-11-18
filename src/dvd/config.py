"""Configuration loading, path resolution and tuned-parameter overrides."""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# copy: deep copies, so applying overrides never mutates the caller's dictionary.
import copy
# json: read/write results/tuning/selected.json.
import json
# os: environment variables (DVD_ROOT, DVD_CONFIG, COLAB_RELEASE_TAG) and the PathLike type hint.
import os
# sys: to check whether the Colab module is loaded.
import sys
# pathlib.Path: portable file paths.
from pathlib import Path

# PyYAML: parses configs/*.yaml into Python dictionaries.
import yaml


def in_colab() -> bool:
    """True when running inside Google Colab (decides where the fast local data folder is)."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def in_kaggle() -> bool:
    """True when running inside a Kaggle notebook (it always has the /kaggle/working output folder)."""
    return not in_colab() and os.path.isdir("/kaggle/working")


def _kaggle_tmp() -> str:
    """Scratch folder on Kaggle that is NOT saved as notebook output (datasets and zips go here)."""
    return "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"


def project_root() -> Path:
    """Project root: $DVD_ROOT if set (notebooks set it), else two levels above this package."""
    env = os.environ.get("DVD_ROOT")
    if env:
        return Path(env).resolve()
    # This file is <root>/src/dvd/config.py, so parents[2] is <root>.
    return Path(__file__).resolve().parents[2]


def _deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into `base` (nested dicts are merged key by key, other values replaced)."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_paths(cfg: dict, root: Path) -> None:
    """Expand `auto` and `${ROOT}` in cfg["paths"] and create the folders."""
    paths = cfg["paths"]
    for k, v in list(paths.items()):
        # `auto` data root: fast local disk on Colab, scratch folder on Kaggle, inside the project elsewhere.
        if v == "auto" and k == "data_root":
            v = "/content/data" if in_colab() else (f"{_kaggle_tmp()}/dvd_data" if in_kaggle() else str(root / "data"))
        # Substitute the project root placeholder.
        paths[k] = str(v).replace("${ROOT}", str(root))
        # On Kaggle everything under /kaggle/working is saved as notebook output, so the big dataset zips go to scratch.
        if k == "download_cache" and in_kaggle():
            paths[k] = f"{_kaggle_tmp()}/dvd_downloads"
    for k in paths:
        Path(paths[k]).mkdir(parents=True, exist_ok=True)


def selected_params_path(cfg: dict) -> Path:
    """File holding the values tuned on held-out data (crop margin, thresholds)."""
    return Path(cfg["paths"]["results_root"]) / "tuning" / "selected.json"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None,
                use_selected: bool = True) -> dict:
    """Load configs/default.yaml, apply `overrides`, then held-out-tuned values.

    Tuned values (results/tuning/selected.json, written by notebooks 03 and 05) are applied
    last so every downstream notebook uses the same crop margin and thresholds.
    """
    root = project_root()
    if path is None:  # DVD_CONFIG=full selects configs/full.yaml (the slow profile); default = configs/default.yaml (fast)
        name = os.environ.get("DVD_CONFIG", "default")
        path = name if name.endswith((".yaml", ".yml")) else root / "configs" / f"{name}.yaml"
    path = Path(path)
    # Parse the YAML file into a dictionary.
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Programmatic overrides (e.g. a different results folder) beat the file.
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    _resolve_paths(cfg, root)
    cfg["root"] = str(root)

    # Values tuned on held-out data are applied last, so every notebook sees the same margin/thresholds.
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
    # Keep previously saved values; only the given keys are replaced.
    current = json.loads(sel.read_text(encoding="utf-8")) if sel.exists() else {}
    current.update(values)
    sel.write_text(json.dumps(current, indent=2), encoding="utf-8")
    # Make the new values visible immediately in the running session.
    cfg["selected"] = current
    if "crop_margin" in values:
        cfg["verifier"]["crop_margin"] = values["crop_margin"]
    return current


def results_dir(cfg: dict, *parts: str) -> Path:
    """Sub-folder of the results root (created if missing), e.g. results_dir(cfg, "verifier")."""
    p = Path(cfg["paths"]["results_root"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
