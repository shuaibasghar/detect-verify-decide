"""Regenerates notebooks/run_all_fast.ipynb (cells live in tools/nb_run_all_fast.py). Usage: python tools/build_notebooks.py notebooks"""
import json  # write the .ipynb files (they are JSON)
import sys  # command-line arguments of this script
import textwrap  # strip the indentation of the cell source strings
from pathlib import Path  # portable file paths

OUT = Path(sys.argv[1])
ONLY = set(sys.argv[2:])  # optional: notebook names to regenerate, e.g. run_all_fast
OUT.mkdir(parents=True, exist_ok=True)


def md(s):
    """Build a Markdown cell from text (indentation stripped, split into lines as nbformat expects)."""
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def code(s):
    """Build a code cell from source text (no outputs, never executed)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def notebook(cells, gpu=True):
    """Wrap cells into a complete notebook dict (nbformat 4); `gpu` asks Colab for a T4 accelerator by default."""
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


NB = {}
exec(compile((Path(__file__).parent / "nb_run_all_fast.py").read_text(encoding="utf-8"), "nb_run_all_fast.py", "exec"))

for name, cells in NB.items():
    if ONLY and name not in ONLY:
        continue
    gpu = name not in ("08_main_comparison_coco",)
    (OUT / f"{name}.ipynb").write_text(json.dumps(notebook(cells, gpu), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)
