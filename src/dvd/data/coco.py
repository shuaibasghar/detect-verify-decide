"""COCO val2017: download, held-out / test splits, image + ground-truth access."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..io_utils import load_json, save_json
from .download import download_and_extract

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/{split}.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def prepare_coco(cfg: dict) -> Path:
    """Ensure images + annotations are on local disk. Returns the COCO root directory."""
    split = cfg["coco"]["split"]
    root = Path(cfg["paths"]["data_root"]) / "coco"
    cache = cfg["paths"]["download_cache"]
    download_and_extract(COCO_ANN_URL, cache, root, root / "annotations" / f"instances_{split}.json")
    download_and_extract(COCO_IMAGES_URL.format(split=split), cache, root, root / split)
    return root


class CocoData:
    """Thin wrapper around pycocotools with xyxy boxes and category-name helpers."""

    def __init__(self, cfg: dict):
        from pycocotools.coco import COCO

        split = cfg["coco"]["split"]
        self.root = Path(cfg["paths"]["data_root"]) / "coco"
        self.ann_file = self.root / "annotations" / f"instances_{split}.json"
        self.img_dir = self.root / split
        self.coco = COCO(str(self.ann_file))
        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.cat_ids = [c["id"] for c in cats]
        self.cat_names = [c["name"] for c in cats]
        self.name_to_cat_id = {c["name"]: c["id"] for c in cats}
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}
        self.image_ids = sorted(self.coco.getImgIds())

    def info(self, image_id: int) -> dict:
        return self.coco.loadImgs(image_id)[0]

    def image(self, image_id: int) -> Image.Image:
        return Image.open(self.img_dir / self.info(image_id)["file_name"]).convert("RGB")

    def gt(self, image_id: int) -> dict:
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=None))
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                           a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32).reshape(-1, 4)
        return {
            "boxes": boxes,
            "category_ids": np.array([a["category_id"] for a in anns], dtype=np.int64),
            "iscrowd": np.array([bool(a.get("iscrowd", 0)) for a in anns], dtype=bool),
            "areas": np.array([a["area"] for a in anns], dtype=np.float32),
        }

    def gts(self, image_ids) -> dict:
        return {i: self.gt(i) for i in image_ids}


def make_splits(cfg: dict, image_ids: list[int]) -> dict:
    """Seeded held-out / test split of val2017, saved to results/splits.json (created once)."""
    path = Path(cfg["paths"]["results_root"]) / "splits.json"
    if path.exists():
        return load_json(path)
    ids = sorted(image_ids)
    random.Random(cfg["seed"]).shuffle(ids)
    n_h = cfg["coco"]["heldout_size"]
    n_t = cfg["coco"]["test_size"]
    heldout = sorted(ids[:n_h])
    rest = ids[n_h:]
    test = sorted(rest if n_t is None else rest[:n_t])
    splits = {"heldout": heldout, "test": test, "seed": cfg["seed"]}
    save_json(splits, path)
    return splits


def load_splits(cfg: dict) -> dict:
    path = Path(cfg["paths"]["results_root"]) / "splits.json"
    if not path.exists():
        raise FileNotFoundError("results/splits.json not found - run notebook 01 first.")
    return load_json(path)
