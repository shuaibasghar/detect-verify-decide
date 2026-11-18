"""COCO val2017: download, held-out / test splits, image + ground-truth access."""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# random: seeded shuffle that creates the held-out / test split.
import random
# pathlib.Path: portable file paths.
from pathlib import Path

# numpy: ground-truth boxes and ids as arrays.
import numpy as np
# Pillow: opening the image files.
from PIL import Image

# JSON helpers (splits.json is saved and loaded with them).
from ..io_utils import load_json, save_json
# Download helpers (progress bar, retries, zip extraction).
from .download import download, download_and_extract

# Whole-split image zip (778 MB for val2017); only used when ALL images are wanted.
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/{split}.zip"
# URL of a single image; lets us download only the few hundred images a split needs.
COCO_IMG_URL = "http://images.cocodataset.org/{split}/{name}"
# Annotation zip (instances_train2017.json + instances_val2017.json + captions + keypoints).
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def prepare_annotations(cfg: dict) -> Path:
    """Annotations only: downloads the 241 MB annotation zip once (cached) and extracts just
    instances_<split>.json (not the 450 MB train annotations). Returns the COCO root directory."""
    split = cfg["coco"]["split"]
    root = Path(cfg["paths"]["data_root"]) / "coco"
    # The marker file (the extracted json) makes repeated calls free; `members` extracts only that one file.
    download_and_extract(COCO_ANN_URL, cfg["paths"]["download_cache"], root,
                         root / "annotations" / f"instances_{split}.json",
                         members=[f"annotations/instances_{split}.json"])
    return root


def image_file_name(image_id: int) -> str:
    """COCO file names are the zero-padded 12-digit image id."""
    return f"{int(image_id):012d}.jpg"


def fetch_images(cfg: dict, image_ids, workers: int = 16) -> None:
    """Make exactly these images available on local disk (fast: a few hundred files instead of the 778 MB zip).
    Uses the cached split zip if one exists (e.g. from a previous run), otherwise downloads them in parallel."""
    # Imported here because only this function needs them.
    from concurrent.futures import ThreadPoolExecutor

    from tqdm.auto import tqdm

    split = cfg["coco"]["split"]
    root = Path(cfg["paths"]["data_root"]) / "coco"
    img_dir = root / split
    img_dir.mkdir(parents=True, exist_ok=True)
    # Only the files that are not on disk yet.
    need = [image_file_name(i) for i in image_ids if not (img_dir / image_file_name(i)).exists()]
    if not need:
        return
    # A zip cached on Drive from an earlier run is much faster than downloading again.
    zpath = Path(cfg["paths"]["download_cache"]) / f"{split}.zip"
    if zpath.exists():
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            for n in tqdm(need, desc="extract images"):
                zf.extract(f"{split}/{n}", root)
        return
    # Otherwise download the images in parallel threads (network-bound, so threads are enough).
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(lambda n: download(COCO_IMG_URL.format(split=split, name=n), img_dir / n, quiet=True), need),
                  total=len(need), desc="download images"))


def prepare_coco(cfg: dict, image_ids=None) -> Path:
    """Ensure annotations + images are on local disk. Returns the COCO root directory.
    `image_ids=None` downloads the whole split (778 MB zip); otherwise only the listed images are fetched."""
    split = cfg["coco"]["split"]
    root = prepare_annotations(cfg)
    if image_ids is not None:
        fetch_images(cfg, image_ids)
    else:
        download_and_extract(COCO_IMAGES_URL.format(split=split), cfg["paths"]["download_cache"], root, root / split)
    return root


class CocoData:
    """Thin wrapper around pycocotools with xyxy boxes and category-name helpers."""

    def __init__(self, cfg: dict):
        # pycocotools is imported lazily so the rest of the package works without it installed.
        from pycocotools.coco import COCO

        split = cfg["coco"]["split"]
        self.root = Path(cfg["paths"]["data_root"]) / "coco"
        self.ann_file = self.root / "annotations" / f"instances_{split}.json"
        self.img_dir = self.root / split
        # Load the annotation file (a few seconds for val2017).
        self.coco = COCO(str(self.ann_file))
        # Categories sorted by id so index <-> id <-> name mappings are stable.
        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.cat_ids = [c["id"] for c in cats]
        self.cat_names = [c["name"] for c in cats]
        self.name_to_cat_id = {c["name"]: c["id"] for c in cats}
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}
        self.image_ids = sorted(self.coco.getImgIds())

    def info(self, image_id: int) -> dict:
        """COCO metadata of one image (file name, width, height, ...)."""
        return self.coco.loadImgs(image_id)[0]

    def image(self, image_id: int) -> Image.Image:
        """Load the image as RGB from the local disk."""
        return Image.open(self.img_dir / self.info(image_id)["file_name"]).convert("RGB")

    def gt(self, image_id: int) -> dict:
        """Ground truth of one image: xyxy boxes, category ids, crowd flags and areas (crowd boxes included)."""
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=None))
        # COCO stores boxes as [x, y, w, h]; convert to [x1, y1, x2, y2] used everywhere else.
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                           a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32).reshape(-1, 4)
        return {
            "boxes": boxes,
            "category_ids": np.array([a["category_id"] for a in anns], dtype=np.int64),
            # Crowd regions are "ignore" regions in the matching (see metrics.DetectionEvaluator).
            "iscrowd": np.array([bool(a.get("iscrowd", 0)) for a in anns], dtype=bool),
            "areas": np.array([a["area"] for a in anns], dtype=np.float32),
        }

    def gts(self, image_ids) -> dict:
        """Ground truth for many images: {image_id: gt dict}."""
        return {i: self.gt(i) for i in image_ids}


def make_splits(cfg: dict, image_ids: list[int]) -> dict:
    """Seeded held-out / test split of val2017, saved to results/splits.json (created once)."""
    path = Path(cfg["paths"]["results_root"]) / "splits.json"
    # The split is frozen after the first creation, so every notebook uses identical images.
    if path.exists():
        return load_json(path)
    ids = sorted(image_ids)
    # Deterministic shuffle (fixed seed) -> same split on every machine.
    random.Random(cfg["seed"]).shuffle(ids)
    n_h = cfg["coco"]["heldout_size"]
    n_t = cfg["coco"]["test_size"]
    # Held-out = first n_h shuffled ids; test = the next n_t (or all the rest if test_size is null).
    heldout = sorted(ids[:n_h])
    rest = ids[n_h:]
    test = sorted(rest if n_t is None else rest[:n_t])
    splits = {"heldout": heldout, "test": test, "seed": cfg["seed"]}
    save_json(splits, path)
    return splits


def load_splits(cfg: dict) -> dict:
    """Read the frozen split; fails with a clear message if notebook 01 has not been run."""
    path = Path(cfg["paths"]["results_root"]) / "splits.json"
    if not path.exists():
        raise FileNotFoundError("results/splits.json not found - run notebook 01 first.")
    return load_json(path)
