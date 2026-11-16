"""Detector Agent: a frozen pretrained open-vocabulary detector proposing candidates (b_i, c_i).

Backends
  * grounding_dino: Hugging Face `IDEA-Research/grounding-dino-*`. Its text encoder is limited
    to 256 tokens, so the 80 COCO class names are split into prompts of `classes_per_prompt`
    names ("person . bicycle . car ."), run one forward pass per prompt, mapped back to COCO
    classes, and merged with class-wise NMS.
  * yolo_world: Ultralytics YOLO-World with the 80 class names set as its vocabulary (1 pass).

Output record (one per image, JSON-serialisable):
  {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...], "category_ids": [...],
   "det_time_s": float, "det_invocations": int}
"""
from __future__ import annotations

import difflib
import inspect
import time

import numpy as np
from PIL import Image

from ..cost import Timer
from ..metrics import nms_per_class


def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def map_phrase(phrase: str, names: list[str]) -> int | None:
    """Map a Grounding DINO phrase (may be partial or merged) to an index in `names`."""
    p = phrase.strip().lower()
    if not p:
        return None
    if p in names:
        return names.index(p)
    hits = [i for i, n in enumerate(names) if n in p or p in n]
    if hits:
        return max(hits, key=lambda i: len(names[i]))
    close = difflib.get_close_matches(p, names, n=1, cutoff=0.6)
    return names.index(close[0]) if close else None


class GroundingDINODetector:
    def __init__(self, class_names: list[str], name_to_cat_id: dict, model_id: str,
                 box_threshold: float = 0.2, text_threshold: float = 0.2, classes_per_prompt: int = 20,
                 nms_iou: float = 0.7, max_detections: int = 100, device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = _device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.class_names = [n.lower() for n in class_names]
        self.name_to_cat_id = {k.lower(): v for k, v in name_to_cat_id.items()}
        self.chunks = [self.class_names[i:i + classes_per_prompt]
                       for i in range(0, len(self.class_names), classes_per_prompt)]
        self.prompts = [" . ".join(c) + " ." for c in self.chunks]
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.nms_iou, self.max_detections = nms_iou, max_detections
        params = inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self._thr_kw = "box_threshold" if "box_threshold" in params else "threshold"

    def _post(self, outputs, input_ids, size):
        kw = {self._thr_kw: self.box_threshold, "text_threshold": self.text_threshold, "target_sizes": [size]}
        res = self.processor.post_process_grounded_object_detection(outputs, input_ids, **kw)[0]
        labels = res.get("text_labels", res.get("labels"))
        return res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), list(labels)

    def detect(self, image: Image.Image, on_progress=None) -> dict:
        """`on_progress(step, n_steps, n_candidates_so_far)` is called after each text prompt."""
        image = image.convert("RGB")
        w, h = image.size
        all_b, all_s, all_c = [], [], []
        cb_time = 0.0
        with Timer() as t, self.torch.inference_mode():
            for step, (names, prompt) in enumerate(zip(self.chunks, self.prompts), 1):
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                boxes, scores, phrases = self._post(outputs, inputs["input_ids"], (h, w))
                for b, s, ph in zip(boxes, scores, phrases):
                    k = map_phrase(str(ph), names)
                    if k is None:
                        continue
                    all_b.append(b)
                    all_s.append(s)
                    all_c.append(self.class_names.index(names[k]))
                if on_progress:  # UI callback time is excluded from det_time_s
                    t_cb = time.perf_counter()
                    on_progress(step, len(self.chunks), len(all_s))
                    cb_time += time.perf_counter() - t_cb
        return self._finalize(all_b, all_s, all_c, t.elapsed - cb_time, len(self.prompts), w, h)

    def _finalize(self, boxes, scores, cls, elapsed, n_calls, w, h) -> dict:
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32)
        cls = np.asarray(cls, np.int64)
        if len(scores):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            keep = nms_per_class(boxes, scores, cls, self.nms_iou)[: self.max_detections]
            boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        labels = [self.class_names[c] for c in cls]
        return {"boxes": boxes.round(2).tolist(), "scores": scores.round(4).tolist(), "labels": labels,
                "category_ids": [int(self.name_to_cat_id[l]) for l in labels],
                "det_time_s": float(elapsed), "det_invocations": int(n_calls), "width": w, "height": h}


class YOLOWorldDetector(GroundingDINODetector):
    def __init__(self, class_names: list[str], name_to_cat_id: dict, model_id: str = "yolov8s-worldv2.pt",
                 box_threshold: float = 0.2, nms_iou: float = 0.7, max_detections: int = 100, device=None, **_):
        import torch
        from ultralytics import YOLOWorld

        self.torch = torch
        self.device = _device(device)
        self.model = YOLOWorld(model_id)
        self.class_names = [n.lower() for n in class_names]
        self.model.set_classes(self.class_names)
        self.name_to_cat_id = {k.lower(): v for k, v in name_to_cat_id.items()}
        self.box_threshold, self.nms_iou, self.max_detections = box_threshold, nms_iou, max_detections

    def detect(self, image: Image.Image, on_progress=None) -> dict:
        image = image.convert("RGB")
        w, h = image.size
        with Timer() as t:
            r = self.model.predict(image, conf=self.box_threshold, iou=self.nms_iou, max_det=self.max_detections,
                                   device=self.device, verbose=False)[0]
        b = r.boxes
        if on_progress:
            on_progress(1, 1, len(b))
        return self._finalize(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                              t.elapsed, 1, w, h)


def build_detector(cfg: dict, class_names: list[str], name_to_cat_id: dict, device=None):
    d = cfg["detector"]
    if d["backend"] == "grounding_dino":
        return GroundingDINODetector(class_names, name_to_cat_id, d["grounding_dino_model"], d["box_threshold"],
                                     d["text_threshold"], d["classes_per_prompt"], d["nms_iou"],
                                     d["max_detections"], device)
    if d["backend"] == "yolo_world":
        return YOLOWorldDetector(class_names, name_to_cat_id, d["yolo_world_model"], d["box_threshold"],
                                 d["nms_iou"], d["max_detections"], device)
    raise ValueError(f"Unknown detector backend: {d['backend']}")
