"""Verifier Agent: a frozen small VLM (default Qwen2-VL-2B-Instruct) that re-examines each candidate.

Independence by construction:
  * it sees only the crop-and-zoomed region (view="crop"), never the full scene;
  * it is never shown the detector's confidence score;
  * the query is neutral and POPE-style ("Is there a dog in the image? Answer yes or no.");
  * mode="open" is fully label-blind: the VLM names the object and the name is matched to the
    candidate label afterwards.

`yes_no` scores the answer from a single forward pass: P(yes) = softmax over the summed
logits of the "Yes"/"yes" vs "No"/"no" first tokens at the answer position. Verdict = P(yes) > 0.5.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# re: whole-word matching of the open-mode answer against label names.
import re

# numpy: probabilities as arrays.
import numpy as np
# Pillow: the image type.
from PIL import Image

# crop_and_zoom builds the verifier's evidence; fit_max_side shrinks full images.
from ..cropping import crop_and_zoom, fit_max_side

# Extra words that count as a match for a COCO label in open mode (e.g. "man" for "person").
SYNONYMS = {
    "person": ["man", "woman", "people", "boy", "girl", "child", "kid", "player", "human", "lady", "guy", "skier",
               "surfer", "baby", "rider"],
    "motorcycle": ["motorbike", "scooter", "moped"], "airplane": ["plane", "aircraft", "jet", "airliner"],
    "couch": ["sofa"], "tv": ["television", "monitor", "screen"], "cell phone": ["phone", "smartphone", "mobile"],
    "dining table": ["table"], "potted plant": ["plant", "flower pot", "houseplant"], "laptop": ["computer", "notebook"],
    "bicycle": ["bike"], "car": ["vehicle", "sedan", "taxi"], "hair drier": ["hair dryer", "blow dryer"],
    "sports ball": ["ball", "football", "soccer ball", "tennis ball", "baseball", "basketball"],
    "handbag": ["bag", "purse"], "backpack": ["bag", "rucksack"], "donut": ["doughnut"],
    "mouse": ["computer mouse"], "remote": ["remote control", "controller"], "cup": ["mug"],
    "wine glass": ["glass"], "bench": ["seat"], "truck": ["lorry"], "boat": ["ship"],
}


def article(label: str) -> str:
    """'an' before a vowel, else 'a' (so the prompt reads 'Is there an apple...')."""
    return "an" if label[:1].lower() in "aeiou" else "a"


def _has_words(text: str, phrase: str) -> bool:
    """Whole-word (optionally plural) match of `phrase` inside `text`, so 'car' does not match 'carrot'."""
    return re.search(rf"\b{re.escape(phrase)}s?\b", text) is not None


def label_matches(answer: str, label: str) -> bool:
    """Whole-word match of the open answer against the label or its synonyms ('car' != 'carrot')."""
    # Normalise: lower case, strip punctuation/whitespace.
    a = answer.lower().strip(" .!\n")
    # The label itself plus its synonyms.
    names = [label.lower()] + SYNONYMS.get(label.lower(), [])
    # Match in either direction (answer contains name, or name contains answer); empty answers never match.
    return bool(a) and any(_has_words(a, n) or _has_words(n, a) for n in names)


def _pick_dtype(torch):
    """Best number format for the GPU: bf16 if supported, else fp16 (T4); fp32 on the CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class VLMVerifier:
    """Qwen2-VL wrapped as the Verifier Agent (frozen: inference only, no training)."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", batch_size: int = 8, device=None,
                 min_pixels: int = 28 * 28 * 4, max_pixels: int = 896 * 896, dtype=None):
        # Heavy libraries are imported here, so importing this module stays cheap.
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or _pick_dtype(torch)
        # Crops processed together in one forward pass.
        self.batch_size = batch_size
        # min/max_pixels bound the number of visual tokens per image (28x28 pixels = 1 token).
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"  # batched last-position logits / generation
        try:
            # Newer transformers expose a generic image-text-to-text class ...
            from transformers import AutoModelForImageTextToText as VLM
            self.model = VLM.from_pretrained(model_id, torch_dtype=self.dtype)
        except (ImportError, ValueError):
            # ... older ones need the Qwen2-VL specific class.
            from transformers import Qwen2VLForConditionalGeneration as VLM
            self.model = VLM.from_pretrained(model_id, torch_dtype=self.dtype)
        # Move to the GPU and switch to eval mode (disables dropout; weights are never updated).
        self.model.to(self.device).eval()
        tok = self.processor.tokenizer
        # First-token ids of the spellings of "yes" and "no": their logits define P(yes).
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes", "YES")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no", "NO")})

    # ------------------------------------------------------------------ low-level
    def _inputs(self, images, prompts):
        """Build model inputs: chat-template text (with an image placeholder) + preprocessed images."""
        texts = []
        for p, im in zip(prompts, images):
            # `None` image = text-only prompt (used by the LLM aggregator).
            content = ([{"type": "image"}] if im is not None else []) + [{"type": "text", "text": p}]
            texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                            tokenize=False, add_generation_prompt=True))
        imgs = [im for im in images if im is not None]
        kw = {"images": imgs} if imgs else {}
        # Pad the batch (left padding) and move the tensors to the device.
        return self.processor(text=texts, padding=True, return_tensors="pt", **kw).to(self.device)

    def _last_logits(self, images, prompts):
        """Logits of the next token after the prompt (the answer position), one row per input."""
        inputs = self._inputs(images, prompts)
        with self.torch.inference_mode():
            try:
                # `logits_to_keep=1` computes logits only for the last position (saves memory/time).
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                # Older transformers do not know that argument.
                out = self.model(**inputs)
        return out.logits[:, -1, :].float()

    def _yes_no_from_logits(self, logits):
        """P(yes) = sigmoid(logsumexp(yes logits) - logsumexp(no logits)), per row."""
        t = self.torch
        ly = t.logsumexp(logits[:, self.yes_ids], dim=-1)
        ln = t.logsumexp(logits[:, self.no_ids], dim=-1)
        return t.sigmoid(ly - ln).cpu().numpy()

    # ------------------------------------------------------------------ public API
    def yes_no(self, images: list[Image.Image], questions: list[str]) -> np.ndarray:
        """P(yes) for each (image, question) pair, batched."""
        out = []
        for s in range(0, len(images), self.batch_size):
            logits = self._last_logits(images[s:s + self.batch_size], questions[s:s + self.batch_size])
            out.append(self._yes_no_from_logits(logits))
        return np.concatenate(out) if out else np.zeros(0)

    def generate(self, images: list, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
        """Greedy text generation for (image, prompt) pairs (used by open mode and the Woodpecker-style baseline)."""
        outs = []
        for s in range(0, len(prompts), self.batch_size):
            inputs = self._inputs(images[s:s + self.batch_size], prompts[s:s + self.batch_size])
            with self.torch.inference_mode():
                # do_sample=False -> deterministic greedy decoding.
                ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            # Drop the prompt tokens; keep only the newly generated ones.
            ids = ids[:, inputs["input_ids"].shape[1]:]
            outs += self.processor.batch_decode(ids, skip_special_tokens=True)
        return [o.strip() for o in outs]

    def generate_text(self, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
        """Text-only generation (no image), used by the LLM aggregator."""
        return self.generate([None] * len(prompts), prompts, max_new_tokens)

    def verify(self, views: list[Image.Image], labels: list[str], mode: str = "existence",
               prompt_existence: str = "Is there {article} {label} in the image? Answer yes or no.",
               prompt_open: str = "What is the main object in this image? Answer with one or two words.",
               open_max_new_tokens: int = 10) -> dict:
        """Verdicts for prepared views (crops or full images). Returns p_yes, verdict, answers."""
        if mode == "existence":
            # Neutral POPE-style question per candidate label; the detector confidence is never included.
            qs = [prompt_existence.format(article=article(l), label=l) for l in labels]
            p = self.yes_no(views, qs)
            return {"p_yes": p.round(4).tolist(), "verdict": (p > 0.5).tolist(), "answers": None}
        if mode == "open":
            # Label-blind: ask what the object is, then compare the answer with the candidate label afterwards.
            ans = self.generate(views, [prompt_open] * len(views), open_max_new_tokens)
            v = [label_matches(a, l) for a, l in zip(ans, labels)]
            return {"p_yes": [float(x) for x in v], "verdict": v, "answers": ans}
        raise ValueError(f"Unknown verifier mode: {mode}")


def build_verifier(cfg: dict, device=None) -> VLMVerifier:
    """Factory: create the Verifier Agent from cfg["verifier"]."""
    v = cfg["verifier"]
    # Token budget must allow the larger of the crop zoom size and the full-image size.
    max_side = max(v["crop_target_size"], v["full_image_max_side"])
    return VLMVerifier(v["model"], v["batch_size"], device, min_pixels=v["min_crop_side"] ** 2 * 4,
                       max_pixels=max_side * max_side)


def make_views(image: Image.Image, boxes, cfg: dict, view: str | None = None, margin: float | None = None) -> list:
    """Crop-and-zoom views (proposed) or the downscaled full image repeated (global ablation)."""
    v = cfg["verifier"]
    view = view or v["view"]
    margin = v["crop_margin"] if margin is None else margin
    if view == "crop":
        # One zoomed crop per candidate box.
        return [crop_and_zoom(image, b, margin, v["crop_target_size"], v["min_crop_side"]) for b in boxes]
    if view == "full":
        # Same downscaled full image for every candidate (only the question differs).
        full = fit_max_side(image, v["full_image_max_side"], v["min_crop_side"])
        return [full] * len(boxes)
    raise ValueError(f"Unknown view: {view}")
