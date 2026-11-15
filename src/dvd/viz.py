"""Plotting helpers: detections overlay, precision-recall frontier, crop galleries, cost curves."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .cropping import crop_and_zoom

STATUS_COLORS = {"accepted": "#2e7d32", "rejected": "#c62828", "verified": "#f9a825", None: "#1565c0"}


def show_detections(image, boxes, labels, scores, status=None, ax=None, title=None):
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    ax.imshow(image)
    for k, (b, l, s) in enumerate(zip(boxes, labels, scores)):
        st = status[k] if status is not None else None
        col = STATUS_COLORS.get(st, "#1565c0")
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2, ec=col,
                                       ls="-" if st != "rejected" else "--"))
        ax.text(b[0], b[1] - 2, f"{l} {s:.2f}", color="white", fontsize=8, bbox=dict(fc=col, ec="none", pad=1))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def plot_pr(rows_by_method: dict, ax=None, highlight: dict | None = None, title="Precision-recall frontier"):
    """rows_by_method: {name: [sweep rows]}. Draws each method's Pareto frontier plus its points."""
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    for name, rows in rows_by_method.items():
        pts = sorted({(r["recall"], r["precision"]) for r in rows})
        front, best = [], -1.0
        for rc, pr in sorted(pts, key=lambda x: -x[0]):
            if pr > best:
                front.append((rc, pr))
                best = pr
        front.sort()
        line, = ax.plot([p[0] for p in front], [p[1] for p in front], "-o", ms=3, label=name)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, alpha=0.25, color=line.get_color())
    for name, r in (highlight or {}).items():
        ax.scatter([r["recall"]], [r["precision"]], marker="*", s=180, edgecolor="k", zorder=5, label=f"{name} (selected)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def crop_gallery(items, image_fn, margin=0.15, ncols=6, title=None, size=448):
    """items: list of dicts with image_id, box, label, score, (optional) p_yes."""
    n = len(items)
    if n == 0:
        print("no items")
        return None
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(crop_and_zoom(image_fn(it["image_id"]), it["box"], margin, size))
        extra = f" p={it['p_yes']:.2f}" if it.get("p_yes") is not None else ""
        ax.set_title(f"{it['label']} c={it['score']:.2f}{extra}", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cost(rows: list[dict], x="total_latency_s", y="fp_removed", label_key="name", ax=None):
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    for r in rows:
        ax.scatter(r[x], r[y], s=40)
        ax.annotate(r[label_key], (r[x], r[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    return ax
