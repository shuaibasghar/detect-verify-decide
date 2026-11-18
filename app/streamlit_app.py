"""Streamlit front-end for Detect-Verify-Decide.

Shows, live, which agent is running, the current step, a progress percentage and a feedback log,
then the final detections with the reason behind every accept / reject.

Run:  streamlit run app/streamlit_app.py
      (on Colab use notebooks/12_streamlit_demo.ipynb, which also opens a public tunnel)

Without torch (e.g. a laptop) or with DVD_MOCK=1 it runs in MOCK mode: fake agents, random outputs,
only for previewing the UI.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# copy: deep copies of the config so slider values never change the shared config.
import copy
# os: environment variables (DVD_ROOT, DVD_MOCK).
import os
# sys: extend the import path so `import dvd` works.
import sys
# time: stopwatch for the per-agent durations and the log timestamps.
import time
# traceback: show the full error text inside the page if something fails.
import traceback
# urllib.request: download the COCO sample image.
import urllib.request
# pathlib.Path: portable file paths.
from pathlib import Path

# Project root = the folder above app/.
ROOT = Path(__file__).resolve().parents[1]
# Tell dvd.config where the project is (does not override a value that is already set).
os.environ.setdefault("DVD_ROOT", str(ROOT))
# Make src/ importable without installing the package.
sys.path.insert(0, str(ROOT / "src"))

# numpy: imported for completeness of the numeric stack (the app mostly uses lists / pandas).
import numpy as np  # noqa: E402
# pandas: the "All candidates" table.
import pandas as pd  # noqa: E402
# streamlit: the web UI framework (widgets, layout, live updates).
import streamlit as st  # noqa: E402
# Pillow: the image type, ImageDraw/ImageFont to draw boxes and labels on the result image.
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# The decision rule object used by the pipeline.
from dvd.agents.decision import RuleDecisionAgent  # noqa: E402
# make_views builds the crops shown in the "Verifier crops" tab.
from dvd.agents.verifier import make_views  # noqa: E402
# The 80 COCO classes (detector vocabulary).
from dvd.coco_classes import COCO_NAME_TO_ID, COCO_NAMES  # noqa: E402
# Config loader (reads configs/default.yaml and tuned values).
from dvd.config import load_config  # noqa: E402
# Downscale very large uploads.
from dvd.cropping import fit_max_side  # noqa: E402
# Fake agents and the synthetic image for the offline UI preview.
from dvd.mock_agents import MockDetector, MockVerifier, synthetic_image  # noqa: E402
# The online pipeline that emits the progress events this UI displays.
from dvd.pipeline import DetectVerifyDecide  # noqa: E402

# Must be the first Streamlit call: browser-tab title, icon and wide layout.
st.set_page_config(page_title="Detect-Verify-Decide", page_icon="🔎", layout="wide")

# A COCO image (two cats on a couch) used by the "Use a COCO sample image" button.
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Uploads larger than this (long side, pixels) are downscaled to keep the demo responsive.
MAX_SIDE = 1600
AGENTS = [  # (event stage, title, idle description)
    ("detector", "Detector Agent", "Grounding DINO proposes candidate boxes"),
    ("gate", "Decision gate", "splits candidates by confidence band"),
    ("crop", "Crop and zoom", "cuts out each uncertain candidate"),
    ("verifier", "Verifier Agent", "frozen VLM re-checks each crop"),
    ("decision", "Decision Agent", "fuses confidence with the verdict"),
]
# Status icons of the five agent cards.
ICONS = {"waiting": "⏳", "running": "🔄", "done": "✅", "skipped": "⏭️"}
# Box colours (RGB): green accepted, red rejected by the verifier, grey rejected for low confidence.
COLORS = {"accepted": (46, 125, 50), "rejected": (198, 40, 40), "auto-rejected": (130, 130, 130)}


def _torch_available() -> bool:
    """True if PyTorch can be imported (if not, the app falls back to mock mode)."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# Mock mode when forced by the environment or when there is no torch.
USE_MOCK = os.environ.get("DVD_MOCK") == "1" or not _torch_available()
# Configuration used for sliders' defaults and for the pipeline.
CFG = load_config()


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner=False)
def _load_detector(mock: bool):
    """Create the Detector Agent once per server process (cache_resource keeps it across page reruns)."""
    if mock:
        return MockDetector(delay=0.35)
    from dvd.agents.detector import build_detector
    return build_detector(load_config(), COCO_NAMES, COCO_NAME_TO_ID)


@st.cache_resource(show_spinner=False)
def _load_verifier(mock: bool):
    """Create the Verifier Agent once per server process."""
    if mock:
        return MockVerifier(delay=0.25)
    from dvd.agents.verifier import build_verifier
    return build_verifier(load_config())


def ensure_models() -> None:
    """Load both frozen models once (cached across reruns), with a visible progress bar."""
    # Already loaded in this browser session: nothing to show.
    if st.session_state.get("models_ready"):
        return
    try:
        with st.status("Loading models (first start only)...", expanded=True) as status:
            bar = st.progress(0, text="0% · starting")
            bar.progress(5, text="5% · loading Detector Agent (Grounding DINO); weights download on first use")
            _load_detector(USE_MOCK)
            bar.progress(50, text="50% · loading Verifier Agent (Qwen2-VL-2B)")
            _load_verifier(USE_MOCK)
            bar.progress(100, text="100% · models ready")
            status.update(label="Models loaded", state="complete", expanded=False)
    except Exception:  # noqa: BLE001
        # Show the error in the page instead of a blank screen, then stop the script.
        st.error("Could not load the models.")
        st.code(traceback.format_exc())
        st.stop()
    st.session_state["models_ready"] = True


# ------------------------------------------------------------------ images
def fetch_sample() -> Image.Image:
    """Download the COCO sample image; fall back to a synthetic image if the network is unavailable."""
    try:
        with urllib.request.urlopen(SAMPLE_URL, timeout=20) as r:
            import io
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:  # noqa: BLE001
        st.sidebar.warning("Could not download the COCO sample; using a synthetic image.")
        return synthetic_image()


def _font(size: int):
    """Default Pillow font at the requested size (older Pillow versions cannot scale it)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def annotate(image: Image.Image, rows: list[dict], show_rejected: bool) -> Image.Image:
    """Draw every candidate on a copy of the image (line width and font scale with the image size)."""
    img = image.copy()
    d = ImageDraw.Draw(img)
    side = max(img.size)
    lw, font = max(2, side // 350), _font(max(12, side // 45))
    for r in rows:
        # Colour key: accepted / rejected (by verifier) / auto-rejected (low confidence).
        key = r["decision"] if r["route"] != "auto-rejected" else "auto-rejected"
        # Skip rejected boxes if the user switched them off.
        if key != "accepted" and not show_rejected:
            continue
        col = COLORS[key]
        x1, y1, x2, y2 = r["box"]
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        # Label text: class and detector confidence, plus P(yes) when the verifier looked at it.
        text = f"{r['label']} {r['conf']:.2f}" + (f" P={r['p_yes']:.2f}" if r["route"] == "verified" else "")
        tb = d.textbbox((0, 0), text, font=font)
        th, tw = tb[3] - tb[1], tb[2] - tb[0]
        # Put the label just above the box (but inside the image).
        ty = max(0, y1 - th - 4)
        d.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col)
        d.text((x1 + 3, ty), text, fill=(255, 255, 255), font=font)
    return img


# ------------------------------------------------------------------ live status
def new_state() -> dict:
    """Fresh status of the five agent cards (all waiting)."""
    return {k: {"status": "waiting", "t0": None, "dur": None, "note": ""} for k, _, _ in AGENTS}


def _close(state: dict, key: str, now: float) -> None:
    """Mark a running stage as done and record how long it ran."""
    s = state[key]
    if s["status"] == "running":
        s.update(status="done", dur=now - s["t0"])


def render_timeline(placeholder, state: dict) -> None:
    """Redraw the row of agent cards (icon, title, current note, duration) inside `placeholder`."""
    with placeholder.container():
        for col, (key, title, desc) in zip(st.columns(len(AGENTS)), AGENTS):
            s = state[key]
            with col:
                st.markdown(f"**{ICONS[s['status']]} {title}**")
                # Show the live note while running, the idle description otherwise.
                st.caption(s["note"] or desc)
                if s["dur"] is not None:
                    st.caption(f"{s['dur']:.2f} s")


def run_pipeline(image, tau_low, tau_high, margin, progress_ph, timeline_ph, log_ph) -> dict:
    """Run the pipeline on one image and update the progress bar, agent cards and log from its events."""
    # Copy the config so the margin slider only affects this run.
    cfg = copy.deepcopy(CFG)
    cfg["verifier"]["crop_margin"] = margin
    pipe = DetectVerifyDecide(_load_detector(USE_MOCK), _load_verifier(USE_MOCK),
                              RuleDecisionAgent(tau_low, tau_high), cfg)
    # state = card status, log = text lines, cur = stage currently running, t_start = run start time.
    state, log, cur, t_start = new_state(), [], {"key": None}, time.perf_counter()

    def on_event(ev: dict) -> None:
        """Callback invoked by the pipeline for every progress event."""
        now, key = time.perf_counter(), ev["stage"]
        if key == "done":
            # Close whatever stage was still running.
            if cur["key"]:
                _close(state, cur["key"], now)
        else:
            # A new stage begins: close the previous one and open this one.
            if key != cur["key"]:
                if cur["key"]:
                    _close(state, cur["key"], now)
                state[key].update(status="running", t0=now)
                cur["key"] = key
            state[key]["note"] = ev["msg"]
            # Nothing in the uncertain band: the crop and verifier stages are skipped.
            if key == "gate" and ev.get("n_verify") == 0:
                for k in ("crop", "verifier"):
                    state[k].update(status="skipped", note="no candidate in the uncertain band")
        # Progress bar with percentage and message.
        pct = int(round(ev["progress"] * 100))
        progress_ph.progress(pct, text=f"{pct}% · {ev['msg']}")
        log.append(f"[{now - t_start:6.2f}s] {ev['msg']}")
        # Per-crop verifier answers are listed under the event line.
        for r in ev.get("results", []):
            log.append(f"           {r['label']} (c={r['score']:.2f}) -> P(yes)={r['p_yes']:.2f} => "
                       f"{'yes, keep' if r['verdict'] else 'no, reject'}")
        render_timeline(timeline_ph, state)
        # Show only the latest 14 log lines to keep the page compact.
        log_ph.code("\n".join(log[-14:]), language=None)

    result = pipe(image, on_event=on_event)
    return {"image": image, "result": result, "tau_low": tau_low, "tau_high": tau_high, "margin": margin,
            "state": state, "log": log}


# ------------------------------------------------------------------ results
def build_rows(r: dict, tau_low: float, tau_high: float) -> list[dict]:
    """One display row per candidate: how it was routed (verified / auto-accepted / auto-rejected) and why."""
    rows = []
    for k, (box, label, c) in enumerate(zip(r["boxes"], r["labels"], r["scores"])):
        p = r["p_yes"][k]
        if r["verified"][k]:
            # In the uncertain band: the verifier's verdict decided.
            route, ok = "verified", r["accept"][k]
            reason = f"verifier said {'yes' if r['verdict'][k] else 'no'} (P(yes)={p:.2f})"
        elif c >= tau_high:
            route, ok, reason = "auto-accepted", True, f"detector confident (c >= {tau_high:.2f})"
        else:
            route, ok, reason = "auto-rejected", False, f"detector confidence too low (c < {tau_low:.2f})"
        rows.append({"idx": k + 1, "label": label, "conf": float(c), "box": box, "route": route,
                     "p_yes": p, "decision": "accepted" if ok else "rejected", "reason": reason})
    return rows


def render_results(res: dict) -> None:
    """Show the summary, metrics and the four result tabs for a finished run."""
    r, image = res["result"], res["image"]
    rows = build_rows(r, res["tau_low"], res["tau_high"])
    if not rows:
        st.info("The detector found no candidates above its threshold in this image.")
        return
    # Counts for the one-line summary.
    n_final = sum(x["decision"] == "accepted" for x in rows)
    n_verified = sum(x["route"] == "verified" for x in rows)
    n_removed = sum(x["route"] == "verified" and x["decision"] == "rejected" for x in rows)
    n_alone = sum(x["route"] != "auto-rejected" for x in rows)
    st.info(f"Detector alone (c >= τ_low) would report **{n_alone}** objects. The pipeline reports "
            f"**{n_final}**; the verifier removed **{n_removed}** of the **{n_verified}** it checked.")

    # Row 1 of metrics: candidate counts.
    a = st.columns(4)
    a[0].metric("Candidates", len(rows))
    a[1].metric("Auto-accepted", sum(x["route"] == "auto-accepted" for x in rows))
    a[2].metric("Verified on crops", n_verified)
    a[3].metric("Final detections", n_final)
    # Row 2 of metrics: timings (UI refresh time is already excluded by the pipeline).
    b = st.columns(5)
    b[0].metric("Detector", f"{r['det_time_s']:.2f} s")
    b[1].metric("Verifier", f"{r['verify_time_s']:.2f} s")
    b[2].metric("Decision", f"{r['decision_time_s'] * 1e3:.2f} ms")
    b[3].metric("Total", f"{r['total_time_s']:.2f} s")
    b[4].metric("Throughput", f"{1 / r['total_time_s']:.2f} img/s" if r["total_time_s"] > 0 else "n/a")

    tab_img, tab_crops, tab_tbl, tab_log = st.tabs(["Result", "Verifier crops", "All candidates", "Run log"])
    with tab_img:
        show_rej = st.checkbox("Show rejected candidates", value=True, key="show_rej")
        left, right = st.columns(2)
        left.image(image, caption="Input", width="stretch")
        right.image(annotate(image, rows, show_rej), width="stretch",
                    caption="Green = accepted · red = rejected by the verifier · grey = rejected, low confidence")
    with tab_crops:
        # Only the candidates that were actually sent to the verifier have a crop.
        ver = [x for x in rows if x["route"] == "verified"]
        if not ver:
            st.info("Nothing fell in the uncertain band, so the verifier was not used.")
        else:
            cfg = copy.deepcopy(CFG)
            cfg["verifier"]["crop_margin"] = res["margin"]
            # Rebuild the exact crops the verifier saw (same margin and zoom).
            views = make_views(image, [x["box"] for x in ver], cfg, "crop")
            st.caption("Exactly what the verifier saw: the label-neutral crop, never the detector confidence.")
            cols = st.columns(4)
            for i, (x, v) in enumerate(zip(ver, views)):
                cols[i % 4].image(v, width="stretch", caption=(
                    f"{x['label']} · c={x['conf']:.2f} · P(yes)={x['p_yes']:.2f} → "
                    f"{'keep' if x['decision'] == 'accepted' else 'reject'}"))
    with tab_tbl:
        df = pd.DataFrame([{"#": x["idx"], "label": x["label"], "detector conf": round(x["conf"], 3),
                            "route": x["route"], "verifier P(yes)": None if x["route"] != "verified" else round(x["p_yes"], 3),
                            "decision": x["decision"], "reason": x["reason"]} for x in rows])
        st.dataframe(df, width="stretch", hide_index=True)
    with tab_log:
        st.code("\n".join(res["log"]), language=None)


# ------------------------------------------------------------------ page
# Everything below runs on every page interaction (Streamlit re-executes the script top to bottom).
st.title("Detect–Verify–Decide")
st.caption("Training-free multi-agent pipeline: a detector proposes, an independent verifier re-checks each "
           "uncertain candidate on a zoomed crop, a decision agent accepts or rejects.")
if USE_MOCK:
    st.warning("MOCK MODE: fake agents with random outputs, only for previewing this UI. "
               "Real detections need a GPU runtime (see notebooks/12_streamlit_demo.ipynb).")
ensure_models()

# ---- sidebar: input image ----
sb = st.sidebar
sb.header("Input")
uploaded = sb.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if sb.button("Use a COCO sample image", width="stretch"):
    st.session_state["sample_image"] = fetch_sample()
image = None
# An uploaded file wins over the sample image.
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif st.session_state.get("sample_image") is not None:
    image = st.session_state["sample_image"]
if image is not None and max(image.size) > MAX_SIDE:
    image = fit_max_side(image, MAX_SIDE)
    sb.caption(f"Image downscaled to {MAX_SIDE}px on the long side.")

# ---- sidebar: Decision Agent settings ----
sb.header("Decision Agent")
# Thresholds tuned on the held-out split (if notebook 05 / run_all_fast has been run), else sensible defaults.
sel = CFG.get("selected", {})
box_thr = float(CFG["detector"]["box_threshold"])
lo0 = min(max(float(sel.get("tau_low", 0.25)), box_thr), 1.0)
hi0 = min(max(float(sel.get("tau_high", 0.60)), lo0), 1.0)
tau_low, tau_high = sb.slider("Uncertain band (τ_low, τ_high)", box_thr, 1.0, (lo0, hi0), 0.05,
                              help="c >= τ_high is accepted; c < τ_low is rejected; in between the verifier decides.")
margin = sb.slider("Crop margin", 0.0, 0.5, float(CFG["verifier"]["crop_margin"]), 0.05,
                   help="How much context around each box the verifier sees.")
run = sb.button("Run pipeline", type="primary", key="run", disabled=image is None, width="stretch")
# Show which device the models run on.
try:
    import torch
    sb.caption("Device: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)"))
except ImportError:
    sb.caption("Device: none (mock mode)")

with st.expander("How it works"):
    st.markdown(
        "1. **Detector Agent** proposes candidate boxes with a confidence *c*.\n"
        "2. **Gate:** *c* ≥ τ_high is accepted, *c* < τ_low is rejected, the rest go to the verifier.\n"
        "3. **Crop and zoom:** each uncertain box is expanded by the margin and upscaled.\n"
        "4. **Verifier Agent** (frozen VLM) sees only the crop and a neutral question, never *c*.\n"
        "5. **Decision Agent:** accept = (c ≥ τ_high) or (verdict = yes and c ≥ τ_low).\n\n"
        "Timings exclude the time spent refreshing this page.")

# Placeholders that the pipeline callback keeps overwriting while it runs.
progress_ph, timeline_ph, log_ph = st.empty(), st.empty(), st.empty()
if run:
    try:
        # Store the finished run so the results survive later widget interactions.
        st.session_state["last"] = run_pipeline(image, tau_low, tau_high, margin, progress_ph, timeline_ph, log_ph)
        progress_ph.progress(100, text="100% · finished")
    except Exception:  # noqa: BLE001
        st.error("The pipeline failed.")
        st.code(traceback.format_exc())
        st.stop()
elif "last" in st.session_state:
    # No new run: redraw the previous run's final state.
    progress_ph.progress(100, text="100% · last run finished")
    render_timeline(timeline_ph, st.session_state["last"]["state"])
elif image is not None:
    st.image(image, caption="Input image. Press 'Run pipeline' in the sidebar.", width="stretch")
else:
    st.info("Upload an image or pick the COCO sample in the sidebar to begin.")

if "last" in st.session_state:
    render_results(st.session_state["last"])
