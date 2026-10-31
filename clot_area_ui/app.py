from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="ClotScope | Area Tracking", page_icon="🩸", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#102b46,#174e68); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .25rem 0; color: white;}
.hero p {margin: 0; opacity: .85;}
.metric-card {border: 1px solid rgba(120,130,140,.25); border-radius: 12px; padding: 1rem;}
.small-note {font-size: .85rem; color: #777;}
</style>
<div class="hero"><h1>ClotScope</h1><p>Interactive clot segmentation · normalized area over time</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Analysis setup")
    st.caption("SAM 2 checkpoint is required for segmentation and tracking.")
    checkpoint = st.text_input("SAM 2 checkpoint path", value="checkpoints/sam2.1_hiera_large.pt")
    config = st.text_input("SAM 2 model config", value="configs/sam2.1/sam2.1_hiera_l.yaml")
    device = st.selectbox("Device", ["cuda", "cpu"], index=0)
    sample_step = st.slider("Analyze every Nth frame", min_value=1, max_value=15, value=3)
    max_frames = st.slider("Maximum analyzed frames", min_value=30, max_value=600, value=240, step=30)

uploaded = st.file_uploader("Upload a catheter video", type=["mp4", "mov", "avi", "mkv"])
if not uploaded:
    st.info("Upload a video to propose clot masks from its first frame.")
    st.stop()

@st.cache_data(show_spinner=False)
def decode_video(video_bytes: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    Path(tmp.name).unlink(missing_ok=True)
    return frames, float(fps)

with st.spinner("Reading video…"):
    frames, fps = decode_video(uploaded.getvalue())
if not frames:
    st.error("Could not read this video. Try exporting it as MP4 (H.264).")
    st.stop()

st.caption(f"{len(frames):,} frames · {fps:.2f} fps · {frames[0].shape[1]} × {frames[0].shape[0]} px")

@st.cache_resource(show_spinner=False)
def load_sam2(checkpoint_path: str, config_name: str, device_name: str):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    import torch
    model = build_sam2(config_name, checkpoint_path, device=device_name)
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=24, pred_iou_thresh=0.70,
        stability_score_thresh=0.85, min_mask_region_area=100
    )
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large") if False else None
    return generator

if "candidate_masks" not in st.session_state or st.session_state.get("video_name") != uploaded.name:
    st.session_state.candidate_masks = None
    st.session_state.video_name = uploaded.name
    st.session_state.approved_mask = None

left, right = st.columns([1.05, 1])
with left:
    st.subheader("01 · Propose a mask")
    st.image(frames[0], caption="First frame", use_container_width=True)
    if st.button("Generate SAM 2 mask proposals", type="primary", use_container_width=True):
        if not Path(checkpoint).exists():
            st.error(f"Checkpoint not found: {checkpoint}. Download the SAM 2.1 checkpoint and enter its path in the sidebar.")
        else:
            try:
                with st.spinner("Generating candidate masks with SAM 2…"):
                    generator = load_sam2(checkpoint, config, device)
                    masks = generator.generate(frames[0])
                    masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:60]
                    st.session_state.candidate_masks = masks
            except Exception as e:
                st.error(f"SAM 2 could not start: {e}")
                st.caption("Check the installation, checkpoint and config path in README.md.")

with right:
    st.subheader("02 · Review and approve")
    masks = st.session_state.candidate_masks
    if masks:
        labels = [f"Candidate {i+1} · {m['area']:,} px · quality {m.get('predicted_iou', 0):.2f}" for i, m in enumerate(masks)]
        idx = st.selectbox("Choose the clot candidate", range(len(masks)), format_func=lambda i: labels[i])
        mask = masks[idx]["segmentation"].astype(bool)
        preview = frames[0].copy()
        tint = np.zeros_like(preview); tint[:] = (255, 75, 80)
        preview[mask] = (0.55 * preview[mask] + 0.45 * tint[mask]).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 255, 255), 2)
        st.image(preview, caption=f"Candidate mask · {int(mask.sum()):,} pixels", use_container_width=True)
        if st.button("Approve this mask and track through video", type="primary", use_container_width=True):
            st.session_state.approved_mask = mask
    elif st.session_state.approved_mask is None:
        st.markdown("<div class='metric-card'>Run mask proposals, then select the candidate that best follows the clot. The overlay is shown in red.</div>", unsafe_allow_html=True)
    if st.session_state.approved_mask is not None:
        st.success("Initial mask approved.")

if st.session_state.approved_mask is None:
    st.stop()

st.divider()
st.subheader("03 · Track and quantify")
if st.button("Run video analysis", type="primary"):
    if not Path(checkpoint).exists():
        st.error("Checkpoint path is invalid.")
        st.stop()
    try:
        from sam2.build_sam import build_sam2_video_predictor
        import torch
        with st.spinner("Tracking the approved mask through the video…"):
            workdir = Path(tempfile.mkdtemp(prefix="clotscope_"))
            frame_dir = workdir / "frames"; frame_dir.mkdir()
            selected_indices = list(range(0, len(frames), sample_step))[:max_frames]
            if selected_indices[-1] != len(frames)-1 and len(selected_indices) < max_frames:
                selected_indices.append(len(frames)-1)
            # SAM2 video predictor expects sequential JPEG frames. Keep all frames for stable tracking.
            for i, frame in enumerate(frames):
                cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            predictor = build_sam2_video_predictor(config, checkpoint, device=device)
            state = predictor.init_state(video_path=str(frame_dir))
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=st.session_state.approved_mask)
            rows, overlay_frames = [], []
            sampled_set = set(selected_indices)
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
                if frame_idx not in sampled_set: continue
                mask = (logits[0] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                area = int(mask.sum())
                rows.append({"frame": int(frame_idx), "time_s": frame_idx / fps, "area_px": area})
                frame = frames[frame_idx].copy()
                tint = np.zeros_like(frame); tint[:] = (255, 65, 80)
                frame[mask] = (0.55 * frame[mask] + 0.45 * tint[mask]).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (255,255,255), 2)
                overlay_frames.append((frame_idx, frame))
            if not rows:
                raise RuntimeError("SAM 2 did not return any masks.")
            df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
            baseline = float(df.loc[0, "area_px"])
            if baseline <= 0: raise RuntimeError("The approved first-frame mask has zero area.")
            df["normalized_area"] = df["area_px"] / baseline
            st.session_state.results_df = df
            st.session_state.overlay_frames = overlay_frames
            st.session_state.result_frames = frames
            st.session_state.analysis_done = True
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.caption("See README.md for SAM 2 installation and model setup.")

if st.session_state.get("analysis_done"):
    df = st.session_state.results_df
    overlays = st.session_state.overlay_frames
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline area", f"{df.area_px.iloc[0]:,.0f} px")
    c2.metric("Latest / initial area", f"{df.normalized_area.iloc[-1]:.3f}×")
    c3.metric("Frames measured", f"{len(df):,}")
    chart_col, frame_col = st.columns([1.2, 1])
    with chart_col:
        st.markdown("#### Normalized clot area")
        st.caption("A(t) / A(0) · dimensionless · first analyzed frame = 1.0")
        st.line_chart(df.set_index("time_s")["normalized_area"], height=340)
    with frame_col:
        st.markdown("#### Segmentation preview")
        if overlays:
            pos = st.slider("Analyzed frame", 0, len(overlays)-1, len(overlays)-1)
            fidx, img = overlays[pos]
            st.image(img, caption=f"Frame {fidx} · {fidx/fps:.2f} s", use_container_width=True)
    st.download_button("Download measurements (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="clot_area_over_time.csv", mime="text/csv")
    st.caption("Area is measured in image pixels. Normalization enables relative comparison; it does not provide physical area or clot volume.")
