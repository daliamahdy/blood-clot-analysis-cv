"""Workflow: first frame -> proposed mask -> human review -> propagate -> measure -> export."""
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .io_utils import video_info, read_first_frame, iter_frames, extract_frames
from .measurements import measure, area_px_to_mm2
from .review_ui import run_review, ReviewSession
from .visualization import overlay_mask, save_sample, open_video_writer, save_area_plot


def make_backend(name, model_size, checkpoint, device):
    if name == "sam2":
        from .sam2_backend import SAM2Backend
        return SAM2Backend(model_size, checkpoint, device)
    from .grabcut_backend import GrabCutBackend
    return GrabCutBackend()


def approve_first_frame(frame, backend, out_dir, points, labels, box, auto_approve):
    """Steps 2-7: propose, display, approve/adjust. Returns the approved mask."""
    if auto_approve:
        backend.set_image(frame)
        cands, scores = backend.propose(points, labels, box)
        if not cands:
            raise SystemExit("--auto-approve: the backend returned no mask for these prompts.")

        mask, prompts = cands[0], {"points": [list(p) for p in points], "labels": labels,
                                   "box_xywh": list(box) if box else None,
                                   "candidate_index": 0, "score": scores[0], "auto_approved": True}
    else:
        mask, prompts = run_review(frame, backend, points, labels, box)
        if mask is None:
            raise SystemExit("Not approved - nothing saved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "first_frame_mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "first_frame_overlay.jpg"), overlay_mask(frame, mask))
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    print(f"Approved mask saved to {out_dir}")
    return mask


def run(video, output="results", backend_name="sam2", model_size="large", checkpoint=None,
        device=None, every=1, vis_every=30, mask_file=None, points=(), labels=(), box=None,
        auto_approve=False, pixels_per_mm=None):
    video, output = Path(video), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    fps = info["fps"]
    frame0 = read_first_frame(video)
    backend = make_backend(backend_name, model_size, checkpoint, device)

    # Steps 1-7: first frame, proposal, approval (or reuse a saved approved mask)
    if mask_file:
        mask0 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 127
        if mask0.shape != frame0.shape[:2]:
            raise SystemExit("Saved mask size does not match the video frame size.")
        backend.set_image(frame0)
        print(f"Using previously approved mask: {mask_file}")
    else:
        mask0 = approve_first_frame(frame0, backend, output / "approved",
                                    list(points), list(labels), box, auto_approve)

    # Step 8: apply to all (sampled) frames
    rows, vis_dir = [], output / "visualizations"
    writer = open_video_writer(output / "tracked_overlay.mp4", fps / every,
                               (info["width"], info["height"]))
    first_area = None
    with tempfile.TemporaryDirectory(prefix="clot_frames_") as tmp:
        print("Extracting frames ...")
        originals = extract_frames(video, tmp, every)
        frames = iter_frames(video, every)
        print(f"Propagating through {len(originals)} frames ...")
        for pos, mask in backend.propagate(tmp, mask0):
            idx, image = next(frames)
            assert idx == originals[pos], "frame order mismatch"
            # Step 9: measurements
            m = measure(image, mask)
            if first_area is None:
                first_area = m["area_px"] or float("nan")
            row = {"sample_number": pos + 1, "source": video.name, "frame_index": idx,
                   "time_seconds": idx / fps if fps == fps and fps > 0 else float("nan"),
                   "backend": backend.name, **m,
                   "area_rel_to_first": m["area_px"] / first_area,
                   "mask_empty": not bool(mask.any())}
            if pixels_per_mm:
                row["area_mm2"] = area_px_to_mm2(m["area_px"], pixels_per_mm)
            rows.append(row)
            overlay = overlay_mask(image, mask)
            writer.write(overlay)
            if vis_every and pos % max(1, vis_every // every) == 0:
                save_sample(vis_dir, f"frame{idx:06d}", image, mask)
            if (pos + 1) % 50 == 0:
                print(f"  {pos + 1}/{len(originals)}")
    writer.release()

    # Step 10: export
    df = pd.DataFrame(rows)
    df.to_csv(output / "measurements.csv", index=False)
    save_area_plot(rows, output / "area_over_time.png", use_mm2=bool(pixels_per_mm))
    print(f"Done. Measurements: {output / 'measurements.csv'}")
    print(f"Plot: {output / 'area_over_time.png'} | video: {output / 'tracked_overlay.mp4'}")
    if df["mask_empty"].any():
        print(f"Note: {int(df['mask_empty'].sum())} frame(s) have an empty mask - check the video.")
    return df
