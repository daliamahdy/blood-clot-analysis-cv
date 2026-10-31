"""Overlays, overlay video and the area-over-time plot."""
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(image, mask, alpha=0.4, colour=(0, 0, 255), contour=(0, 255, 0)):
    """Tint the mask and draw its outline on a copy of a BGR image."""
    out = image.copy()
    if mask is None or not mask.any():
        return out
    tint = np.empty_like(out)
    tint[:] = colour
    out[mask] = cv2.addWeighted(out[mask], 1 - alpha, tint[mask], alpha, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, contour, 2)
    return out


def save_sample(output_dir, name, image, mask):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{name}_mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / f"{name}_overlay.jpg"), overlay_mask(image, mask))


def open_video_writer(path, fps, size):
    fps = fps if fps and fps == fps and fps > 0 else 30.0
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def save_area_plot(rows, output_file, use_mm2=False):
    if not rows:
        return
    times = [r["time_seconds"] for r in rows]
    valid = all(t == t for t in times)
    x = times if valid else [r["frame_index"] for r in rows]
    key = "area_mm2" if use_mm2 else "area_px"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, [r[key] for r in rows], linewidth=1.4)
    ax.set_xlabel("Time (seconds)" if valid else "Frame index")
    ax.set_ylabel("Clot area (mm²)" if use_mm2 else "Clot area (pixel²)")
    ax.set_title("Tracked clot area")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)
