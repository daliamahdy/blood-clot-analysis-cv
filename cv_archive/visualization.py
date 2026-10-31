
# Save full-frame clot overlays, binary masks, and area plots.

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_sample(output_dir, name, full_image, roi, roi_image, mask):
    # Save the original full frame with the segmented clot highlighted.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_mask = np.asarray(mask) > 0

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {binary_mask.shape}.")

    if roi_image.shape[:2] != binary_mask.shape:
        raise ValueError(
            f"Mask shape {binary_mask.shape} does not match "
            f"ROI image shape {roi_image.shape[:2]}."
        )

    x, y, width, height = map(int, roi)
    full_h, full_w = full_image.shape[:2]
    roi_h, roi_w = binary_mask.shape

    if (
        x < 0 or y < 0
        or x + roi_w > full_w
        or y + roi_h > full_h
    ):
        raise ValueError("ROI mask does not fit inside the full image.")

    # Save the original frame with the ROI rectangle for reference.
    preview = full_image.copy()
    cv2.rectangle(
        preview,
        (x, y),
        (x + width, y + height),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / f"{name}_roi.jpg"), preview)

    # Place the original segmentation mask into full-frame coordinates.
    # Do not smooth, crop, or reconstruct its contours here.
    full_mask = np.zeros((full_h, full_w), dtype=bool)
    full_mask[y:y + roi_h, x:x + roi_w] = binary_mask

    # Save the exact binary mask for inspecting segmentation errors.
    cv2.imwrite(
        str(output_dir / f"{name}_mask.png"),
        full_mask.astype(np.uint8) * 255,
    )

    # Blend the segmented pixels with dark blue at 50% opacity.
    overlay = full_image.copy()
    dark_blue = np.array([140, 55, 15], dtype=np.float32)
    alpha = 0.50

    blended = (
        full_image.astype(np.float32) * (1.0 - alpha)
        + dark_blue * alpha
    ).clip(0, 255).astype(np.uint8)

    overlay[full_mask] = blended[full_mask]

    # Save the overlay without drawing artificial contour lines.
    overlay_path = output_dir / f"{name}_overlay.png"

    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Failed to save overlay: {overlay_path}")

    cv2.imwrite(
        str(output_dir / f"{name}_overlay.jpg"),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


def save_area_plot(rows, output_file):
    # Save the measured clot area curve.
    if not rows:
        return

    times = [row["time_seconds"] for row in rows]
    valid_times = all(
        t is not None and np.isfinite(t) for t in times
    )

    x_values = (
        times if valid_times
        else [row["sample_number"] for row in rows]
    )
    y_values = [row["area_px"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_values, y_values, linewidth=1.8, color="#e63946")
    ax.set_xlabel(
        "Time (seconds)" if valid_times else "Sample number"
    )
    ax.set_ylabel("Contour area (pixel²)")
    ax.set_title("Segmented clot area")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
