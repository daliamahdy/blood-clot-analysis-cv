"""Connect the smaller functions into one analysis pipeline."""
from pathlib import Path
import pandas as pd
from .io_utils import load_samples
from .roi import choose_roi, crop_roi
from .segmentation import segment, clean_mask
from .measurements import measure
from .visualization import save_sample, save_area_plot


def run(input_path, output_dir="results", roi_mode="manual", method="otsu",
        polarity="dark", every=1, max_samples=None, min_object_area=100,
        adaptive_block=51, adaptive_offset=5, fill_holes=False):
    output = Path(output_dir)
    visual_dir = output / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    rows, roi = [], None

    for sample_no, (source, frame_index, time_s, image) in enumerate(
        load_samples(input_path, every, max_samples), start=1
    ):
        if roi is None:
            roi = choose_roi(image, roi_mode)
            print("ROI:", roi)
        cropped = crop_roi(image, roi)
        mask, threshold = segment(cropped, method, polarity, adaptive_block, adaptive_offset)
        mask = clean_mask(mask, min_object_area, fill_holes=fill_holes)
        values = measure(cropped, mask)
        row = {
            "sample_number": sample_no, "source": source,
            "frame_index": frame_index, "time_seconds": time_s,
            "roi_x": roi[0], "roi_y": roi[1], "roi_width": roi[2], "roi_height": roi[3],
            "method": method, "polarity": polarity, "threshold": threshold,
            **values,
        }
        rows.append(row)
        stem = f"{sample_no:05d}_frame{frame_index:06d}"
        save_sample(visual_dir, stem, image, roi, cropped, mask)
        if sample_no % 25 == 0:
            print(f"Processed {sample_no} samples")

    if not rows:
        raise RuntimeError("No images or video frames were processed.")
    df = pd.DataFrame(rows)
    df.to_csv(output / "measurements.csv", index=False)
    save_area_plot(rows, output / "area_over_time.png")
    print(f"Saved {len(rows)} measurements to {output / 'measurements.csv'}")
    print(f"Inspect visualizations in {visual_dir}")
    return df
