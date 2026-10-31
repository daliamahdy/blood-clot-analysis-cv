import argparse
import os
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# Blood clot segmentation and robot-head localization
#
# Blood clot: red-channel segmentation
# Head: dark-spot detection and centered box
#
# Every analysis step is saved as a numbered image in clotcv/results/visualizations/steps/
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(
    os.environ.get("IN", PROJECT_DIR / "input" / "images" / "Set1")
)
OUTPUT_DIR = Path(os.environ.get("OUT", Path(__file__).resolve().parent / "results"))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# vertical limits of the tube INTERIOR (fractions of image height)
TUBE_Y_TOP_RATIO = 0.26
TUBE_Y_BOTTOM_RATIO = 0.66

CLOT_ALPHA = 0.50
CLOT_COLOR = (140, 55, 15)       # BGR dark blue
HEAD_BOX_COLOR = (45, 125, 30)   # BGR dark green
HEAD_BOX_ALPHA = 0.50            # same opacity as the clot
HEAD_BOX_PAD = 4                 # small padding around the head core, in pixels
HEAD_CLOT_MARGIN_KERNEL = 61
HEAD_MIN_CANDIDATE_HEIGHT = 16
HEAD_MAX_CANDIDATE_ASPECT = 2.0


# ---------------- step recorder ----------------
class StepSaver:
    #Saves one numbered image after every analysis step.

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob("*.png"):      # start clean on every run
            old.unlink()
        self.count = 0

    def save(self, name, image):
        img = np.asarray(image)
        if img.dtype != np.uint8:                  # float images -> 0..255 for viewing
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.count += 1
        path = self.folder / f"{self.count:02d}_{name}.png"
        if not cv2.imwrite(str(path), img):
            raise OSError(f"Could not save pipeline step: {path}")
        return image


# ---------------- blood clot ----------------
def gamma_correct(image, gamma=2.0):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def threshold_clot(channel, threshold=40, blur=7, steps=None):
    smooth = cv2.GaussianBlur(channel, (blur, blur), 0)
    if steps:
        steps.save("clot_blurred", smooth)
    _, mask = cv2.threshold(smooth, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def close_and_fill(mask, ksize=25, steps=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if steps:
        steps.save("clot_closed", closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if steps:
            steps.save("clot_no_contours_found", closed)
        return closed

    largest = max(contours, key=cv2.contourArea)
    if steps:
        contour_image = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 1)
        steps.save("clot_all_contours", contour_image)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    if steps:
        largest_contour = np.zeros_like(mask)
        cv2.drawContours(largest_contour, [largest], -1, 255, thickness=1)
        steps.save("clot_largest_contour", largest_contour)
        steps.save("clot_filled", filled)
    return filled


def segment_clot(image, gamma=2.0, threshold=40, ksize=25, steps=None):
    red = image[:, :, 2]                      # OpenCV is BGR
    if steps:
        steps.save("clot_red_channel", red)

    red_gamma = gamma_correct(red, gamma)
    if steps:
        steps.save("clot_red_gamma", red_gamma)

    mask = threshold_clot(red_gamma, threshold, steps=steps)
    if steps:
        steps.save("clot_threshold", mask)

    clot_mask = close_and_fill(mask, ksize, steps=steps)
    if steps:
        steps.save("clot_mask_final", clot_mask)
    return clot_mask


def make_head_exclusion_mask(clot_mask):
    margin = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (HEAD_CLOT_MARGIN_KERNEL, HEAD_CLOT_MARGIN_KERNEL),
    )
    return cv2.dilate(clot_mask, margin)


# ---------------- helical robot head ----------------
def find_head(
    image,
    clot_mask,
    y_band,
    ksize=61,
    thresh=20,
    min_area=150,
    core_frac=0.5,
    steps=None,
):

    # Robot head = small dark blob (black-hat on red), away from the clot.
    # Returns a tight box (x, y, w, h) CENTERED on the head:
    #  - core_frac keeps only the darkest part of the blob (higher = tighter box)
    #  - the box is centered on the intensity-weighted center of that core

    red = cv2.GaussianBlur(image[:, :, 2], (9, 9), 0)
    if steps:
        steps.save("head_red_blurred", red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(red, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    if steps:
        steps.save("head_blackhat", blackhat)

    _, mask = cv2.threshold(blackhat.astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)
    if steps:
        steps.save("head_threshold", mask)

    # ignore the clot and a margin around it, and everything outside the tube
    expanded_clot = make_head_exclusion_mask(clot_mask)
    if steps:
        steps.save("head_exclusion_margin_around_clot", expanded_clot)
    mask[expanded_clot > 0] = 0
    if steps:
        steps.save("head_clot_removed", mask)
    mask[:y_band[0]] = 0
    mask[y_band[1]:] = 0
    if steps:
        tube_band = np.zeros_like(mask)
        tube_band[y_band[0]:y_band[1]] = 255
        steps.save("head_tube_search_band", tube_band)
        steps.save("head_inside_tube_band", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if steps:
        area_filtered = np.zeros_like(mask)
        cv2.drawContours(area_filtered, contours, -1, 255, cv2.FILLED)
        steps.save("head_area_filtered_candidates", area_filtered)
        steps.save("head_clot_and_outside_removed", mask)
    if not contours:
        empty_mask = np.zeros_like(mask)
        if steps:
            steps.save("head_no_candidate", empty_mask)
        return None, empty_mask, False

    plausible = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else float("inf")
        if h >= HEAD_MIN_CANDIDATE_HEIGHT and aspect <= HEAD_MAX_CANDIDATE_ASPECT:
            plausible.append(contour)
    if not plausible:
        empty_mask = np.zeros_like(mask)
        if steps:
            steps.save("head_no_plausible_candidate", empty_mask)
        return None, empty_mask, False
    blob = max(plausible, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(blob)
    head_mask = np.zeros_like(mask)
    cv2.drawContours(head_mask, [blob], -1, 255, cv2.FILLED)

    if steps:
        steps.save("head_blob", head_mask)

    # darkest core of the blob
    roi = blackhat[y:y + h, x:x + w]
    core = (roi > core_frac * roi.max()).astype(np.uint8)
    if steps:
        core_threshold = np.zeros_like(mask)
        core_threshold[y:y + h, x:x + w] = core * 255
        steps.save("head_core_threshold", core_threshold)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    if n < 2:
        box = (x, y, w, h)
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.where(labels == best)
        weights = roi[ys, xs]

        if steps:
            core_mask = np.zeros_like(mask)
            core_mask[y + ys, x + xs] = 255
            steps.save("head_core", core_mask)

        cx = (xs * weights).sum() / weights.sum() + x
        cy = (ys * weights).sum() / weights.sum() + y
        half_w = max(abs(xs.min() + x - cx), abs(xs.max() + x - cx))
        half_h = max(abs(ys.min() + y - cy), abs(ys.max() + y - cy))
        bw = int(round(2 * half_w)) + 1
        bh = int(round(2 * half_h)) + 1
        box = (int(round(cx - half_w)), int(round(cy - half_h)), bw, bh)

    if steps:
        steps.save("robot_head_mask", head_mask)
        center_image = image.copy()
        cv2.rectangle(
            center_image,
            (box[0], box[1]),
            (box[0] + box[2], box[1] + box[3]),
            (0, 255, 0),
            2,
        )
        steps.save("head_center_and_box", center_image)
    return box, head_mask, True


# ---------------- visualization ----------------
def make_overlay(image, mask, color=CLOT_COLOR, alpha=CLOT_ALPHA):
    overlay = image.copy()
    m = mask > 0
    blended = image.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha
    overlay[m] = blended.clip(0, 255).astype(np.uint8)[m]
    return overlay


def draw_head_box(image, box, color=HEAD_BOX_COLOR, alpha=HEAD_BOX_ALPHA, pad=HEAD_BOX_PAD):
    # Filled, translucent box around the robot head.
    if box is None:
        return image
    x, y, w, h = box
    H, W = image.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

    out = image.copy()
    region = image[y0:y1, x0:x1].astype(np.float32)
    mixed = region * (1 - alpha) + np.array(color, np.float32) * alpha
    out[y0:y1, x0:x1] = mixed.clip(0, 255).astype(np.uint8)
    return out


# ---------------- processing ----------------
def process_frame(
    image,
    steps,
    clot_mask=None,
    save_original=True,
):
    if save_original:
        steps.save("original", image)
    height = image.shape[0]
    y_band = (
        int(height * TUBE_Y_TOP_RATIO),
        int(height * TUBE_Y_BOTTOM_RATIO),
    )

    if clot_mask is None:
        clot_mask = segment_clot(image, steps=steps)
    clot_overlay = make_overlay(image, make_head_exclusion_mask(clot_mask))
    steps.save("clot_overlay", clot_overlay)

    tracked_head_box, head_mask, head_detected = find_head(
        image,
        clot_mask,
        y_band,
        steps=steps,
    )
    visible_head_box = tracked_head_box if head_detected else None
    overlay = draw_head_box(clot_overlay, visible_head_box)
    steps.save("final_overlay", overlay)
    return clot_mask, overlay, tracked_head_box, head_mask, head_detected


def process_image(input_file, show=True):
    image = cv2.imread(str(input_file))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_file}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_dir = OUTPUT_DIR / "visualizations" / "steps" / input_file.stem
    steps = StepSaver(steps_dir)
    clot_mask, overlay, head_box, head_mask, head_detected = process_frame(
        image, steps
    )

    mask_path = OUTPUT_DIR / f"{input_file.stem}_clot_mask.png"
    overlay_path = OUTPUT_DIR / f"{input_file.stem}_final_overlay.png"
    head_mask_path = OUTPUT_DIR / f"{input_file.stem}_robot_head_mask.png"
    if not cv2.imwrite(str(mask_path), clot_mask):
        raise OSError(f"Could not save clot mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not save result overlay: {overlay_path}")
    if not cv2.imwrite(str(head_mask_path), head_mask):
        raise OSError(f"Could not save robot-head mask: {head_mask_path}")
    print("clot area (px):", int((clot_mask > 0).sum()))
    print("head box (x, y, w, h):", head_box if head_detected else None)
    print("robot-head mask saved to:", head_mask_path)
    print("steps saved in:", steps_dir)

    if show and not os.environ.get("NOSHOW"):
        cv2.imshow("final overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Process every photo in a folder to segment the blood clot and locate the robot head."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=INPUT_PATH,
        help="Folder containing photos (default: %(default)s)",
    )
    input_path = parser.parse_args().input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input must be a folder of photos, not a file: {input_path}")

    image_files = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No supported photos found in: {input_path}")
    for image_file in image_files:
        print(f"Processing image: {image_file}")
        process_image(image_file, show=False)


if __name__ == "__main__":
    main()