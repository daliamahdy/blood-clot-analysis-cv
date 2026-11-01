import argparse
import cv2
import numpy as np
from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("input/images/Set1")
OUTPUT_DIR = Path("results")

# Clot segmentation
BLUR_SIZE = 5
THRESHOLD = 100
MORPH_SIZE = 5

RECOVERY_THRESHOLD = 125
RECOVERY_RADIUS = 12

# Temporal stabilisation
TEMPORAL_RADIUS = 3
TEMPORAL_KEEP_THRESHOLD = 145

# Clot overlay (BGR)
OVERLAY_ALPHA = 0.50
OVERLAY_COLOR = (140, 55, 15)

# Robot detection: rightmost third only
ROBOT_DARK_THRESHOLD = 65
ROBOT_MIN_AREA = 10
ROBOT_MIN_CIRCULARITY = 0.35
ROBOT_MIN_ASPECT_RATIO = 0.65

ROBOT_BOX_SCALE = 2.5
ROBOT_BOX_MIN_SIZE = 30
ROBOT_BOX_MAX_SIZE = 100

ROBOT_OVERLAY_ALPHA = 0.35
ROBOT_COLOR = (0, 255, 0)


# ============================================================
# GENERAL HELPERS
# ============================================================

def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def largest_component(mask):
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if count <= 1:
        return np.zeros_like(binary)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    result = np.zeros_like(binary)
    result[labels == largest_label] = 255

    return result


def save_debug(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


# ============================================================
# CLOT SEGMENTATION
# ============================================================

def segment_clot(image, previous_mask=None):

    red = image[:, :, 2]

    blurred = cv2.GaussianBlur(
        red,
        (BLUR_SIZE, BLUR_SIZE),
        0
    )

    # Initial dark clot candidates
    _, core = cv2.threshold(
        blurred,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_SIZE, MORPH_SIZE)
    )

    core = cv2.morphologyEx(
        core,
        cv2.MORPH_OPEN,
        kernel
    )

    core = cv2.morphologyEx(
        core,
        cv2.MORPH_CLOSE,
        kernel
    )

    core = largest_component(core)

    # Relaxed candidates
    _, relaxed_candidates = cv2.threshold(
        blurred,
        RECOVERY_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    recovery_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * RECOVERY_RADIUS + 1, 2 * RECOVERY_RADIUS + 1)
    )

    expanded_core = cv2.dilate(
        core,
        recovery_kernel,
        iterations=1
    )

    relaxed_candidates = cv2.bitwise_and(
        relaxed_candidates,
        expanded_core
    )

    # Recover clot pixels on the left side
    ys, xs = np.where(core > 0)

    if len(xs) > 0:

        center_x = int(np.mean(xs))

        left_region = np.zeros_like(core)
        left_region[:, :center_x + 1] = 255

        recovery = cv2.bitwise_and(
            relaxed_candidates,
            left_region
        )

        clot_mask = cv2.bitwise_or(
            core,
            recovery
        )

    else:
        clot_mask = core.copy()

    clot_mask = cv2.morphologyEx(
        clot_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    clot_mask = largest_component(clot_mask)

    # Temporal stabilisation
    if previous_mask is not None:

        temporal_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * TEMPORAL_RADIUS + 1,
                2 * TEMPORAL_RADIUS + 1
            )
        )

        previous_nearby = cv2.dilate(
            previous_mask,
            temporal_kernel,
            iterations=1
        )

        plausible_previous = np.zeros_like(previous_mask)

        plausible_previous[
            blurred < TEMPORAL_KEEP_THRESHOLD
        ] = 255

        retained = cv2.bitwise_and(
            previous_nearby,
            plausible_previous
        )

        clot_mask = cv2.bitwise_or(
            clot_mask,
            retained
        )

        clot_mask = largest_component(clot_mask)

    return clot_mask, core, relaxed_candidates


# ============================================================
# ROBOT HEAD DETECTION
# ============================================================

def detect_robot_head(image, steps_dir):

    height, width = image.shape[:2]

    # Search ONLY the rightmost third
    start_x = (width * 2) // 3

    roi = image[:, start_x:]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detect dark regions
    _, dark_mask = cv2.threshold(
        blurred,
        ROBOT_DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    diagnostic = cv2.cvtColor(
        dark_mask,
        cv2.COLOR_GRAY2BGR
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < ROBOT_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4.0 * np.pi * area /
            (perimeter * perimeter)
        )

        aspect_ratio = min(w, h) / max(w, h)

        if circularity < ROBOT_MIN_CIRCULARITY:
            continue

        if aspect_ratio < ROBOT_MIN_ASPECT_RATIO:
            continue

        global_x = x + start_x

        candidates.append({
            "x": global_x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": global_x + w // 2,
            "cy": y + h // 2,
            "circularity": circularity
        })

    # Diagnostic image: all valid candidates
    for candidate in candidates:

        x = candidate["x"] - start_x
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            diagnostic,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    save_debug(
        steps_dir / "11_robot_candidates.png",
        diagnostic
    )

    if not candidates:
        print("Robot head not detected in the rightmost third.")
        return None

    # Select the largest circular dark object
    selected = max(
        candidates,
        key=lambda c: c["area"] * c["circularity"]
    )

    cx = selected["cx"]
    cy = selected["cy"]

    # Create square around the detected circle
    box_size = int(
        max(selected["w"], selected["h"]) *
        ROBOT_BOX_SCALE
    )

    box_size = max(
        ROBOT_BOX_MIN_SIZE,
        min(box_size, ROBOT_BOX_MAX_SIZE)
    )

    half = box_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)

    x2 = min(width - 1, cx + half)
    y2 = min(height - 1, cy + half)

    print(
        f"Robot head detected: centre=({cx}, {cy}), "
        f"area={selected['area']:.0f}, "
        f"box=({x1}, {y1}, {x2}, {y2})"
    )

    return (x1, y1, x2, y2)


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(image, clot_mask, robot_box=None):

    overlay = image.copy()

    # Clot overlay
    clot_layer = overlay.copy()

    clot_layer[clot_mask > 0] = OVERLAY_COLOR

    overlay = cv2.addWeighted(
        clot_layer,
        OVERLAY_ALPHA,
        overlay,
        1.0 - OVERLAY_ALPHA,
        0
    )

    # No yellow clot contour.

    # Robot green square
    if robot_box is not None:

        x1, y1, x2, y2 = robot_box

        green_layer = overlay.copy()

        cv2.rectangle(
            green_layer,
            (x1, y1),
            (x2, y2),
            ROBOT_COLOR,
            -1
        )

        overlay = cv2.addWeighted(
            green_layer,
            ROBOT_OVERLAY_ALPHA,
            overlay,
            1.0 - ROBOT_OVERLAY_ALPHA,
            0
        )

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            ROBOT_COLOR,
            3
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.circle(
            overlay,
            (cx, cy),
            3,
            (255, 255, 255),
            -1
        )

    return overlay


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, output_dir, previous_mask=None):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return previous_mask

    frame_name = image_path.stem

    steps_dir = output_dir / frame_name
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Clot segmentation
    clot_mask, core, relaxed_candidates = segment_clot(
        image,
        previous_mask
    )

    # Save intermediate images
    save_debug(
        steps_dir / "01_red_channel.png",
        image[:, :, 2]
    )

    save_debug(
        steps_dir / "02_clot_core.png",
        core
    )

    save_debug(
        steps_dir / "05_relaxed_candidates.png",
        relaxed_candidates
    )

    save_debug(
        steps_dir / "06_final_clot_mask.png",
        clot_mask
    )

    # Robot detection from original image, rightmost third only
    robot_box = detect_robot_head(
        image,
        steps_dir
    )

    # Final overlay
    overlay = create_overlay(
        image,
        clot_mask,
        robot_box
    )

    save_debug(
        output_dir / f"{frame_name}_mask.png",
        clot_mask
    )

    save_debug(
        output_dir / f"{frame_name}_overlay.png",
        overlay
    )

    print(f"Processed: {image_path.name}")

    return clot_mask


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Blood clot segmentation and robot head detection"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DIR)
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR)
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    extensions = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    image_paths = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    image_paths.sort(key=natural_sort_key)

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    previous_mask = None

    for image_path in image_paths:

        previous_mask = process_image(
            image_path,
            output_dir,
            previous_mask
        )

    print("\nProcessing completed.")
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()