import cv2
import numpy as np
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

IMAGE_PATH = Path("input/images/Set1")
OUTPUT_DIR = Path("diagnostics/candidates")

THRESHOLDS = [70, 85, 100, 115]

# Minimum saturation to reject the pale background
MIN_SATURATION = 30

# Small cleanup only; preserve irregular boundaries
KERNEL_SIZE = 3

# ==========================================
# LOAD IMAGE
# ==========================================

if IMAGE_PATH.is_dir():
    files = sorted(
        p for p in IMAGE_PATH.iterdir()
        if p.suffix.lower() in
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not files:
        raise FileNotFoundError("No images found.")
    IMAGE_PATH = files[0]

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(IMAGE_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

h, w = img.shape[:2]

# ==========================================
# SELECT SEARCH AREA
# ==========================================

print("\nSelect a rectangle containing the ENTIRE clot")
print("Include a little surrounding blood stream.")
print("This is only a search area, NOT the segmentation shape.")

x, y, rw, rh = map(
    int,
    cv2.selectROI("Select clot search area", img, False)
)

cv2.destroyAllWindows()

if rw == 0 or rh == 0:
    raise ValueError("No ROI selected.")

roi = img[y:y+rh, x:x+rw]

# ==========================================
# COLOUR FEATURES
# ==========================================

b = roi[:, :, 0].astype(np.int16)
g = roi[:, :, 1].astype(np.int16)
r = roi[:, :, 2].astype(np.int16)

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
saturation = hsv[:, :, 1]

# Difference between red and green channels
red_green = r - g

# ==========================================
# CREATE CANDIDATES
# ==========================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (KERNEL_SIZE, KERNEL_SIZE)
)

for threshold in THRESHOLDS:

    # Clot should have:
    # 1. Red-green difference below threshold
    # 2. Enough saturation to reject pale background
    #
    # No rectangular or circular shape assumption.

    mask = (
        (red_green > 10) &
        (red_green < threshold) &
        (saturation > MIN_SATURATION)
    ).astype(np.uint8) * 255

    # Very mild cleanup
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Put ROI mask back into full image coordinates
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y:y+rh, x:x+rw] = mask

    # Overlay
    overlay = img.copy()

    colour = np.zeros_like(img)
    colour[:, :] = (140, 55, 15)

    selected = full_mask > 0

    overlay[selected] = cv2.addWeighted(
        img[selected],
        0.45,
        colour[selected],
        0.55,
        0
    )

    # Draw irregular contours
    contours, _ = cv2.findContours(
        full_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        2
    )

    # Save results
    cv2.imwrite(
        str(OUTPUT_DIR / f"mask_{threshold}.png"),
        full_mask
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"overlay_{threshold}.png"),
        overlay
    )

    percentage = np.count_nonzero(mask) / mask.size * 100

    print(
        f"Threshold {threshold}: "
        f"{len(contours)} contours, "
        f"{np.count_nonzero(mask)} pixels, "
        f"{percentage:.2f}% of ROI"
    )

# ==========================================
# SAVE SEARCH AREA
# ==========================================

marked = img.copy()

cv2.rectangle(
    marked,
    (x, y),
    (x + rw, y + rh),
    (0, 255, 0),
    2
)

cv2.imwrite(
    str(OUTPUT_DIR / "search_area.png"),
    marked
)

print("\nFinished!")
print("Results:", OUTPUT_DIR.resolve())