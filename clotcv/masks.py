import cv2
import numpy as np
from pathlib import Path

BASE = Path("diagnostics/spatial_test")
IMAGE_DIR = Path("input/images/Set1")
OUT = BASE / "comparison"

OUT.mkdir(exist_ok=True)

# Load original image
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

img = cv2.imread(str(images[0]))

# Load masks
m3 = cv2.imread(str(BASE / "03_tighter_ellipse_mask.png"), 0)
m4 = cv2.imread(str(BASE / "04_gaussian_mask.png"), 0)

if img is None or m3 is None or m4 is None:
    raise FileNotFoundError("Could not load original image or masks.")

# Binary masks
m3 = m3 > 0
m4 = m4 > 0

# Pixels detected by 04 but not 03
extra = m4 & ~m3

ys, xs = np.where(m3)

if len(xs) == 0:
    raise ValueError("Candidate 03 is empty.")

cx = (xs.min() + xs.max()) / 2
cy = (ys.min() + ys.max()) / 2

# Divide image into four regions around candidate 03
regions = {
    "LEFT": extra & (np.indices(extra.shape)[1] < cx),
    "RIGHT": extra & (np.indices(extra.shape)[1] >= cx),
    "TOP": extra & (np.indices(extra.shape)[0] < cy),
    "BOTTOM": extra & (np.indices(extra.shape)[0] >= cy),
}

print("\n========== MASK COMPARISON ==========")
print("Candidate 03:", np.count_nonzero(m3))
print("Candidate 04:", np.count_nonzero(m4))
print("Additional pixels in 04:", np.count_nonzero(extra))

for name, region in regions.items():
    print(f"{name}: {np.count_nonzero(region)} additional pixels")

# Create difference image
# Green = common pixels
# Red = extra pixels in candidate 04
# Blue = pixels only in candidate 03

visual = np.zeros_like(img)
visual[m3 & m4] = (0, 180, 0)
visual[extra] = (0, 0, 255)
visual[m3 & ~m4] = (255, 0, 0)

cv2.imwrite(str(OUT / "difference.png"), visual)

# Overlay candidate 04 extra pixels on original
overlay = img.copy()
overlay[extra] = (
    0.4 * overlay[extra] + 0.6 * np.array([0, 0, 255])
).astype(np.uint8)

cv2.imwrite(str(OUT / "extra_pixels.png"), overlay)

print("\nSaved:")
print(OUT / "difference.png")
print(OUT / "extra_pixels.png")