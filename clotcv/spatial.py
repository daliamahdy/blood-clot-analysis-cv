import cv2
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("input/images/Set1")
OUTPUT = Path("diagnostics/spatial_test")

THRESHOLD = 100
MIN_SATURATION = 30

# --------------------------------
# LOAD IMAGE
# --------------------------------

if IMAGE_PATH.is_dir():
    files = sorted(
        p for p in IMAGE_PATH.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not files:
        raise FileNotFoundError("No images found")
    IMAGE_PATH = files[0]

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(IMAGE_PATH)

OUTPUT.mkdir(parents=True, exist_ok=True)

h, w = img.shape[:2]

# --------------------------------
# SELECT APPROXIMATE CLOT AREA
# --------------------------------

print("Select a rectangle containing the whole clot.")
print("Leave a small margin around it.")

x, y, rw, rh = map(
    int, cv2.selectROI("Select clot area", img, False)
)
cv2.destroyAllWindows()

if rw == 0 or rh == 0:
    raise ValueError("No area selected")

# --------------------------------
# SELECT APPROXIMATE CENTRE
# --------------------------------

print("Click approximately in the centre of the clot.")

points = []

def mouse(event, px, py, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((px, py))

cv2.namedWindow("Select clot centre")
cv2.setMouseCallback("Select clot centre", mouse)

while True:
    view = img.copy()
    cv2.rectangle(view, (x, y), (x+rw, y+rh), (0, 255, 0), 2)

    if points:
        cv2.circle(view, points[0], 5, (0, 0, 255), -1)

    cv2.imshow("Select clot centre", view)

    if cv2.waitKey(20) & 0xFF == 13 and points:
        break

cv2.destroyAllWindows()

cx, cy = points[0]

# --------------------------------
# COLOUR MASK
# --------------------------------

r = img[:, :, 2].astype(np.int16)
g = img[:, :, 1].astype(np.int16)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
sat = hsv[:, :, 1]

red_green = r - g

colour_mask = (
    (red_green > 10) &
    (red_green < THRESHOLD) &
    (sat > MIN_SATURATION)
)

# --------------------------------
# SOFT SPATIAL DISTANCE
# --------------------------------

Y, X = np.ogrid[:h, :w]

# Elliptical distance based on selected rectangle
rx = max(rw / 2, 1)
ry = max(rh / 2, 1)

distance = np.sqrt(
    ((X - cx) / rx) ** 2 +
    ((Y - cy) / ry) ** 2
)

# Four candidates
masks = {}

# 1. Colour only, restricted to rectangle
m1 = np.zeros((h, w), dtype=np.uint8)
m1[y:y+rh, x:x+rw] = colour_mask[y:y+rh, x:x+rw] * 255
masks["01_colour_roi"] = m1

# 2. Colour + broad elliptical spatial constraint
m2 = (colour_mask & (distance < 1.15)).astype(np.uint8) * 255
masks["02_soft_ellipse"] = m2

# 3. Colour + tighter elliptical constraint
m3 = (colour_mask & (distance < 0.95)).astype(np.uint8) * 255
masks["03_tighter_ellipse"] = m3

# 4. Colour + smooth Gaussian spatial weighting
weight = np.exp(-2.0 * distance ** 2)
m4 = (colour_mask & (weight > 0.12)).astype(np.uint8) * 255
masks["04_gaussian"] = m4

# --------------------------------
# VISUALISE AND SAVE
# --------------------------------

for name, mask in masks.items():

    # Keep largest connected component
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 8
    )

    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = (labels == largest).astype(np.uint8) * 255
    else:
        clean = mask.copy()

    overlay = img.copy()
    tint = np.zeros_like(img)
    tint[:] = (140, 55, 15)

    selected = clean > 0
    overlay[selected] = cv2.addWeighted(
        img[selected], 0.45,
        tint[selected], 0.55,
        0
    )

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    cv2.imwrite(str(OUTPUT / f"{name}.png"), overlay)
    cv2.imwrite(str(OUTPUT / f"{name}_mask.png"), clean)

    ys, xs = np.where(clean > 0)

    if len(xs):
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    else:
        bbox = None

    print(
        f"{name}: area={np.count_nonzero(clean)}, "
        f"bounding_box={bbox}"
    )

print("\nResults saved in:", OUTPUT.resolve())