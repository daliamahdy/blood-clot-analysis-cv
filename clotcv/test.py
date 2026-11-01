import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_PATH = Path("input/images/Set1")  # Change to your image file
OUTPUT_DIR = Path("diagnostics")

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

if IMAGE_PATH.is_dir():
    images = sorted(
        p for p in IMAGE_PATH.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    )
    if not images:
        raise FileNotFoundError("No images found in input folder.")
    IMAGE_PATH = images[0]

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(exist_ok=True)

h, w = img.shape[:2]

print(f"\nImage: {IMAGE_PATH}")
print(f"Dimensions: {w} x {h}")

# --------------------------------------------------
# SELECT REGIONS
# --------------------------------------------------

display = img.copy()

print("\nSelect THREE regions:")
print("1. Inside the clot")
print("2. Blood stream / surrounding area")
print("3. Background outside the blood stream")
print("\nDraw a rectangle around each region and press ENTER.")
print("Press ESC to cancel.")

regions = {}

for name in ["CLOT", "STREAM", "BACKGROUND"]:
    roi = cv2.selectROI(
        f"Select {name}",
        display,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow(f"Select {name}")

    x, y, rw, rh = map(int, roi)

    if rw == 0 or rh == 0:
        raise ValueError(f"No region selected for {name}")

    regions[name] = img[y:y+rh, x:x+rw]

    cv2.rectangle(
        display,
        (x, y),
        (x+rw, y+rh),
        (0, 255, 255),
        2
    )

# --------------------------------------------------
# ANALYSE PIXEL DISTRIBUTIONS
# --------------------------------------------------

def analyse(name, region):

    print(f"\n========== {name} ==========")

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)

    channels = {
        "GRAY": gray,
        "B": region[:, :, 0],
        "G": region[:, :, 1],
        "R": region[:, :, 2],
        "H": hsv[:, :, 0],
        "S": hsv[:, :, 1],
        "V": hsv[:, :, 2],
        "L": lab[:, :, 0],
        "A": lab[:, :, 1],
        "B_LAB": lab[:, :, 2],
    }

    for channel, data in channels.items():
        p = np.percentile(data, [5, 25, 50, 75, 95])

        print(
            f"{channel:8s} | "
            f"min={data.min():3d} "
            f"mean={data.mean():6.1f} "
            f"median={p[2]:6.1f} "
            f"p5={p[0]:6.1f} "
            f"p25={p[1]:6.1f} "
            f"p75={p[3]:6.1f} "
            f"p95={p[4]:6.1f} "
            f"max={data.max():3d}"
        )


for name, region in regions.items():
    analyse(name, region)

# --------------------------------------------------
# SAVE VISUAL DIAGNOSTICS
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_DIR / "selected_regions.png"), display)

for name, region in regions.items():
    cv2.imwrite(
        str(OUTPUT_DIR / f"{name.lower()}_region.png"),
        region
    )

# --------------------------------------------------
# CHANNEL VISUALISATION
# --------------------------------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

channels = {
    "gray": gray,
    "red": img[:, :, 2],
    "green": img[:, :, 1],
    "blue": img[:, :, 0],
    "hue": hsv[:, :, 0],
    "saturation": hsv[:, :, 1],
    "value": hsv[:, :, 2],
    "lab_a": lab[:, :, 1],
    "lab_b": lab[:, :, 2],
}

for name, channel in channels.items():
    cv2.imwrite(
        str(OUTPUT_DIR / f"channel_{name}.png"),
        channel
    )

print("\nDiagnostic images saved in:", OUTPUT_DIR.resolve())
print("Analysis complete.")