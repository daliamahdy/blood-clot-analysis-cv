"""ROI selection: manual, full image, or a simple automatic proposal."""
import cv2
import numpy as np


def propose_auto_roi(image):
    """Propose a crop around the largest threshold component; not clot recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    best, best_score = None, 0
    for candidate in (binary, cv2.bitwise_not(binary)):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            fraction = area / (h * w)
            if not 0.005 <= fraction <= 0.80:
                continue
            score = area * (0.6 if x == 0 or y == 0 or x+bw >= w or y+bh >= h else 1)
            if score > best_score:
                px, py = max(5, int(bw*.08)), max(5, int(bh*.08))
                x0, y0 = max(0, x-px), max(0, y-py)
                x1, y1 = min(w, x+bw+px), min(h, y+bh+py)
                best, best_score = (x0, y0, x1-x0, y1-y0), score
    return best or (0, 0, w, h)


def choose_roi(image, mode):
    h, w = image.shape[:2]
    if mode == "full":
        return (0, 0, w, h)
    if mode == "auto":
        roi = propose_auto_roi(image)
        print(f"Proposed ROI (x, y, width, height): {roi}; inspect the preview.")
        return roi
    if mode == "manual":
        preview = image.copy()
        cv2.putText(preview, "Draw ROI; ENTER/SPACE accepts, C cancels",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        x, y, width, height = map(int, cv2.selectROI(
            "Select ROI", preview, showCrosshair=True, fromCenter=False))
        cv2.destroyWindow("Select ROI")
        if width <= 0 or height <= 0:
            raise RuntimeError("ROI selection cancelled or empty.")
        return x, y, width, height
    raise ValueError(f"Unknown ROI mode: {mode}")


def crop_roi(image, roi):
    x, y, width, height = roi
    h, w = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x+width), min(h, y+height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI is outside image bounds.")
    return image[y0:y1, x0:x1]
