"""Area, perimeter, and intensity measurements."""
import cv2
import numpy as np


def to_gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def measure(image, mask):
    gray = to_gray(image)
    inside = gray[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    return {
        "area_px": float(cv2.contourArea(contour)) if contour is not None else 0.0,
        "foreground_pixels": int(mask.sum()),
        "perimeter_px": float(cv2.arcLength(contour, True)) if contour is not None else 0.0,
        "mean_gray_inside": float(inside.mean()) if inside.size else float("nan"),
        "median_gray_inside": float(np.median(inside)) if inside.size else float("nan"),
        "std_gray_inside": float(inside.std()) if inside.size else float("nan"),
    }


def area_px_to_mm2(area_px, pixels_per_mm):
    """Convert pixel² to mm² after calibration using a known length in this setup."""
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    return area_px / (pixels_per_mm ** 2)
