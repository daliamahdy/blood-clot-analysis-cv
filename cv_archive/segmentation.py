
# Image segmentation methods and binary-mask cleanup.
import cv2
import numpy as np
from skimage.filters import threshold_local


def to_gray(image):
    # Convert a BGR or grayscale image to grayscale.
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def segment(
    image,
    method="otsu",
    polarity="dark",
    adaptive_block=51,
    adaptive_offset=5,
    enhance_contrast=True,
):
    # Return a boolean mask and its numeric threshold.
    # Methods: otsu, adaptive, grabcut.
    # Polarity: dark selects darker pixels; bright selects brighter pixels.
    # Threshold is None for methods without a global threshold.

    gray = to_gray(image)

    if polarity not in ("dark", "bright"):
        raise ValueError("polarity must be 'dark' or 'bright'.")

    # Improve local contrast when the clot is faint or illumination varies.
    if enhance_contrast:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        processed = clahe.apply(gray)
    else:
        processed = gray.copy()

    blurred = cv2.GaussianBlur(processed, (5, 5), 0)

    if method == "otsu":
        # Invert the binary threshold to select dark regions.
        threshold_type = (
            cv2.THRESH_BINARY_INV
            if polarity == "dark"
            else cv2.THRESH_BINARY
        )
        threshold, binary = cv2.threshold(
            blurred,
            0,
            255,
            threshold_type + cv2.THRESH_OTSU,
        )
        return binary > 0, float(threshold)

    if method == "adaptive":
        # Ensure the local threshold window is odd and valid for the image.
        block = max(3, int(adaptive_block))
        if block % 2 == 0:
            block += 1

        max_block = min(gray.shape[:2])
        if max_block % 2 == 0:
            max_block -= 1
        if max_block < 3:
            raise ValueError("Image is too small for adaptive thresholding.")
        block = min(block, max_block)

        # Calculate a local threshold to handle uneven illumination.
        local_threshold = threshold_local(
            blurred,
            block_size=block,
            offset=adaptive_offset,
        )

        if polarity == "dark":
            mask = blurred < local_threshold
        else:
            mask = blurred > local_threshold

        return mask, None

    if method == "grabcut":
        # Segment the ROI using GrabCut.
        h, w = image.shape[:2]
        if h < 5 or w < 5:
            raise ValueError("ROI too small for GrabCut.")

        color = (
            image
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        )

        # Initialize pixels as probable foreground and mark the border as background.
        gc_mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
        gc_mask[:2, :] = cv2.GC_BGD
        gc_mask[-2:, :] = cv2.GC_BGD
        gc_mask[:, :2] = cv2.GC_BGD
        gc_mask[:, -2:] = cv2.GC_BGD

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            color,
            gc_mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        return np.isin(
            gc_mask,
            [cv2.GC_FGD, cv2.GC_PR_FGD],
        ), None

    raise ValueError(f"Unknown method: {method}")


def clean_mask(
    mask,
    min_object_area=100,
    close_kernel=3,
    fill_holes=False,
    largest_component=False,
):
    # Close small gaps and remove small connected components.
    # Optionally fill holes and keep only the largest component.
    # Use largest_component=True only when the ROI contains one main object.

    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255

    if binary.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    k = max(1, int(close_kernel))
    if k % 2 == 0:
        k += 1

    if k > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)
    components = []

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_object_area:
            components.append((label, area))

    if largest_component and components:
        # Keep only the largest connected foreground component.
        largest_label = max(components, key=lambda item: item[1])[0]
        cleaned[labels == largest_label] = 255
    else:
        for label, _ in components:
            cleaned[labels == label] = 255

    if fill_holes and cleaned.any():
        # Fill enclosed holes without changing the exterior background.
        flood = cleaned.copy()
        flood_mask = np.zeros(
            (cleaned.shape[0] + 2, cleaned.shape[1] + 2),
            np.uint8,
        )
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        cleaned = cv2.bitwise_or(cleaned, holes)

    return cleaned > 0
