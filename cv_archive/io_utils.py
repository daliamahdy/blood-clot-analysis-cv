"""Load images from a folder or sampled frames from a video."""
from pathlib import Path
import cv2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_samples(input_path, every=1, max_samples=None):
    """Yield (name, index, time_seconds, BGR image)."""
    path = Path(input_path)
    if every < 1:
        raise ValueError("every must be >= 1")

    if path.is_dir():
        paths = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise FileNotFoundError(f"No supported images in {path}")
        count = 0
        for index, image_path in enumerate(paths):
            if index % every:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            yield image_path.name, index, float("nan"), image
            count += 1
            if max_samples is not None and count >= max_samples:
                break
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    index = count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every == 0:
                time_s = index / fps if fps and fps > 0 else float("nan")
                yield path.name, index, time_s, frame
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
            index += 1
    finally:
        cap.release()
