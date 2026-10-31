"""Video helpers: read the first frame, iterate frames, dump frames for SAM 2."""
from pathlib import Path
import cv2


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or float("nan"),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {path}")
    return frame


def iter_frames(path, every=1):
    """Yield (original_frame_index, BGR frame) for every Nth frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


def extract_frames(path, out_dir, every=1, quality=95):
    """Write sampled frames as 00000.jpg, 00001.jpg, ... (SAM 2's expected layout).

    Returns the list mapping position -> original frame index.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    originals = []
    for pos, (index, frame) in enumerate(iter_frames(path, every)):
        cv2.imwrite(str(out_dir / f"{pos:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        originals.append(index)
    if not originals:
        raise RuntimeError("No frames could be read from the video.")
    return originals
