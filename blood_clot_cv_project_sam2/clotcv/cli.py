"""Command-line arguments."""
import argparse
from .pipeline import run


def _xy(text):
    x, y = text.split(",")
    return float(x), float(y)


def _box(text):
    x, y, w, h = (float(v) for v in text.split(","))
    return x, y, w, h


def main():
    p = argparse.ArgumentParser(description="Blood-clot tracking: SAM 2 + human approval")
    p.add_argument("--input", required=True, help="Video file")
    p.add_argument("--output", default="results")
    p.add_argument("--backend", choices=["sam2", "grabcut"], default="sam2",
                   help="grabcut = lightweight stand-in for testing without SAM 2")
    p.add_argument("--model-size", choices=["tiny", "small", "base_plus", "large"], default="large")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--every", type=int, default=1, help="process every Nth frame")
    p.add_argument("--vis-every", type=int, default=30,
                   help="save an overlay JPG/mask PNG every N original frames (0 = none)")
    p.add_argument("--mask-file", default=None,
                   help="reuse an approved first_frame_mask.png and skip the review window")
    p.add_argument("--box", type=_box, default=None, help="initial box x,y,w,h (pixels)")
    p.add_argument("--point", type=_xy, action="append", default=[], help="initial + point x,y")
    p.add_argument("--neg-point", type=_xy, action="append", default=[], help="initial - point x,y")
    p.add_argument("--auto-approve", action="store_true",
                   help="skip the review window (needs --box/--point); for scripted runs")
    p.add_argument("--pixels-per-mm", type=float, default=None, help="adds area_mm2 column")
    a = p.parse_args()
    if a.every < 1:
        p.error("--every must be >= 1")
    points = a.point + a.neg_point
    labels = [1] * len(a.point) + [0] * len(a.neg_point)
    run(a.input, a.output, a.backend, a.model_size, a.checkpoint, a.device, a.every,
        a.vis_every, a.mask_file, points, labels, a.box, a.auto_approve, a.pixels_per_mm)
