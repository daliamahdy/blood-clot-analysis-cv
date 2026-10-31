"""Command-line arguments."""
import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Blood-clot image/video analysis")
    parser.add_argument("--input", required=True, help="Folder of screenshots or video file")
    parser.add_argument("--output", default="results")
    parser.add_argument("--roi", choices=["manual", "auto", "full"], default="manual")
    parser.add_argument("--method", choices=["otsu", "adaptive", "grabcut"], default="otsu")
    parser.add_argument("--polarity", choices=["dark", "bright"], default="dark")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-object-area", type=int, default=100)
    parser.add_argument("--adaptive-block", type=int, default=51)
    parser.add_argument("--adaptive-offset", type=int, default=5)
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, args.roi, args.method, args.polarity,
        args.every, args.max_samples, args.min_object_area,
        args.adaptive_block, args.adaptive_offset, args.fill_holes)


if __name__ == "__main__":
    main()
