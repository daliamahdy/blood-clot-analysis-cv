# Blood Clot Segmentation + Robot Head Detection

Classical computer vision (OpenCV, no ML) to segment a blood clot inside a tube and locate the robot head in every photo in a folder.

![final overlay](docs/final_overlay.png)

## How it works

**Clot**

1. Take the red channel (the clot is dark there, so contrast is high)
2. Gamma correction to separate the clot from the background
3. Gaussian blur + inverse threshold
4. Morphological close, fill holes, keep the largest blob

**Robot head**

1. Take the blurred red channel
2. Black-hat transform to find small dark blobs, independent of uneven lighting
3. Threshold, then remove the clot (plus a margin) and everything outside the tube
4. Keep the largest remaining blob
5. Keep its darkest core and center a box on the intensity-weighted centroid

Every step is saved as a numbered image in `results/visualizations/steps/`.

## Setup

```bash
pip install opencv-python numpy
```

Expected layout:

```
.
└── clotcv/
    ├── input/images/Set1/
    │   ├── photo_01.png
    │   └── photo_02.png
    └── ground.py
```

## Usage

```bash
# From the clotcv folder, process the default folder: input/images/Set1
python ground.py

# Or pass another folder containing photos
python ground.py input/images/Set2
```

Optional environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `IN` | Input folder containing photos | `input/images/Set1` |
| `OUT` | Output folder | `clotcv/results` |
| `NOSHOW` | Set to any value to skip the preview window | unset |

## Output

- `<photo>_clot_mask.png` – binary clot mask
- `<photo>_final_overlay.png` – clot (blue) and head box (green) over the original
- `<photo>_robot_head_mask.png` – robot-head mask
- `visualizations/steps/` – one image per pipeline step
- Console: clot area in pixels and head box `(x, y, w, h)`

## Parameters

Tuned for one imaging setup. Adjust at the top of `ground.py` or in the function arguments:

- `TUBE_Y_TOP_RATIO`, `TUBE_Y_BOTTOM_RATIO` – vertical limits of the tube interior
- `gamma`, `threshold`, `ksize` in `segment_clot` – clot segmentation
- `thresh`, `core_frac`, `min_area` in `find_head` – head detection
