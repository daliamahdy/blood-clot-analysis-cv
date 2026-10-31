# ClotScope — interactive clot area tracking

A small Streamlit UI for reviewing SAM 2 mask proposals, approving a clot mask, tracking it through a video, and plotting normalized area **A(t) / A(0)**.

## 1. Environment

Use Python 3.10 or 3.11. Install a PyTorch build that matches your CUDA setup from https://pytorch.org/get-started/locally/ first, then install SAM 2 following its official repository instructions:

https://github.com/facebookresearch/sam2

Install the UI dependencies:

```bash
pip install -r requirements.txt
```

From the SAM 2 repository, install the package (for example, from a clone):

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

Download a SAM 2.1 checkpoint from the official SAM 2 repository's checkpoints instructions. Put it at `checkpoints/sam2.1_hiera_large.pt` or enter the actual path in the app sidebar. The default config is `configs/sam2.1/sam2.1_hiera_l.yaml`; this path is resolved by SAM 2's installed package/config setup, so follow the current official installation guidance if config lookup fails.

## 2. Run

```bash
streamlit run app.py
```

Upload an MP4/MOV/AVI/MKV video, generate mask proposals from the first frame, choose the mask that corresponds to the clot, approve it, and run tracking. The app shows an overlay preview, a normalized area curve, and a CSV download.

## Notes

- The first approved mask is the reference: normalized area is `A(t) / A(0)`, so the baseline is 1.0 and the result is dimensionless.
- SAM 2 propagates the approved mask through the video. Inspect the overlay at multiple points; tracking can drift if the clot changes appearance, becomes occluded, or moves out of view.
- The measured area is a 2D pixel area, not true physical surface area or volume. Calibration is needed for mm², and volume requires additional imaging/geometric assumptions.
- Large videos require RAM and disk space because frames are temporarily extracted for SAM 2.
