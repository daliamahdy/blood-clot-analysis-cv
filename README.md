# Blood Clot Analysis using Computer Vision
![Blood Clot Analysis Demo](demo_output.png)

A Python-based computer vision project for automated blood clot segmentation, area analysis, and robot-head detection in image and video data.

The project uses **classical computer vision techniques with OpenCV and NumPy**, without relying on deep-learning models or segmentation frameworks.

## Approach

The project explores traditional image-processing methods for biological image analysis, combining colour-channel analysis, thresholding, morphological refinement, contour detection, and temporal information.

## Pipeline Overview

```mermaid
flowchart TD
    A[Input Images / Video] --> B[Frame Extraction]
    B --> C[Image Preprocessing]

    C --> D1[Blood Clot Analysis]
    C --> D2[Robot-Head Detection]

    D1 --> E1[Red Channel Extraction]
    E1 --> F1[Thresholding]
    F1 --> G1[Morphological Refinement]
    G1 --> H1[Region Selection & Recovery]
    H1 --> I1[Temporal Refinement]
    I1 --> J1[Clot Segmentation Mask]

    D2 --> E2[Right-Side ROI Selection]
    E2 --> F2[Grayscale & Thresholding]
    F2 --> G2[Contour Detection]
    G2 --> H2[Shape-Based Filtering]
    H2 --> I2[Robot-Head Localization]

    J1 --> K[Visualization & Results]
    I2 --> K
```

## Features

### 1. Blood Clot Segmentation
- Red-channel-based image analysis.
- Gaussian smoothing and threshold.
- Morphological operations to refine segmentation.
- Connected-component analysis to identify the clot region.

### 2. Clot Visualization
- Semi-transparent overlay of the detected clot.
- Individual binary segmentation masks.
- Diagnostic images to inspect intermediate processing steps.

### 3. Robot-Head Detection
- Detection of the robot head within the rightmost third of the image.
- Grayscale conversion and threshold-based candidate extraction.
- Contour analysis using area, circularity, and aspect ratio.
- Visual bounding-box annotation.

## Output

The pipeline generates results such as:

- Segmentation masks.
- Clot overlays on original images.
- Intermediate diagnostic images.
- Robot-head detection visualizations.
- Area-analysis results, where enabled.

Results are saved in the configured output directory.
## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib

## Project Structure

```text
blood_clot_cv_project/
│
├── clotcv/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline.py
│   └── ...
│
├── input/
│   ├── images/
│   └── videos/
│
├── results/
│
├── requirements.txt
└── README.md
```

*The structure above is illustrative; adapt it to the actual files in your repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/daliamahdy/blood-clot-analysis-cv.git
cd blood-clot-analysis-cv
```

Create and activate a virtual environment:

```bash
python -m venv .clotenv
source .clotenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis on a video:

```bash
python -m clotcv.cli --input "input/videos/v1.mp4" --every 10 --max-samples 20
```

Run the analysis on an image directory:

```bash
python -m clotcv.cli --input "input/images/Set1"
```

Adjust the input paths and command-line arguments according to your dataset and configuration.



