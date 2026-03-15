# Analog Gauge Reader

Paper: "Under Pressure: Learning-Based Analog Gauge Reading In The Wild" (Reitsma et al., ETH Zurich, 2024)

## What it does

Reads analog industrial gauges from images without prior knowledge of gauge type, scale range, or units. Outputs a numerical reading and unit (e.g., `3.1 psi`).

## Pipeline stages

The main entry point is `pipeline.py`. It runs 6 sequential stages:

### 1. Gauge Detection (`gauge_detection/detection_inference.py`)
- YOLOv8 (fine-tuned on ~2000 gauge images) detects gauge face bounding boxes
- Crops the best detection and resizes to 448x448

### 2. Key Point (Notch) Detection (`key_point_detection/`)
- DINOv2 backbone with a lightweight CNN decoder predicts heatmaps for scale notches
- Three heatmap channels: start notch, intermediate notches, end notch
- Mean-Shift clustering extracts individual notch coordinates from heatmaps
- Requires at least 5 notches for the next stage

### 3. Ellipse Fitting & Zero-Point (`geometry/ellipse.py`)
- Fits an ellipse through the detected notch points (Halir & Flusser algorithm)
- Handles perspective distortion — gauge scales appear elliptical when viewed at an angle
- Computes the **zero-point** (wrap-around point) — see below

#### The zero-point (wrap-around point)

Positions on the ellipse are represented as polar angles (0 to 2π). The "zero-point" defines
where the angle wraps from 2π back to 0. If this wrap happens in the middle of the gauge scale,
adjacent scale markers would have discontinuous angles (e.g. 5.8 rad → 0.1 rad), breaking the
linear fit in stage 6.

The solution: place the zero-point in the **dead zone** — the gap between the scale's end and
start, where there are no markings. The code computes it as the midpoint angle between the
detected start and end notches (`pipeline.py:244-246`):

```python
theta_start = get_polar_angle(start_point, ellipse_params)
theta_end = get_polar_angle(end_point, ellipse_params)
theta_zero = get_theta_middle(theta_start, theta_end)
```

This guarantees all scale positions map to a continuous, monotonically increasing angle range,
so the angle → reading linear fit works cleanly. In the debug output (`ellipse_zero_point.jpg`),
the green dot shows the zero-point and the red dots show the start/end of the scale.

### 4. Needle Segmentation (`segmentation/segmenation_inference.py`)
- YOLOv8 instance segmentation model (fine-tuned on ~2000 gauge images) segments needle pixels
- Orthogonal Distance Regression fits a line through the mask pixels
- Note: typo in filename (`segmenation`) is intentional — don't rename, it would break imports

### 5. OCR — Scale Marker Recognition (`ocr/ocr_inference.py`)
- PaddleOCR v5 mobile (PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec)
- Replaced the original mmocr (DBNet + ABINet) which could not read decimal numbers
- Image is warped to correct perspective (ellipse → circle) and rotated before OCR
- Detected text is classified as numbers or units (bar, mbar, psi, MPa)
- **Merged detection splitting** (`ocr/ocr_reading.py:split_ocr_readings`): PaddleOCR's text
  detector sometimes merges nearby labels into one box (e.g. `"-2bar 2"`). The split function
  tokenizes these into individual numbers and units, estimating sub-polygons by interpolating
  along the original bounding box proportionally to character position.
- Numbers are filtered: confidence > 0.7, heuristics to reject serial numbers

### 6. Reading Computation (back in `pipeline.py`)
- Each OCR number's bounding box center is projected onto the ellipse → angle
- RANSAC linear fit maps angles to scale values (rejects outlier OCR detections)
- Needle line intersects the ellipse → needle angle
- Needle angle is interpolated through the linear model → final reading

## Known limitations

- **OCR** was the weakest stage with the original ABINet model (~40-57% failure rate, couldn't read decimals). Swapping to PaddleOCR v5 significantly improved this — it correctly reads "0.2", "0.4", etc.
- **Needle segmentation** can pick up printed text/numbers as needle pixels when they have similar contrast, corrupting the line fit.
- **Small or distant gauges** (like 7.jpg) fail at detection.
- **Blue/colored needles** matching label colors confuse segmentation.
- When OCR works, the system achieves ~2% relative reading error.

## Running

```shell
# Setup
uv venv --python 3.10
source .venv/bin/activate
uv pip install "setuptools<58" wheel
uv pip install torch==2.0.0 torchvision==0.15.1
pip install mmcv==2.0.0 --no-build-isolation
uv pip install mmengine==0.7.2 mmdet==3.0.0 ultralytics==8.0.66 scikit-learn==1.2.2
uv pip install "numpy<2"
uv pip install paddlepaddle paddleocr

# Models are stored in Git LFS — pull them first
git lfs pull

# Run
python pipeline.py --input <image_or_folder> --base_path <output_dir> --debug --eval
```

- `--debug` saves intermediate visualizations for each stage
- `--eval` saves `result_full.json` with detailed per-stage outputs
- Models default to `models/gauge_detection_model.pt`, `models/key_point_model.pt`, `models/segmentation_model.pt`

## Project structure

- `pipeline.py` — main pipeline, `process_image()` is the core function
- `gauge_detection/` — YOLOv8 gauge face detector
- `key_point_detection/` — DINOv2-based notch/keypoint detector + training code
- `geometry/` — ellipse fitting and polar coordinate math
- `segmentation/` — YOLOv8 needle segmentation
- `ocr/` — PaddleOCR-based text detection and recognition (was mmocr)
- `angle_reading_fit/` — angle-to-reading linear model and RANSAC
- `evaluation/` — evaluation scripts and metrics
- `plots.py` — debug visualization (saves images per stage)
- `server.py` — FastAPI web server (`POST /api/read` + serves React SPA)
- `frontend/` — React + Vite SPA for uploading images and viewing results
- `ros_node.py` — ROS wrapper for robotic deployment
