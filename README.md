# analog_gauge_reader

<p align="center">
<img src=method_overview.png>
</p>

Fork of [ethz-asl/analog_gauge_reader](https://github.com/ethz-asl/analog_gauge_reader) — the code for the paper [Under Pressure: Learning-Based Analog Gauge Reading In The Wild](https://arxiv.org/abs/2404.08785) by Maurits Reitsma, Julian Keller, Kenneth Blomqvist and Roland Siegwart.

### Changes in this fork

- **Replaced mmocr (DBNet + ABINet) with PaddleOCR v5** for text detection and recognition. PaddleOCR correctly reads decimal numbers (0.2, 0.4, 0.6, 0.8) that ABINet consistently missed, dramatically improving results on real-world gauges.
- **Switched to `uv` for environment management** instead of Poetry/conda.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install dependencies:

```shell
uv venv --python 3.10
source .venv/bin/activate

# Core dependencies
uv pip install "setuptools<58" wheel
uv pip install torch==2.0.0 torchvision==0.15.1
pip install mmcv==2.0.0 --no-build-isolation
uv pip install mmengine==0.7.2 mmdet==3.0.0 ultralytics==8.0.66 scikit-learn==1.2.2
uv pip install "numpy<2"

# OCR (PaddleOCR replaces mmocr)
uv pip install paddlepaddle paddleocr
```

Pull the model weights (stored in Git LFS):

```shell
git lfs install
git lfs pull
```

## Run pipeline

```shell
python pipeline.py --input path/to/image_or_folder --base_path path/to/results --debug --eval
```

- `--input` — a single image or a directory of images
- `--base_path` — output directory (a timestamped run folder is created inside)
- `--debug` — save intermediate visualizations for each pipeline stage
- `--eval` — save detailed per-stage results to `result_full.json`

Models default to the `models/` directory. Override with `--detection_model`, `--key_point_model`, `--segmentation_model`.

### Output

Each image gets a folder containing:
- `result.json` — the final reading and unit
- `error.json` — error metrics for each stage
- `result_full.json` — detailed per-stage outputs (with `--eval`)
- Debug images for each stage (with `--debug`)

## Run experiments

The scripts `experiments.sh` and `evaluations.sh` run the pipeline and evaluations on multiple folders. Modify the paths inside to match your data.
