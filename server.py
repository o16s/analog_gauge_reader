import base64
import json
import os
import tempfile
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline import process_image
from evaluation import constants

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DETECTION_MODEL = "models/gauge_detection_model.pt"
KEY_POINT_MODEL = "models/key_point_model.pt"
SEGMENTATION_MODEL = "models/segmentation_model.pt"

DEBUG_IMAGE_ORDER = [
    "original_image",
    "bbox_results",
    "image_cropped",
    "heatmaps_results",
    "key_point_results",
    "ellipse_results_key_points",
    "ellipse_zero_point",
    "ocr_visualization_results_chosen",
    "ocr_results_full",
    "ocr_results_numbers",
    "ocr_results_unit",
    "segmentation_results",
    "ellipse_results_projected",
    "ellipse_results_needle_point",
    "reading_line_fit",
    "ellipse_results_final",
]


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.post("/api/read")
async def read_gauge(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_path = os.path.join(tmp_dir, "run")

        reading = None
        unit = None
        errors = {}
        pipeline_error = None

        try:
            result = process_image(
                image=image,
                detection_model_path=DETECTION_MODEL,
                key_point_model_path=KEY_POINT_MODEL,
                segmentation_model_path=SEGMENTATION_MODEL,
                run_path=run_path,
                debug=True,
                eval_mode=True,
                image_is_raw=True,
            )
            reading = result["value"]
            unit = result["unit"]
        except Exception as e:
            pipeline_error = str(e)

        # Read error metrics
        error_path = os.path.join(run_path, constants.ERROR_FILE_NAME)
        if os.path.exists(error_path):
            with open(error_path) as f:
                errors = json.load(f)

        # Collect debug images in display order
        debug_images = {}
        if os.path.isdir(run_path):
            available = {os.path.splitext(f)[0]: f for f in os.listdir(run_path) if f.endswith(".jpg")}
            for name in DEBUG_IMAGE_ORDER:
                if name in available:
                    img_path = os.path.join(run_path, available[name])
                    debug_images[name] = encode_image(img_path)
                    del available[name]
            # Include any remaining images not in the predefined order
            for name, filename in sorted(available.items()):
                img_path = os.path.join(run_path, filename)
                debug_images[name] = encode_image(img_path)

        plt.close("all")

        if reading is not None:
            reading = round(float(reading), 4)

        return {
            "reading": reading,
            "unit": unit,
            "error": pipeline_error,
            "error_metrics": errors,
            "debug_images": debug_images,
        }


# Serve React build if it exists
frontend_build = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_build):
    from starlette.responses import FileResponse

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = os.path.join(frontend_build, path)
        if path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_build, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7432)
