from ultralytics import YOLO
import numpy as np
import cv2
from scipy import odr


def _extract_mask(results, image_shape):
    """Extract the top-confidence needle mask from YOLO results.
    Returns (x_coords, y_coords) or None if no mask found."""
    masks = results[0].masks
    if masks is None:
        return None
    try:
        needle_mask = masks.data[0].numpy()
    except:
        needle_mask = masks.data[0].cpu().numpy()
    needle_mask_resized = cv2.resize(needle_mask,
                                     dsize=(image_shape[1], image_shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    y_coords, x_coords = np.where(needle_mask_resized)
    if len(x_coords) == 0:
        return None
    return x_coords, y_coords


def segment_gauge_needle(image, model_path='best.pt'):
    """
    uses fine-tuned yolo v8 to get mask of segmentation.
    Falls back to grayscale if color segmentation finds no needle
    (handles colored needles the model wasn't trained on).
    :param image: numpy image (RGB)
    :param model_path: path to yolov8 segmentation model
    :return: segmentation of needle as (x_coords, y_coords)
    """
    model = YOLO(model_path)

    results = model.predict(image)
    mask = _extract_mask(results, image.shape)
    if mask is not None:
        return mask

    # Fallback: convert to grayscale so colored needles appear dark
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    results = model.predict(gray_3ch)
    mask = _extract_mask(results, image.shape)
    if mask is not None:
        return mask

    # Neither worked — raise AttributeError to match original behavior
    raise AttributeError("No needle mask found in color or grayscale")


def get_fitted_line(x_coords, y_coords):
    """
    Do orthogonal distance regression (odr) for this.
    """
    odr_model = odr.Model(linear)
    data = odr.Data(x_coords, y_coords)
    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0.2, 1.], maxit=600)
    out = ordinal_distance_reg.run()
    line_coeffs = out.beta
    residual_variance = out.res_var
    return line_coeffs, residual_variance


def linear(B, x):
    return B[0] * x + B[1]


def get_start_end_line(needle_mask):
    return np.min(needle_mask), np.max(needle_mask)


def cut_off_line(x, y_min, y_max, line_coeffs):
    line = np.poly1d(line_coeffs)
    y = line(x)
    _cut_off(x, y, y_min, y_max, line_coeffs, 0)
    _cut_off(x, y, y_min, y_max, line_coeffs, 1)
    return x[0], x[1]


def _cut_off(x, y, y_min, y_max, line_coeffs, i):
    if y[i] > y_max:
        y[i] = y_max
        x[i] = 1 / line_coeffs[0] * (y_max - line_coeffs[1])
    if y[i] < y_min:
        y[i] = y_min
        x[i] = 1 / line_coeffs[0] * (y_min - line_coeffs[1])
