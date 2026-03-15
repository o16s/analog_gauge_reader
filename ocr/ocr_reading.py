import re

import numpy as np

UNIT_LIST = ["bar", "mbar", "millibars", "MPa", "psi", "C", "°C", "F", "°F", "%"]

# Pattern to tokenize merged OCR text like "-2bar 2" into ["-2", "bar", "2"]
_TOKEN_RE = re.compile(
    r"-?\d+\.?\d*"   # numbers (including negative and decimal)
    r"|"
    r"[A-Za-z°%]+"   # letter/unit tokens
)


def _sub_polygon(polygon, text, start, end):
    """Estimate a sub-polygon for a token spanning characters [start, end)
    within the full text. Interpolates along the text direction (top-left →
    top-right and bottom-left → bottom-right edges of the quad)."""
    n = max(len(text), 1)
    t0 = start / n
    t1 = end / n
    # PaddleOCR polygons are ordered: top-left, top-right, bottom-right, bottom-left
    tl, tr, br, bl = polygon[0], polygon[1], polygon[2], polygon[3]
    new_tl = tl + t0 * (tr - tl)
    new_tr = tl + t1 * (tr - tl)
    new_br = bl + t1 * (br - bl)
    new_bl = bl + t0 * (br - bl)
    return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)


def split_ocr_readings(readings):
    """Split OCR readings that contain merged text (e.g. '-2bar 2') into
    individual OCRReading objects with estimated sub-polygons. Readings that
    are already a single number or unit are passed through unchanged."""
    out = []
    for r in readings:
        if r.is_number() or r.is_unit():
            out.append(r)
            continue
        tokens = _TOKEN_RE.findall(r.reading)
        if len(tokens) <= 1:
            out.append(r)
            continue
        # Find character offsets of each token in the original text
        raw = r.reading
        pos = 0
        for token in tokens:
            idx = raw.find(token, pos)
            if idx == -1:
                idx = pos
            sub_poly = _sub_polygon(r.polygon, raw, idx, idx + len(token))
            out.append(OCRReading(sub_poly, token, r.confidence))
            pos = idx + len(token)
    return out


class OCRReading:
    def __init__(self, polygon, reading, confidence):
        self.polygon = polygon
        self.reading = reading.strip()
        self.confidence = confidence

        if self.is_number():
            self.number = float(self.reading)

        self.center = self._get_centroid()

        self.theta = None

    def _get_centroid(self):
        x_mean = np.mean(self.polygon[:, 0])
        y_mean = np.mean(self.polygon[:, 1])

        return (x_mean, y_mean)

    def is_number(self):
        try:
            float(self.reading)
            return True
        except ValueError:
            return False

    def is_unit(self):
        return self.reading.lower() in [unit.lower() for unit in UNIT_LIST]

    def set_polygon(self, polygon):
        self.polygon = polygon
        self.center = self._get_centroid()

    def set_theta(self, theta):
        self.theta = theta

    def get_bounding_box(self):
        x_min = np.min(self.polygon[:, 0])
        y_min = np.min(self.polygon[:, 1])
        x_max = np.max(self.polygon[:, 0])
        y_max = np.max(self.polygon[:, 1])

        return (x_min, y_min, x_max, y_max)
