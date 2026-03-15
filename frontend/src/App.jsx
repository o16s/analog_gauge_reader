import { useState, useRef, useCallback } from "react";
import "./index.css";

const STAGE_LABELS = {
  original_image: "Original Image",
  bbox_results: "Gauge Detection",
  image_cropped: "Cropped Gauge",
  heatmaps_results: "Key Point Heatmaps",
  key_point_results: "Key Points",
  ellipse_results_key_points: "Ellipse Fit",
  ellipse_zero_point: "Zero Point",
  ocr_visualization_results_chosen: "OCR Visualization",
  ocr_results_full: "OCR Results (All)",
  ocr_results_numbers: "OCR Numbers",
  ocr_results_unit: "OCR Unit",
  segmentation_results: "Needle Segmentation",
  ellipse_results_projected: "Projected OCR Points",
  ellipse_results_needle_point: "Needle on Ellipse",
  reading_line_fit: "Angle-Reading Fit",
  ellipse_results_final: "Final Reading",
};

function label(name) {
  return STAGE_LABELS[name] || name.replace(/_/g, " ");
}

const METRIC_INFO = {
  "Ellipse fit error": {
    label: "Ellipse Fit",
    unit: "px",
    description:
      "Mean distance from detected notch points to the fitted ellipse. Measures how well the ellipse matches the scale markings.",
    thresholds: [3, 8],
  },
  "OCR numbers mean lack of confidence": {
    label: "OCR Confidence Loss",
    unit: "",
    description:
      "1 minus the mean OCR confidence across detected numbers. 0 = perfect confidence, 1 = no confidence.",
    thresholds: [0.1, 0.3],
    format: (v) => v.toFixed(3),
  },
  "Needle line residual variance": {
    label: "Needle Line Fit",
    unit: "",
    description:
      "Residual variance of the line fitted through segmented needle pixels (orthogonal distance regression). High values mean the needle mask is noisy or contains non-needle pixels.",
    thresholds: [50, 200],
  },
  "Mean residual on fitted angle line": {
    label: "Reading Fit",
    unit: "rad",
    description:
      "Mean absolute residual of the linear fit mapping ellipse angles to scale values. Indicates how consistently OCR numbers are spaced around the gauge.",
    thresholds: [1, 4],
  },
};

function metricLevel(value, thresholds) {
  if (value <= thresholds[0]) return "good";
  if (value <= thresholds[1]) return "fair";
  return "poor";
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [enlarged, setEnlarged] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/read", { method: "POST", body });
      setResult(await res.json());
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const debugImages = result?.debug_images;
  const hasDebug = debugImages && Object.keys(debugImages).length > 0;
  const hasMetrics =
    result?.error_metrics && Object.keys(result.error_metrics).length > 0;

  return (
    <div className="layout">
      <h1>Analog Gauge Reader</h1>

      <div
        className={`upload-area${dragOver ? " drag-over" : ""}${preview ? " has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        {preview ? (
          <img src={preview} alt="Preview" />
        ) : (
          <p>Drop a gauge image here, or click to select</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      <button className="primary" onClick={submit} disabled={!file || loading}>
        {loading ? "Processing\u2026" : "Read Gauge"}
      </button>

      {loading && (
        <div className="spinner-wrap">
          <div className="spinner" />
        </div>
      )}

      {result && (
        <>
          {result.error && result.reading == null && (
            <div className="error-msg">{result.error}</div>
          )}

          {result.reading != null && (
            <div className="reading-card">
              <span className="reading-value">{result.reading}</span>
              {result.unit && (
                <span className="reading-unit">{result.unit}</span>
              )}
            </div>
          )}

          {hasMetrics && (
            <div className="metrics">
              <div className="section-heading">Diagnostics</div>
              <div className="metrics-explanation">
                All values are error metrics &mdash; lower is better.
              </div>
              <div className="metrics-grid">
                {Object.entries(result.error_metrics).map(([key, value]) => {
                  const info = METRIC_INFO[key];
                  if (!info || typeof value !== "number") return null;
                  const level = metricLevel(value, info.thresholds);
                  const display = info.format
                    ? info.format(value)
                    : value.toFixed(2);
                  return (
                    <div key={key} className="metric-card">
                      <div className="metric-header">
                        <span className="metric-label">{info.label}</span>
                        <span className={`metric-dot ${level}`} />
                      </div>
                      <div className="metric-value">
                        {display}
                        {info.unit && (
                          <span className="metric-unit">{info.unit}</span>
                        )}
                      </div>
                      <div className="metric-desc">{info.description}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {hasDebug && (
            <>
              <div className="section-heading">Pipeline stages</div>
              <div className="image-grid">
                {Object.entries(debugImages).map(([name, b64]) => (
                  <div
                    key={name}
                    className="image-card"
                    onClick={() => setEnlarged(name)}
                  >
                    <img
                      src={`data:image/jpeg;base64,${b64}`}
                      alt={label(name)}
                    />
                    <div className="image-card-label">{label(name)}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {enlarged && debugImages?.[enlarged] && (
        <div className="lightbox" onClick={() => setEnlarged(null)}>
          <img
            src={`data:image/jpeg;base64,${debugImages[enlarged]}`}
            alt={label(enlarged)}
          />
          <div className="lightbox-label">{label(enlarged)}</div>
        </div>
      )}
    </div>
  );
}
