"""Finetune the needle segmentation model on new data.

Tracks experiments with MLflow (live per-epoch curves). Launch the UI with: mlflow ui

Usage:
    python scripts/finetune_segmentation.py
    python scripts/finetune_segmentation.py --epochs 100 --batch 8
    python scripts/finetune_segmentation.py --data path/to/data.yaml --base yolov8n-seg.pt
    python scripts/finetune_segmentation.py --run-name "colored-needles-v2"
"""
import argparse
import shutil
from pathlib import Path

import mlflow
from ultralytics import YOLO

DEFAULT_MODEL = "models/segmentation_model.pt"
DEFAULT_DATA = "data/segmentation/data.yaml"
EXPERIMENT_NAME = "needle-segmentation"


def _on_train_epoch_end(trainer):
    """YOLO callback: log training loss metrics to MLflow after each epoch."""
    epoch = trainer.epoch
    for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
        mlflow.log_metric(k, float(v), step=epoch)
    mlflow.log_metric("train/lr", float(trainer.optimizer.param_groups[0]["lr"]), step=epoch)


def _on_val_end(validator):
    """YOLO callback: log validation metrics to MLflow after each validation."""
    epoch = getattr(validator, "epoch", None)
    if epoch is None and hasattr(validator, "training") and hasattr(validator.training, "epoch"):
        epoch = validator.training.epoch
    metrics = validator.metrics
    if hasattr(metrics, "results_dict"):
        for k, v in metrics.results_dict.items():
            if isinstance(v, (int, float)):
                name = "val/" + k.replace("(", "_").replace(")", "")
                mlflow.log_metric(name, float(v), step=epoch)


def _on_fit_epoch_end(trainer):
    """YOLO callback: log combined metrics after each train+val epoch."""
    epoch = trainer.epoch
    for k, v in trainer.metrics.items():
        if isinstance(v, (int, float)):
            name = k.replace("(", "_").replace(")", "")
            mlflow.log_metric(name, float(v), step=epoch)


def main():
    parser = argparse.ArgumentParser(description="Finetune needle segmentation model")
    parser.add_argument("--base", default=DEFAULT_MODEL,
                        help="Base model to finetune from (default: %(default)s)")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to data.yaml (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu', '0', 'mps', etc. (default: auto)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers (default: %(default)s)")
    parser.add_argument("--install", action="store_true",
                        help="Copy best weights to models/segmentation_model.pt after training")
    parser.add_argument("--run-name", default=None,
                        help="MLflow run name (default: auto-generated)")
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "base_model": args.base,
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr,
            "device": str(args.device),
        })

        model = YOLO(args.base)

        # Register live logging callbacks
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        results = model.train(
            task="segment",
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr,
            device=args.device,
            workers=args.workers,
            plots=True,
        )

        # Log training plots and curves as artifacts
        save_dir = Path(results.save_dir)
        for pattern in ("*.png", "*.jpg", "*.csv"):
            for f in save_dir.glob(pattern):
                mlflow.log_artifact(str(f))

        # Log best weights
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")

        print(f"\nBest weights: {best}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")

        if args.install:
            dest = Path(DEFAULT_MODEL)
            shutil.copy2(best, dest)
            print(f"Installed to {dest}")
            mlflow.set_tag("installed", "true")
        else:
            print(f"To install: cp {best} {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
