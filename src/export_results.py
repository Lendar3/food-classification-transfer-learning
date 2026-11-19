"""Export saved, completed experiments without evaluating or retraining models."""

import csv
from pathlib import Path

import numpy as np
import torch

from src.training import RESULTS_DIR, WEIGHTS_DIR


def export_results():
    output = RESULTS_DIR / "summary"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["run", "method", "dataset", "split", "seed", "device", "batch_size",
              "train_images", "evaluation_images", "epochs", "best_epoch", "learning_rate",
              "backbone_learning_rate", "total_parameters", "trainable_parameters",
              "stage_seconds", "total_training_seconds", "cross_entropy", "accuracy",
              "macro_f1", "parent_checkpoint", "source"]
    rows = []
    skipped = []
    for path in sorted(WEIGHTS_DIR.glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not all(key in saved for key in ("epochs", "training_seconds", "device", "history_file")):
            skipped.append(path.name)
            continue
        validation = saved["validation"]
        row = dict(zip(fields, [path.stem, saved["method"], saved["dataset_name"], "validation",
                               saved["seed"], saved["device"], saved["batch_size"],
                               saved["train_images"], saved["val_images"], saved["epochs"],
                               saved["best_epoch"], saved["learning_rate"], saved["backbone_learning_rate"],
                               saved["total_parameters"], saved["trainable_parameters"],
                               saved["stage_seconds"], saved["training_seconds"], validation["loss"],
                               validation["accuracy"], validation["macro_f1"], saved["parent_checkpoint"],
                               f"data/weights/{path.name}"]))
        rows.append(row)
        # Export only an existing test measurement, never run a new test evaluation here.
        predictions_path = RESULTS_DIR / "plots" / path.stem / "test" / "predictions.npz"
        if predictions_path.exists():
            with np.load(predictions_path) as predictions:
                rows.append({**row, "split": "test", "evaluation_images": len(predictions["answers"]),
                             "cross_entropy": float(predictions["loss"]),
                             "accuracy": float(predictions["accuracy"]),
                             "macro_f1": float(predictions["macro_f1"]),
                             "source": predictions_path.relative_to(RESULTS_DIR.parent).as_posix()})
    if not rows:
        raise ValueError("No completed checkpoints found in data/weights/.")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to results/summary/metrics.csv")
    for name in skipped:
        print(f"Skipped incomplete run: {name}")
    return output / "metrics.csv"


if __name__ == "__main__":
    export_results()
