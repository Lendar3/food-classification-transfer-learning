from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch

from src.data.data import PROCESSED_DIR
from src.data.dataset import load_dataset
from src.training import RESULTS_DIR, WEIGHTS_DIR, load_model

PLOTS_DIR = RESULTS_DIR / "plots"
COLORS = {"cnn": "#3977b8", "mobilenet_frozen": "#d68b29", "mobilenet_finetune": "#728b46"}
NAMES = {"cnn": "Custom CNN", "mobilenet_frozen": "Frozen MobileNetV2",
         "mobilenet_finetune": "Fine-tuned MobileNetV2"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_history(path, output, best_epoch):
    with np.load(path) as history:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for axis, metric, title in zip(axes, ["loss", "accuracy", "macro_f1"],
                                      ["Cross-entropy", "Accuracy", "Macro-F1"]):
            for split, color, style in [("train", "#3977b8", "-"), ("val", "#d68b29", "--")]:
                axis.plot(history["epoch"], history[f"{split}_{metric}"],
                          color=color, linestyle=style, marker=".", label=split)
            axis.axvline(best_epoch, color="#666666", linestyle=":", label="Best epoch")
            axis.set(xlabel="Epoch", ylabel=title, title=title)
            axis.grid(alpha=0.2)
            epochs = history["epoch"]
            axis.set_xticks(np.unique(np.linspace(epochs[0], epochs[-1], min(len(epochs), 6), dtype=int)))
            if metric != "loss":
                axis.set_ylim(0, 1.02)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend()
        fig.suptitle("Learning curves · train metrics measured during weight updates")
        save_figure(fig, output / "learning_curves.png")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    answers, predictions, confidence = [], [], []
    total_loss = 0.0
    for images, labels in loader:
        logits = model(images.to(device))
        total_loss += torch.nn.functional.cross_entropy(
            logits, labels.to(device), reduction="sum").item()
        scores, classes = logits.softmax(dim=1).max(dim=1)
        answers.extend(labels.tolist())
        predictions.extend(classes.cpu().tolist())
        confidence.extend(scores.cpu().tolist())
    return np.array(answers), np.array(predictions), np.array(confidence), total_loss / len(answers)


def plot_confusion(matrix, names, output, title):
    size = max(7, len(names) * 0.23)
    for normalized in [False, True]:
        values = matrix
        if normalized:
            totals = matrix.sum(axis=1, keepdims=True)
            values = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
        fig, axis = plt.subplots(figsize=(size, size), layout="constrained")
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        axis.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
                 xticklabels=names, yticklabels=names, xlabel="Predicted class", ylabel="True class",
                 title=f"{title} · {'row-normalized confusion matrix' if normalized else 'confusion matrix (counts)'}")
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        bar = fig.colorbar(image, ax=axis, shrink=0.7)
        if normalized:
            bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
        if len(names) <= 15:
            for i, j in np.ndindex(matrix.shape):
                text = f"{values[i, j]:.0%}" if normalized else str(matrix[i, j])
                axis.text(j, i, text, ha="center", va="center",
                          color="white" if values[i, j] > values.max() / 2 else "black")
        save_figure(fig, output / ("confusion_normalized.png" if normalized else "confusion_counts.png"))


def plot_class_metrics(precision, recall, f1, support, names, output, title):
    indices = np.flatnonzero(support)
    indices = indices[np.argsort(f1[indices], kind="stable")[:15]]
    y = np.arange(len(indices))
    fig, axis = plt.subplots(figsize=(11, max(4, len(indices) * 0.45)), layout="constrained")
    for offset, values, label, color, hatch in [(-0.24, precision, "Precision", "#3977b8", ""),
                                               (0, recall, "Recall", "#d68b29", "//"),
                                               (0.24, f1, "F1", "#555555", "..")]:
        bars = axis.barh(y + offset, values[indices], height=0.23, color=color, hatch=hatch, label=label)
        axis.bar_label(bars, fmt=lambda value: f"{value:.0%}", padding=3, fontsize=7)
    axis.set(yticks=y, yticklabels=[f"{names[i]} (n={support[i]})" for i in indices],
             xlim=(0, 1.12), xlabel="Score", title=f"{title} · lowest-F1 supported classes (up to 15)")
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="lower right")
    save_figure(fig, output / "class_metrics.png")


def plot_errors(matrix, answers, predictions, confidence, dataset, output, title):
    names = dataset.class_names
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    pairs = np.argwhere(errors > 0)
    pairs = sorted(pairs, key=lambda pair: -errors[tuple(pair)])[:10]
    if pairs:
        fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
        axis.barh(np.arange(len(pairs)), [errors[i, j] for i, j in pairs], color="#3977b8")
        axis.set(yticks=np.arange(len(pairs)), yticklabels=[fill(f"{names[i]} → {names[j]}", 40) for i, j in pairs],
                 xlabel="Number of errors", title=f"{title} · most frequent confusions")
        axis.invert_yaxis()
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        save_figure(fig, output / "top_confusions.png")

    indices = np.flatnonzero(answers != predictions)
    indices = indices[np.argsort(-confidence[indices])[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for axis, i in zip(axes.flat, indices):
        axis.imshow(dataset.X[i].permute(1, 2, 0).numpy())
        axis.set_title(f"True: {names[answers[i]]}\nPred: {names[predictions[i]]}\np={confidence[i]:.1%}", fontsize=9)
    fig.suptitle(f"{title} · most confident errors" if len(indices) else f"{title} · no errors")
    save_figure(fig, output / "error_examples.png")


@torch.no_grad()
def plot_features(model, dataset, device, output):
    image, label = dataset[0]
    early = model[1](model[0](image.unsqueeze(0).to(device)))
    deeper = model[4](model[3](model[2](early)))
    fig, axes = plt.subplots(2, 9, figsize=(16, 4), layout="constrained")
    for row, features in enumerate([early, deeper]):
        axes[row, 0].imshow(dataset.X[0].permute(1, 2, 0).numpy())
        axes[row, 0].set_title("Input")
        for i in range(8):
            axes[row, i + 1].imshow(features[0, i].cpu().numpy(), cmap="gray")
            axes[row, i + 1].set_title(f"Layer {row + 1}, map {i + 1}", fontsize=8)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(f"CNN feature maps · {dataset.class_names[label]} · each map scaled independently")
    save_figure(fig, output / "feature_maps.png")


def plot_model(checkpoint_path, device, split="val"):
    checkpoint_path = Path(checkpoint_path)
    model, saved = load_model(checkpoint_path, device)
    loader = load_dataset(f"{split}_{saved['dataset_name']}", saved["batch_size"])
    output = PLOTS_DIR / checkpoint_path.stem / split
    title = f"{NAMES[saved.get('method', 'cnn')]} · {'validation' if split == 'val' else 'test'} (n={len(loader.dataset)})"
    answers, predictions, confidence, loss = predict(model, loader, device)
    names = loader.dataset.class_names
    precision, recall, f1, support = precision_recall_fscore_support(
        answers, predictions, labels=np.arange(len(names)), zero_division=0)
    matrix = confusion_matrix(answers, predictions, labels=np.arange(len(names)))
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "predictions.npz", answers=answers, predictions=predictions, confidence=confidence,
             class_names=names, loss=loss, accuracy=np.mean(answers == predictions), macro_f1=f1.mean(),
             precision=precision, recall=recall, f1=f1, support=support)
    history_path = RESULTS_DIR / saved.get("history_file", f"{checkpoint_path.stem}_history.npz")
    if history_path.exists():
        plot_history(history_path, output, saved["best_epoch"])
    else:
        print("No epoch history saved for this model.")
    plot_confusion(matrix, names, output, title)
    plot_class_metrics(precision, recall, f1, support, names, output, title)
    plot_errors(matrix, answers, predictions, confidence, loader.dataset, output, title)
    if saved.get("method", "cnn") == "cnn":
        plot_features(model, loader.dataset, device, output)
    print(f"Plots saved: {output}")
    return output


def plot_comparison(reference_path):
    _, reference = load_model(reference_path, torch.device("cpu"))
    keys = ["dataset_name", "batch_size", "seed", "device"]
    rows = []
    for path in sorted(WEIGHTS_DIR.glob("*.pt")):
        model, saved = load_model(path, torch.device("cpu"))
        if not all(key in saved for key in ("epochs", "training_seconds", "device")):
            print(f"Skipping incomplete training run: {path.name}")
            continue
        if all(saved.get(key) == reference.get(key) for key in keys):
            method = saved.get("method", "cnn")
            label = f"{NAMES[method]}\n{saved['epochs']} epochs, LR {saved['learning_rate']}"
            if saved.get("parent_checkpoint"):
                parent = torch.load(WEIGHTS_DIR / saved["parent_checkpoint"], map_location="cpu", weights_only=True)
                label += f"\nParent: {parent['epochs']} epochs, LR {parent['learning_rate']}"
            rows.append({"label": label,
                         "color": COLORS[method], "accuracy": saved["validation"]["accuracy"],
                         "f1": saved["validation"]["macro_f1"], "seconds": saved["training_seconds"],
                         "total": sum(p.numel() for p in model.parameters()),
                         "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    if not rows:
        raise ValueError("No completed runs match this dataset, batch size, seed, and device.")
    y = np.arange(len(rows))
    labels = [f"{i + 1}. {row['label']}" for i, row in enumerate(rows)]
    colors = [row["color"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, max(9, len(rows) * 1.3)), layout="constrained")
    for axis in axes.flat[:3]:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    for offset, key, label, hatch in [(-0.18, "accuracy", "Accuracy", "//"), (0.18, "f1", "Macro-F1", "")]:
        bars = axes[0, 0].barh(y + offset, [r[key] for r in rows], height=0.34, color=colors, hatch=hatch, label=label)
        axes[0, 0].bar_label(bars, fmt=lambda value: f"{value:.1%}", padding=3, fontsize=9)
    axes[0, 0].set(xlim=(0, 1.12), title="Validation quality", xlabel="Score")
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend()
    axes[0, 1].barh(y, [r["seconds"] for r in rows], color=colors)
    axes[0, 1].set(xlabel="Seconds (includes parent stage for fine-tuning)", title="Total training time", xlim=(0, None))
    for offset, key, label, hatch in [(-0.18, "total", "Total", "//"), (0.18, "trainable", "Trainable", "")]:
        axes[1, 0].barh(y + offset, [r[key] for r in rows], height=0.34, color=colors, hatch=hatch, label=label)
    axes[1, 0].set(xlabel="Parameters", title="Parameter counts", xlim=(0, None))
    axes[1, 0].legend()
    label_heights = np.array([row["f1"] for row in rows])
    previous = -1.0
    for i in np.argsort(label_heights):
        label_heights[i] = max(label_heights[i], previous + 0.025)
        previous = label_heights[i]
    for i, row in enumerate(rows):
        axes[1, 1].scatter(row["seconds"], row["f1"], color=row["color"], marker=["o", "s", "^"][i % 3])
        axes[1, 1].annotate(str(i + 1), (row["seconds"], row["f1"]),
                             xytext=(row["seconds"] + max(r["seconds"] for r in rows) * 0.025, label_heights[i]),
                             textcoords="data", fontsize=10,
                             arrowprops={"arrowstyle": "-", "color": "#777777"})
    axes[1, 1].set(xlabel="Total training time (s)", ylabel="Validation macro-F1", title="Quality vs time · numbers match the rows",
                   xlim=(0, max(r["seconds"] for r in rows) * 1.6 + 0.01),
                   ylim=(0, min(1.05, max(0.05, max(r["f1"] for r in rows) * 1.5))))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    with np.load(PROCESSED_DIR / f"train_{reference['dataset_name']}") as data:
        train_count = len(data["Y"])
    fig.suptitle(f"{reference['dataset_name']} · train n={train_count} · batch {reference['batch_size']} · "
                 f"seed {reference.get('seed')} · training device: {reference.get('device')}")
    output = PLOTS_DIR / (f"comparison_{Path(reference['dataset_name']).stem}"
                          f"_batch{reference['batch_size']}_{reference.get('device')}.png")
    save_figure(fig, output)
    print(f"Comparison saved: {output}")
    return output
