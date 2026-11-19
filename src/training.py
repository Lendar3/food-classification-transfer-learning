import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.data.data import PROJECT_ROOT, SEED

WEIGHTS_DIR = PROJECT_ROOT / "data" / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_cnn(num_classes=101):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def create_model(method, num_classes=101, pretrained=True):
    if method not in {"cnn", "mobilenet_frozen", "mobilenet_finetune"}:
        raise ValueError(f"Unknown model method: {method}")
    if method == "cnn":
        return create_cnn(num_classes)

    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Linear(1280, num_classes)
    if method == "mobilenet_finetune":
        for parameter in model.features[-3:].parameters():
            parameter.requires_grad = True
        for layer in model.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                for parameter in layer.parameters():
                    parameter.requires_grad = False
    model.features.eval()
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    answers = []
    predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)
        total_loss += loss.item() * len(labels)
        answers.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    correct = sum(answer == prediction for answer, prediction in zip(answers, predictions))
    return {
        "loss": total_loss / len(answers),
        "accuracy": correct / len(answers),
        "macro_f1": float(f1_score(
            answers, predictions, average="macro", zero_division=0,
            labels=list(range(len(loader.dataset.class_names))),
        )),
    }


def train_model(train_loader, val_loader, dataset_name, epochs, learning_rate, device,
                method="cnn", parent_path=None):
    if epochs < 1:
        raise ValueError("Epochs must be a positive integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError("Training and validation datasets must not be empty.")
    torch.manual_seed(SEED)
    train_loader.generator.manual_seed(SEED)
    class_names = train_loader.dataset.class_names.tolist()
    fine_tuning = method == "mobilenet_finetune"
    if fine_tuning and parent_path is None:
        raise ValueError("Fine-tuning requires a frozen MobileNetV2 checkpoint.")
    model = create_model(method, len(class_names), pretrained=not fine_tuning).to(device)
    parent = None
    if fine_tuning:
        parent = torch.load(parent_path, map_location=device, weights_only=True)
        if parent.get("method") != "mobilenet_frozen":
            raise ValueError("The parent must be a frozen MobileNetV2 checkpoint.")
        if parent["dataset_name"] != dataset_name or parent["class_names"] != class_names:
            raise ValueError("The parent checkpoint uses a different dataset or class order.")
        if "training_seconds" not in parent:
            raise ValueError("The parent training run is incomplete; choose a completed checkpoint.")
        model.load_state_dict(parent["model_state"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if method == "mobilenet_finetune":
        backbone_parameters = [p for p in model.features.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": backbone_parameters, "lr": learning_rate * 0.1},
            {"params": model.classifier.parameters(), "lr": learning_rate},
        ])
    else:
        optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    batch_size = train_loader.batch_size

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = (f"{method}_{Path(dataset_name).stem}_batch{batch_size}"
                f"_epochs{epochs}_lr{learning_rate}")
    if parent is not None:
        run_name = f"finetune_{Path(parent_path).stem}_epochs{epochs}_lr{learning_rate}"
    checkpoint_path = WEIGHTS_DIR / f"{run_name}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path.name}. "
                              "Move it before repeating the same configuration.")
    best_score = (-1.0, float("-inf"))
    history = {name: [] for name in ["epoch", "train_loss", "val_loss", "train_accuracy",
                                   "val_accuracy", "train_macro_f1", "val_macro_f1", "seconds"]}

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        if method != "cnn":
            model.features.eval()
        total_loss = 0.0
        answers, predictions = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            answers.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader.dataset),
            "val_loss": validation["loss"],
            "train_accuracy": np.mean(np.array(answers) == predictions),
            "val_accuracy": validation["accuracy"],
            "train_macro_f1": f1_score(answers, predictions, average="macro", zero_division=0,
                                      labels=list(range(len(class_names)))),
            "val_macro_f1": validation["macro_f1"],
            "seconds": time.perf_counter() - start,
        }
        for name, value in row.items():
            history[name].append(value)
        print(f"Epoch {epoch}/{epochs} | loss: {total_loss / len(train_loader.dataset):.4f} | "
              f"val loss: {validation['loss']:.4f} | accuracy: {validation['accuracy']:.4f} | "
              f"F1: {validation['macro_f1']:.4f}", flush=True)

        score = (validation["macro_f1"], -validation["loss"])
        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "method": method,
                "pretrained_weights": (parent["pretrained_weights"] if parent is not None
                                       else "IMAGENET1K_V2" if method == "mobilenet_frozen" else None),
                "class_names": class_names,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "seed": SEED,
                "learning_rate": learning_rate,
                "backbone_learning_rate": learning_rate * 0.1 if parent is not None else None,
                "parent_checkpoint": Path(parent_path).name if parent is not None else None,
                "unfrozen_blocks": [f"features.{i}" for i in range(len(model.features) - 3, len(model.features))]
                                   if fine_tuning else [],
                "best_epoch": epoch,
                "validation": validation,
            }, checkpoint_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    stage_seconds = time.perf_counter() - start
    training_seconds = stage_seconds + (parent["training_seconds"] if parent is not None else 0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint["epochs"] = epochs
    checkpoint["training_seconds"] = training_seconds
    checkpoint["stage_seconds"] = stage_seconds
    checkpoint["device"] = str(device)
    checkpoint["trainable_parameters"] = sum(p.numel() for p in trainable)
    checkpoint["total_parameters"] = sum(p.numel() for p in model.parameters())
    checkpoint["train_images"] = len(train_loader.dataset)
    checkpoint["val_images"] = len(val_loader.dataset)
    checkpoint["history_file"] = f"{run_name}_history.npz"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_DIR / checkpoint["history_file"], **history)
    torch.save(checkpoint, checkpoint_path)

    print(f"Saved: {checkpoint_path.name}")
    return model, checkpoint_path


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    method = checkpoint.get("method", "cnn")
    model = create_model(method, len(checkpoint["class_names"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
