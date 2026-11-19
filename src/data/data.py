from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 12


def split_data(image_count=None):
    with np.load(PROCESSED_DIR / "food101.npz") as data:
        Y = data["Y"]
        class_names = data["class_names"]
        if image_count is None:
            image_count = len(Y)
        if image_count > len(Y) or image_count < 1:
            raise ValueError(f"Image count must be between 1 and {len(Y)}.")

        train_count = image_count * 70 // 100
        val_count = image_count * 15 // 100
        test_count = image_count - train_count - val_count
        num_classes = len(class_names)
        if min(train_count, val_count, test_count) < num_classes:
            raise ValueError(f"Each split needs at least {num_classes} images for stratification. "
                             "Use at least 700 total images for Food-101.")

        indices = resample(
            np.arange(len(Y)), n_samples=image_count,
            replace=False, stratify=Y, random_state=SEED,
        )
        train_indices, remaining_indices = train_test_split(
            indices, train_size=train_count,
            stratify=Y[indices], random_state=SEED,
        )
        val_indices, test_indices = train_test_split(
            remaining_indices, train_size=val_count,
            stratify=Y[remaining_indices], random_state=SEED,
        )
        X = data["X"]

    for name, indices in [("train", train_indices),
                          ("val", val_indices), ("test", test_indices)]:
        np.savez(
            PROCESSED_DIR / f"{name}_{image_count}img_seed{SEED}.npz",
            X=X[indices], Y=Y[indices], class_names=class_names,
        )

    return train_count, val_count, test_count
