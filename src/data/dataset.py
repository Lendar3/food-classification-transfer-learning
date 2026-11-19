import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.data import PROCESSED_DIR, SEED


class FoodDataset(Dataset):
    def __init__(self, file_name):
        with np.load(PROCESSED_DIR / file_name) as data:
            self.X = torch.from_numpy(data["X"])
            self.Y = torch.from_numpy(data["Y"])
            self.class_names = data["class_names"]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        image = self.X[index].float() / 255
        image = (image - self.mean) / self.std
        return image, self.Y[index]


def load_dataset(file_name, batch_size=32, shuffle=False):
    if batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    dataset = FoodDataset(file_name)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
    )


def load_data(dataset_name, batch_size):
    train = load_dataset(f"train_{dataset_name}", batch_size, shuffle=True)
    validation = load_dataset(f"val_{dataset_name}", batch_size)
    test = load_dataset(f"test_{dataset_name}", batch_size)
    return train, validation, test
