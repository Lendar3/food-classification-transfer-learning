"""Small regression checks; no Food-101 download or pretrained weights required."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from main import choose_file, run_option
from src.data.data import split_data
from src.data.dataset import load_dataset
from src.data.prepare_data import prepare_data
from src.training import create_cnn, create_model, load_model, train_model

torch.set_num_threads(1)


class ProjectTests(unittest.TestCase):
    def test_cnn_accepts_real_image_size_and_saved_classifier_keys(self):
        model = create_cnn()
        self.assertEqual(model(torch.zeros(2, 3, 160, 160)).shape, (2, 101))
        self.assertEqual(model.state_dict()["11.weight"].shape, (101, 128))

    def test_transfer_trainable_parameters_and_batchnorm(self):
        frozen = create_model("mobilenet_frozen", pretrained=False)
        self.assertFalse(any(p.requires_grad for p in frozen.features.parameters()))
        self.assertEqual(sum(p.numel() for p in frozen.parameters() if p.requires_grad), 129381)
        fine = create_model("mobilenet_finetune", pretrained=False)
        self.assertFalse(any(p.requires_grad for p in fine.features[:-3].parameters()))
        self.assertTrue(any(p.requires_grad for p in fine.features[-3:].parameters()))
        self.assertTrue(all(not layer.training for layer in fine.features.modules()
                            if isinstance(layer, torch.nn.BatchNorm2d)))
        self.assertEqual(fine(torch.zeros(1, 3, 160, 160)).shape, (1, 101))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            create_model("typo", pretrained=False)

    def test_menu_handles_empty_choices_and_invalid_indices(self):
        self.assertIsNone(choose_file([]))
        with patch("builtins.input", return_value="-1"):
            with self.assertRaises(ValueError):
                choose_file([Path("example.pt")])
        with patch("builtins.input", return_value="1"):
            with self.assertRaisesRegex(ValueError, "Load a dataset"):
                run_option("3", torch.device("cpu"), None, 32)

    def test_prepare_tiny_dataset_and_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["bread", "rice"]:
                (root / "images" / name).mkdir(parents=True)
                Image.new("RGB", (12, 18), (30, 60, 90)).save(root / "images" / name / "1.jpg")
            output = prepare_data(root / "images", root / "processed" / "food101.npz")
            with np.load(output) as data:
                self.assertEqual(data["X"].shape, (2, 3, 160, 160))
                self.assertEqual(data["class_names"].tolist(), ["bread", "rice"])
                self.assertEqual(data["Y"].tolist(), [0, 1])
            with self.assertRaises(FileExistsError):
                prepare_data(root / "images", output)

    def test_split_is_reproducible_disjoint_and_loader_normalizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = np.zeros((40, 3, 8, 8), dtype=np.uint8)
            images[:, 0, 0, 0] = np.arange(40)
            np.savez(root / "food101.npz", X=images, Y=np.repeat([0, 1], 20),
                     class_names=np.array(["bread", "rice"]))
            with patch("src.data.data.PROCESSED_DIR", root), patch("src.data.dataset.PROCESSED_DIR", root):
                self.assertEqual(split_data(30), (21, 4, 5))
                arrays = [np.load(root / f"{name}_30img_seed12.npz") for name in ["train", "val", "test"]]
                try:
                    ids = [set(a["X"][:, 0, 0, 0].tolist()) for a in arrays]
                    original_train = arrays[0]["X"].copy()
                    self.assertEqual(len(set.union(*ids)), 30)
                    self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
                finally:
                    for array in arrays:
                        array.close()
                split_data(30)
                with np.load(root / "train_30img_seed12.npz") as data:
                    np.testing.assert_array_equal(original_train, data["X"])
                batch, labels = next(iter(load_dataset("train_30img_seed12.npz", 4)))
                self.assertEqual(batch.shape, (4, 3, 8, 8))
                self.assertEqual(labels.dtype, torch.int64)
                self.assertAlmostEqual(batch[0, 1, 0, 0].item(), -0.456 / 0.224, places=5)
                with self.assertRaises(ValueError):
                    split_data(2)

    def test_training_roundtrip_and_no_silent_checkpoint_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / "train_tiny.npz", X=np.zeros((4, 3, 16, 16), dtype=np.uint8),
                     Y=np.array([0, 1, 0, 1]), class_names=np.array(["bread", "rice"]))
            with patch("src.data.dataset.PROCESSED_DIR", root), \
                 patch("src.training.WEIGHTS_DIR", root / "weights"), \
                 patch("src.training.RESULTS_DIR", root / "results"):
                loader = load_dataset("train_tiny.npz", 2, shuffle=True)
                model, path = train_model(loader, loader, "tiny.npz", 1, 0.001, torch.device("cpu"))
                restored, saved = load_model(path, torch.device("cpu"))
                batch, _ = next(iter(loader))
                torch.testing.assert_close(model(batch), restored(batch))
                self.assertTrue((root / "results" / saved["history_file"]).is_file())
                self.assertEqual(saved["train_images"], 4)
                with self.assertRaises(FileExistsError):
                    train_model(loader, loader, "tiny.npz", 1, 0.001, torch.device("cpu"))
                with self.assertRaises(ValueError):
                    train_model(loader, loader, "tiny.npz", 0, 0.001, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
