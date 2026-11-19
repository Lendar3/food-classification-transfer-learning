"""Convert locally downloaded Food-101 images to the project's NumPy format."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from src.data.data import PROJECT_ROOT, PROCESSED_DIR


def prepare_data(image_dir=None, output_path=None, overwrite=False):
    image_dir = Path(image_dir) if image_dir is not None else PROJECT_ROOT / "data/raw/food-101/images"
    output_path = Path(output_path) if output_path is not None else PROCESSED_DIR / "food101.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}. "
                                "Download and extract Food-101 first (see README.md).")
    class_names = sorted(folder.name for folder in image_dir.iterdir() if folder.is_dir())
    image_paths = [(path, label) for label, name in enumerate(class_names)
                   for path in sorted((image_dir / name).glob("*.jpg"))]
    if not image_paths:
        raise ValueError(f"No class-folder JPEG images found in {image_dir}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Disk-backed storage avoids keeping both a list and a full array of images in RAM.
    with TemporaryDirectory(prefix="food101-", dir=output_path.parent) as temporary:
        images = np.lib.format.open_memmap(
            Path(temporary) / "images.npy", mode="w+", dtype=np.uint8,
            shape=(len(image_paths), 3, 160, 160),
        )
        labels = np.empty(len(image_paths), dtype=np.int64)
        for index, (image_path, label) in enumerate(image_paths):
            with Image.open(image_path) as image:
                image = ImageOps.fit(image.convert("RGB"), (160, 160),
                                     method=Image.Resampling.BILINEAR)
                images[index] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
            labels[index] = label
            if (index + 1) % 1000 == 0:
                print(f"Prepared {index + 1}/{len(image_paths)} images", flush=True)
        images.flush()
        temporary_output = Path(temporary) / "food101.npz"
        np.savez(temporary_output, X=images, Y=labels, class_names=np.array(class_names))
        del images
        temporary_output.replace(output_path)
    print(f"Saved {len(image_paths)} images, {len(class_names)} classes: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, help="Directory containing class folders of JPEGs")
    parser.add_argument("--output", type=Path, help="Destination .npz file")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared dataset")
    args = parser.parse_args()
    try:
        prepare_data(args.image_dir, args.output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
