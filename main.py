from src.data.data import PROCESSED_DIR, split_data
from src.data.dataset import load_data
from src.training import WEIGHTS_DIR, get_device, load_model, train_model
from src.plots import plot_comparison, plot_model


def choose_file(paths):
    if not paths:
        print("No matching files found. Prepare a dataset or train a model first.")
        return None
    for number, path in enumerate(paths, 1):
        print(f"{number}. {path.name}")
    number = int(input("File (0 to cancel): "))
    if number == 0:
        return None
    if not 1 <= number <= len(paths):
        raise ValueError(f"Choose a file number between 1 and {len(paths)}.")
    return paths[number - 1]


def run_option(choice, device, selected_name, batch_size):
    if choice == "1":
        value = input("Images (total before the 70/15/15 split; Enter for all): ").strip()
        counts = split_data(int(value) if value else None)
        print(f"Train: {counts[0]}, validation: {counts[1]}, test: {counts[2]}")

    elif choice in ("2", "4"):
        paths = (sorted(PROCESSED_DIR.glob("train_*img_seed*.npz")) if choice == "2"
                 else sorted(WEIGHTS_DIR.glob("*.pt")))
        path = choose_file(paths)
        if path is None:
            return selected_name, batch_size
        if choice == "2":
            name = path.name.removeprefix("train_")
            size = int(input(f"Batch size [{batch_size}]: ").strip() or batch_size)
        else:
            _, saved = load_model(path, device)
            name, size = saved["dataset_name"], saved["batch_size"]
        load_data(name, size)
        selected_name, batch_size = name, size
        print(f"Loaded: {path.name} | batch size: {batch_size}")

    elif choice == "3":
        print("1. Custom CNN\n2. Frozen MobileNetV2\n3. Partial fine-tuning")
        methods = ["cnn", "mobilenet_frozen", "mobilenet_finetune"]
        number = int(input("Model (0 to cancel): "))
        if number == 0:
            return selected_name, batch_size
        if not 1 <= number <= len(methods):
            raise ValueError("Choose model 1, 2, or 3.")
        method = methods[number - 1]
        fine_tuning = method == "mobilenet_finetune"
        parent_path = None
        if fine_tuning:
            parent_path = choose_file(sorted(WEIGHTS_DIR.glob("mobilenet_frozen_*.pt")))
            if parent_path is None:
                return selected_name, batch_size
            _, saved = load_model(parent_path, device)
            selected_name, batch_size = saved["dataset_name"], saved["batch_size"]
        if selected_name is None:
            raise ValueError("Load a dataset with option 2 before training.")
        train_loader, val_loader, _ = load_data(selected_name, batch_size)
        epochs = 5 if fine_tuning else 15
        learning_rate = 0.0001 if fine_tuning else 0.001
        epochs = int(input(f"Epochs [{epochs}]: ").strip() or epochs)
        learning_rate = float(input(f"Learning rate [{learning_rate}]: ").strip() or learning_rate)
        train_model(train_loader, val_loader, selected_name, epochs, learning_rate, device,
                    method=method, parent_path=parent_path)

    elif choice == "5":
        print("1. Model diagnostics (validation)\n2. Compare models (validation)\n3. Final test diagnostics")
        mode = input("Plots: ").strip()
        if mode not in {"1", "2", "3"}:
            raise ValueError("Choose plot mode 1, 2, or 3.")
        path = choose_file(sorted(WEIGHTS_DIR.glob("*.pt")))
        if path is None:
            return selected_name, batch_size
        if mode == "2":
            plot_comparison(path)
        else:
            plot_model(path, device, split="test" if mode == "3" else "val")
    else:
        raise ValueError("Choose an option from 0 to 5.")
    return selected_name, batch_size


def main():
    device = get_device()
    batch_size = 32
    selected_name = None
    print(f"Device: {device}")

    while True:
        print("\n1. Create dataset subset\n2. Load dataset\n3. Train model\n4. Load model\n5. Plot results\n0. Exit")
        choice = input("Option: ").strip()
        if choice == "0":
            break

        try:
            selected_name, batch_size = run_option(choice, device, selected_name, batch_size)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
