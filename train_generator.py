import ast
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from embedding import encode
from generator_model import PaletteGeneratorModel


DATASET = "palette_and_text_train.csv"
VALIDATION_DATASET = "palette_and_text_val.csv"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
METRIC_TOLERANCE = 10.0

MODEL_FILE = "palette_generator.pth"


def load_training_data(dataset_file, log_progress=True):
    df = pd.read_csv(dataset_file)

    texts = []

    for text in df["text_input"]:
        try:
            words = ast.literal_eval(text)
            text = " ".join(words)
        except (ValueError, SyntaxError):
            text = str(text)

        texts.append(text)

    if log_progress:
        print(f"Encoding {dataset_file}...")

    embeddings = encode(texts)

    if log_progress:
        print("Text encoded.")

    targets = []

    for palette in df["palette_lab_reorder"]:
        palette = ast.literal_eval(palette)

        palette = np.array(
            palette,
            dtype=np.float32
        ).reshape(-1)

        targets.append(palette)

    targets = np.array(
        targets,
        dtype=np.float32
    )

    return embeddings, targets


def calculate_metrics(predictions, targets):
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2
    correct = absolute_errors <= METRIC_TOLERANCE

    true_positive = np.sum(correct)
    predicted_positive = correct.size
    actual_positive = correct.size
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    f1_score = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(np.mean(correct)),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors)))
    }


def get_system_metrics():
    process = psutil.Process(os.getpid())
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
        "process_ram_mb": process.memory_info().rss / (1024 ** 2)
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        metrics["gpu_memory_percent"] = (
            torch.cuda.memory_allocated(device) /
            torch.cuda.get_device_properties(device).total_memory * 100
        )
        metrics["gpu_memory_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        metrics["gpu_utilization"] = "unavailable via PyTorch"
    else:
        metrics["gpu_memory_percent"] = 0.0
        metrics["gpu_memory_mb"] = 0.0
        metrics["gpu_utilization"] = "N/A"

    return metrics


def print_histogram(name, values):
    histogram, edges = np.histogram(values, bins=8)
    buckets = ", ".join(
        f"[{edges[index]:.2f}, {edges[index + 1]:.2f}): {histogram[index]}"
        for index in range(len(histogram))
    )
    return name, buckets


def print_table(title, headers, rows):
    print(f"\n{title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")


def train():
    embeddings, targets = load_training_data(DATASET)
    validation_embeddings, validation_targets = load_training_data(
        VALIDATION_DATASET
    )

    print_table(
        "Training parameters",
        ["Parameter", "Value"],
        [
            ["Dataset", DATASET],
            ["Validation dataset", VALIDATION_DATASET],
            ["Batch size", BATCH_SIZE],
            ["Epochs", EPOCHS],
            ["Learning rate", LEARNING_RATE],
            ["Metric tolerance", f"+/-{METRIC_TOLERANCE} LAB units"],
            ["Optimizer", "Adam"],
            ["Loss function", "MSELoss"],
            ["Shuffle", True],
            ["Model output", MODEL_FILE]
        ]
    )
    print_table(
        "Dataset information",
        ["Dataset", "Palettes", "Embedding size", "Output size"],
        [
            ["Training", len(embeddings), embeddings.shape[1], targets.shape[1]],
            ["Validation", len(validation_embeddings),
             validation_embeddings.shape[1], validation_targets.shape[1]]
        ]
    )

    x = torch.tensor(
        embeddings,
        dtype=torch.float32
    )

    y = torch.tensor(
        targets,
        dtype=torch.float32
    )

    validation_x = torch.tensor(
        validation_embeddings,
        dtype=torch.float32
    )

    validation_y = torch.tensor(
        validation_targets,
        dtype=torch.float32
    )

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = PaletteGeneratorModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.MSELoss()
    activation_outputs = {}

    def capture_activation(name):
        def hook(_, __, output):
            activation_outputs[name] = output.detach().cpu().numpy()

        return hook

    activation_hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.GELU):
            activation_hooks.append(
                layer.register_forward_hook(capture_activation(name))
            )

    epoch_start = time.perf_counter()
    epoch_rows = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        gradient_norm_total = 0.0
        step_start = time.perf_counter()

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            predictions = model(batch_x)

            loss = loss_function(
                predictions,
                batch_y
            )

            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")
            )
            gradient_norm_total += gradient_norm.item()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_x).numpy()
            validation_loss = loss_function(
                torch.tensor(validation_predictions),
                validation_y
            ).item()
            training_predictions = model(x).numpy()
        model.train()

        training_metrics = calculate_metrics(training_predictions, targets)
        validation_metrics = calculate_metrics(
            validation_predictions,
            validation_targets
        )
        epoch_time = time.perf_counter() - epoch_start
        step_time = (time.perf_counter() - step_start) / len(loader)
        system_metrics = get_system_metrics()

        epoch_rows.append([
            epoch + 1,
            f"{average_loss:.4f}",
            f"{validation_loss:.4f}",
            f"{training_metrics['accuracy']:.4f}",
            f"{validation_metrics['accuracy']:.4f}",
            f"{validation_metrics['f1_score']:.4f}",
            f"{validation_metrics['precision']:.4f}",
            f"{validation_metrics['recall']:.4f}",
            f"{validation_metrics['mae']:.4f}",
            f"{validation_metrics['rmse']:.4f}",
            f"{gradient_norm_total / len(loader):.4f}",
            f"{epoch_time:.2f}",
            f"{step_time:.4f}",
            f"{system_metrics['cpu_percent']:.1f}%",
            f"{system_metrics['ram_percent']:.1f}%",
            f"{system_metrics['process_ram_mb']:.1f}",
            f"{system_metrics['gpu_memory_mb']:.1f}",
            f"{system_metrics['gpu_memory_percent']:.1f}%",
            system_metrics["gpu_utilization"]
        ])
        epoch_start = time.perf_counter()

    print_table(
        "Loss and accuracy by epoch",
        ["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"],
        [row[:5] for row in epoch_rows]
    )
    print_table(
        "Validation quality by epoch",
        ["Epoch", "F1", "Precision", "Recall", "MAE", "RMSE"],
        [[row[0], *row[5:10]] for row in epoch_rows]
    )
    print_table(
        "Optimization performance by epoch",
        ["Epoch", "Grad Norm", "Time/Epoch (s)", "Time/Batch (s)"],
        [[row[0], *row[10:13]] for row in epoch_rows]
    )
    print_table(
        "Hardware and system metrics by epoch",
        ["Epoch", "CPU", "RAM", "Process RAM (MB)", "VRAM (MB)", "VRAM", "GPU"],
        [[row[0], *row[13:]] for row in epoch_rows]
    )

    for hook in activation_hooks:
        hook.remove()

    distribution_rows = []
    for name, parameter in model.named_parameters():
        if name.endswith("weight"):
            distribution_name, histogram = print_histogram(
                name,
                parameter.detach().cpu().numpy().ravel()
            )
            distribution_rows.append(["Weight", distribution_name, histogram])

    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            distribution_name, histogram = print_histogram(
                name,
                parameter.detach().cpu().numpy().ravel()
            )
            distribution_rows.append(["Bias", distribution_name, histogram])

    for name, output in activation_outputs.items():
        distribution_name, histogram = print_histogram(name, output.ravel())
        distribution_rows.append(["Activation", distribution_name, histogram])

    print_table(
        "Weight, bias, and activation distributions (histograms)",
        ["Type", "Layer", "Bins"],
        distribution_rows
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print()
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train()