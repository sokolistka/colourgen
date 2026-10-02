import ast
import json

import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from color_loss import CIEDE2000Loss
from data_loader import load_dataset
from palette_network import PaletteNetwork
from target_scaling import inverse_scale_targets, scale_targets

METRIC_TOLERANCE = 10.0
VALIDATION_INTERVAL = 10
EARLY_STOPPING_PATIENCE = 5


def prepare_data(df):

    palettes = []

    if "text_embedding" not in df.columns:
        raise ValueError(
            "Training data must be normalized and contain text_embedding"
        )

    embeddings = np.array(
        [json.loads(value) for value in df["text_embedding"]],
        dtype=np.float32
    )

    for palette in df["palette_lab_reorder"]:
        palette = ast.literal_eval(palette)

        palettes.append(
            np.array(
                palette,
                dtype=np.float32
            ).reshape(-1)
        )

    palettes = np.array(
        palettes,
        dtype=np.float32
    )

    return embeddings, palettes


def evaluate(model, loader, loss_function, teacher_forcing=False):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    total_values = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            predicted_palette = model.generate_palette(
                batch_x,
                targets=batch_y if teacher_forcing else None
            )
            target_palette = batch_y.reshape(-1, model.color_count, 3)
            color_losses = [
                loss_function(predicted_palette[:, color_index], target_palette[:, color_index])
                for color_index in range(model.color_count)
            ]

            batch_loss = torch.stack(color_losses).mean()
            total_loss += batch_loss.item() * batch_x.shape[0]
            total_examples += batch_x.shape[0]

            errors = torch.abs(predicted_palette - target_palette) * 255.0
            total_correct += (errors <= METRIC_TOLERANCE).sum().item()
            total_values += errors.numel()

    return total_loss / total_examples, total_correct / total_values


def train():

    dataset_file = "palette_and_text_train_balanced.csv"
    validation_dataset_file = "palette_and_text_val_normalized.csv"
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    optimizer_name = "Adam"
    loss_name = "Mean CIEDE2000 across 5 colors"
    shuffle = True
    palette_normalization = "LAB / 255.0 (shared train + validation transform)"

    df = load_dataset(
        dataset_file
    )
    validation_df = load_dataset(
        validation_dataset_file
    )

    print("Training parameters:")
    print(f"  Dataset: {dataset_file}")
    print(f"  Validation dataset: {validation_dataset_file}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Loss function: {loss_name}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Palette normalization: {palette_normalization}")

    print(f"Loaded {len(df)} training examples")

    embeddings, palettes = prepare_data(df)
    validation_embeddings, validation_palettes = prepare_data(validation_df)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Palette shape: {palettes.shape}"
    )

    x = torch.tensor(
        embeddings,
        dtype=torch.float32
    )

    # Normalize LAB values to approximately 0-1.
    # The same transform must be used for every target split.
    y = torch.tensor(
        np.clip(scale_targets(palettes), 1e-6, 1.0 - 1e-6),
        dtype=torch.float32
    )
    validation_x = torch.tensor(
        validation_embeddings,
        dtype=torch.float32
    )
    validation_y = torch.tensor(
        np.clip(scale_targets(validation_palettes), 1e-6, 1.0 - 1e-6),
        dtype=torch.float32
    )

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    train_evaluation_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    validation_dataset = TensorDataset(validation_x, validation_y)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = PaletteNetwork(
        input_size=embeddings.shape[1]
    )

    loss_function = CIEDE2000Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):

        model.train()

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            predicted_palette = model.generate_palette(batch_x, targets=batch_y)
            target_palette = batch_y.reshape(-1, model.color_count, 3)
            color_losses = [
                loss_function(predicted_palette[:, color_index], target_palette[:, color_index])
                for color_index in range(model.color_count)
            ]
            loss = torch.stack(color_losses).mean()

            loss.backward()

            optimizer.step()

        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            train_loss, train_accuracy = evaluate(
                model,
                train_evaluation_loader,
                loss_function,
                teacher_forcing=True
            )
            validation_loss, _ = evaluate(
                model,
                validation_loader,
                loss_function
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train Loss (teacher-forced): {train_loss:.6f} "
                f"Validation Loss (free-running): {validation_loss:.6f} "
                f"Train Accuracy (teacher-forced, ±{METRIC_TOLERANCE:g} Lab units): "
                f"{train_accuracy:.2%}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
                best_model_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                torch.save(best_model_state, "palette_network.pth")
                print("  Saved new best model to palette_network.pth")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping after {EARLY_STOPPING_PATIENCE} "
                        "validation checks without improvement."
                    )
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Reconstruct original LAB values only for reporting; training is done on normalized targets.
    with torch.no_grad():
        sample_colors = model.generate_palette(x[:1]).cpu().numpy()
        sample_prediction = inverse_scale_targets(sample_colors)
        print(
            f"Example prediction (reconstructed LAB): "
            f"{sample_prediction[0].reshape(-1).round(2).tolist()}"
        )

    print(
        f"\nBest model saved as palette_network.pth "
        f"(validation loss: {best_validation_loss:.6f})"
    )


if __name__ == "__main__":
    train()