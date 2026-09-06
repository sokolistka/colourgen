import ast
import json

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_dataset
from palette_network import PaletteNetwork
from target_scaling import inverse_scale_targets, scale_targets


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


def train():

    dataset_file = "palette_and_text_train_balanced.csv"
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    optimizer_name = "Adam"
    loss_name = "MSELoss"
    shuffle = True
    palette_normalization = "LAB / 255.0 (shared train + validation transform)"

    df = load_dataset(
        dataset_file
    )

    print("Training parameters:")
    print(f"  Dataset: {dataset_file}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Loss function: {loss_name}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Palette normalization: {palette_normalization}")

    print(f"Loaded {len(df)} training examples")

    embeddings, palettes = prepare_data(df)

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
        scale_targets(palettes),
        dtype=torch.float32
    )

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    model = PaletteNetwork(
        input_size=embeddings.shape[1]
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):

        total_loss = 0.0

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            prediction = model(batch_x)

            loss = loss_function(
                prediction,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Loss: {average_loss:.6f}"
            )

    # Reconstruct original LAB values only for reporting; training is done on normalized targets.
    with torch.no_grad():
        sample_prediction = inverse_scale_targets(model(x[:1]).cpu().numpy())
        print(
            f"Example prediction (reconstructed LAB): {sample_prediction[0].round(2).tolist()}"
        )

    torch.save(
        model.state_dict(),
        "palette_network.pth"
    )

    print(
        "\nModel saved as palette_network.pth"
    )


if __name__ == "__main__":
    train()