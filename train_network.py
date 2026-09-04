import ast

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_dataset
from embedding import encode
from palette_network import PaletteNetwork


def prepare_data(df):

    descriptions = []

    palettes = []

    for _, row in df.iterrows():

        text = row["text_input"]

        try:
            words = ast.literal_eval(text)
            text = " ".join(words)
        except (ValueError, SyntaxError):
            text = str(text)

        descriptions.append(text)

        palette = ast.literal_eval(
            row["palette_lab_reorder"]
        )

        palettes.append(
            np.array(
                palette,
                dtype=np.float32
            ).reshape(-1)
        )

    embeddings = encode(descriptions)

    palettes = np.array(
        palettes,
        dtype=np.float32
    )

    return embeddings, palettes


def train():

    dataset_file = "palette_and_text_train.csv"
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    optimizer_name = "Adam"
    loss_name = "MSELoss"
    shuffle = True
    palette_normalization = "/ 255.0"

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

    # Normalize LAB values to approximately 0-1
    y = torch.tensor(
        palettes / 255.0,
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

    torch.save(
        model.state_dict(),
        "palette_network.pth"
    )

    print(
        "\nModel saved as palette_network.pth"
    )


if __name__ == "__main__":
    train()