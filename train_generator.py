import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from embedding import encode
from generator_model import PaletteGeneratorModel


DATASET = "palette_and_text_train.csv"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_FILE = "palette_generator.pth"


def load_training_data():
    df = pd.read_csv(DATASET)

    texts = []

    for text in df["text_input"]:
        try:
            words = ast.literal_eval(text)
            text = " ".join(words)
        except (ValueError, SyntaxError):
            text = str(text)

        texts.append(text)

    print("Encoding text...")

    embeddings = encode(texts)

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


def train():
    embeddings, targets = load_training_data()

    print(f"Training examples: {len(embeddings)}")
    print(f"Embedding size: {embeddings.shape[1]}")
    print(f"Output size: {targets.shape[1]}")

    x = torch.tensor(
        embeddings,
        dtype=torch.float32
    )

    y = torch.tensor(
        targets,
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

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            predictions = model(batch_x)

            loss = loss_function(
                predictions,
                batch_y
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print()
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train()