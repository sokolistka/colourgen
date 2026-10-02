"""Run the embedding-only palette experiment.

This experiment tests a non-autoregressive alternative to the sequential palette
network. It maps each 384-value text embedding directly to all 15 palette
channels in one pass, so later colors do not receive earlier colors as context.
The five RGB-derived OpenCV Lab colors are trained in their existing palette
order using targets scaled by 255 and clamped to [1e-6, 1 - 1e-6].

It uses palette_and_text_train_balanced.csv for training and
palette_and_text_val_normalized.csv for validation, mean CIEDE2000 loss, Adam
with learning rate 0.001, batches of 32, and validation-based early stopping.
The best weights are saved separately as palette_embedding_experiment.pth.

In the measured comparison run, this model reached validation loss 24.8225 at
epoch 10 (training loss 19.2951); the autoregressive model reached validation
loss 25.3474 (training loss 19.2903). This is a diagnostic comparison, not a
claim that either model generalizes well. Exact results can vary by runtime.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from color_loss import CIEDE2000Loss
from data_loader import load_dataset
from target_scaling import scale_targets
from train_network import prepare_data

TRAIN_DATASET = "palette_and_text_train_balanced.csv"
VALIDATION_DATASET = "palette_and_text_val_normalized.csv"
MODEL_FILE = "palette_embedding_experiment.pth"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_INTERVAL = 10
EARLY_STOPPING_PATIENCE = 5
METRIC_TOLERANCE = 10.0


class EmbeddingOnlyPaletteNetwork(nn.Module):
    def __init__(self, input_size=384):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 15),
            nn.Sigmoid(),
        )

    def forward(self, embeddings):
        return self.network(embeddings).reshape(-1, 5, 3)


def load_tensor_dataset(filename):
    dataframe = load_dataset(filename)
    embeddings, palettes = prepare_data(dataframe)
    scaled_palettes = np.clip(
        scale_targets(palettes),
        1e-6,
        1.0 - 1e-6,
    ).reshape(-1, 5, 3)
    return TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(scaled_palettes, dtype=torch.float32),
    )


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    total_values = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            loss = loss_function(
                predictions.reshape(-1, 3),
                batch_y.reshape(-1, 3),
            )

            total_loss += loss.item() * batch_x.shape[0]
            total_rows += batch_x.shape[0]
            errors = torch.abs(predictions - batch_y) * 255.0
            total_correct += (errors <= METRIC_TOLERANCE).sum().item()
            total_values += errors.numel()

    return total_loss / total_rows, total_correct / total_values


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_tensor_dataset(TRAIN_DATASET)
    validation_dataset = load_tensor_dataset(VALIDATION_DATASET)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_evaluation_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EmbeddingOnlyPaletteNetwork(
        input_size=train_dataset.tensors[0].shape[1]
    ).to(device)
    loss_function = CIEDE2000Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    print(f"Training rows: {len(train_dataset)}")
    print(f"Validation rows: {len(validation_dataset)}")
    print(f"Device: {device}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(
                predictions.reshape(-1, 3),
                batch_y.reshape(-1, 3),
            )
            loss.backward()
            optimizer.step()

        if epoch % VALIDATION_INTERVAL != 0:
            continue

        train_loss, train_accuracy = evaluate(
            model,
            train_evaluation_loader,
            loss_function,
            device,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )
        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {validation_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2%} "
            f"Validation Accuracy: {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            best_model_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            torch.save(best_model_state, MODEL_FILE)
            print(f"  Saved best experiment to {MODEL_FILE}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "Early stopping after "
                    f"{EARLY_STOPPING_PATIENCE} validation checks without improvement."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    train_loss, train_accuracy = evaluate(
        model,
        train_evaluation_loader,
        loss_function,
        device,
    )
    validation_loss, validation_accuracy = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
    )
    print(
        f"Best experiment: train loss={train_loss:.4f}, "
        f"validation loss={validation_loss:.4f}, "
        f"train accuracy={train_accuracy:.2%}, "
        f"validation accuracy={validation_accuracy:.2%}"
    )
    print(f"Best experiment checkpoint: {MODEL_FILE}")


if __name__ == "__main__":
    train()
