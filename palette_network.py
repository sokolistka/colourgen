import torch
import torch.nn as nn


class PaletteNetwork(nn.Module):

    def __init__(self, input_size=384):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 15)
        )

    def forward(self, x):

        return self.network(x)