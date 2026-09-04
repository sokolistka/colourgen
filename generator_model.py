import torch
import torch.nn as nn

class PaletteGeneratorModel(nn.Module):


    def __init__(self, input_size=384, hidden_size=256, output_size=15):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, 128),
            nn.GELU(),

            nn.Linear(128, output_size)
        )

    def forward(self, x):

        return self.network(x)
