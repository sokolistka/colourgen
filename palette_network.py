import torch
import torch.nn as nn
import torch.nn.functional as F


class PaletteNetwork(nn.Module):

    def __init__(self, input_size=384, color_count=5):
        super().__init__()
        self.color_count = color_count

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x, color_index):
        color_index = torch.as_tensor(
            color_index,
            device=x.device,
            dtype=torch.long
        )
        if color_index.ndim == 0:
            color_index = color_index.expand(x.shape[0])

        color_position = F.one_hot(
            color_index,
            num_classes=self.color_count
        ).to(dtype=x.dtype)
        return self.network(torch.cat((x, color_position), dim=-1))