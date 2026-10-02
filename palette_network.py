import torch
import torch.nn as nn
import torch.nn.functional as F


class PaletteNetwork(nn.Module):

    def __init__(self, input_size=384, color_count=5):
        super().__init__()
        self.color_count = color_count

        self.network = nn.Sequential(
            nn.Linear(input_size + color_count * 4, 256),
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

    def forward(self, x, previous_colors, color_index):
        previous_colors = torch.as_tensor(
            previous_colors,
            device=x.device,
            dtype=x.dtype
        )
        if previous_colors.shape != (x.shape[0], self.color_count, 3):
            raise ValueError(
                f"previous_colors must have shape "
                f"({x.shape[0]}, {self.color_count}, 3)"
            )

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
        return self.network(
            torch.cat((x, previous_colors.flatten(start_dim=1), color_position), dim=-1)
        )

    def generate_palette(self, x, targets=None):
        target_colors = None
        if targets is not None:
            target_colors = torch.as_tensor(
                targets,
                device=x.device,
                dtype=x.dtype
            )
            if target_colors.ndim == 2 and target_colors.shape == (
                x.shape[0], self.color_count * 3
            ):
                target_colors = target_colors.reshape(-1, self.color_count, 3)
            if target_colors.shape != (x.shape[0], self.color_count, 3):
                raise ValueError(
                    f"targets must have shape "
                    f"({x.shape[0]}, {self.color_count}, 3) or "
                    f"({x.shape[0]}, {self.color_count * 3})"
                )

        generated_colors = []
        for color_index in range(self.color_count):
            if target_colors is None:
                history = generated_colors
            else:
                history = [target_colors[:, :color_index, :]]
            padding = x.new_zeros(
                (x.shape[0], self.color_count - color_index, 3)
            )
            previous_colors = torch.cat((*history, padding), dim=1)
            generated_colors.append(
                self(x, previous_colors, color_index).unsqueeze(dim=1)
            )

        return torch.cat(generated_colors, dim=1)