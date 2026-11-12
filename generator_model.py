import torch
import torch.nn as nn


class PaletteGeneratorModel(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256):
        super().__init__()
        self.color_projection = nn.Linear(3, hidden_dim)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sequence_processor = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )
        self.color_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, text_emb, targets=None):
        batch_size = text_emb.size(0)
        current_color = torch.zeros(
            batch_size,
            3,
            device=text_emb.device,
            dtype=text_emb.dtype
        )
        text_context = self.text_projection(text_emb)
        color_tokens = []
        predictions = []

        for step in range(5):
            input_color = current_color
            if self.training and targets is not None:
                input_color = input_color + torch.randn_like(input_color) * 0.1

            color_token = self.color_projection(input_color) + text_context
            color_tokens.append(color_token)
            token_sequence = torch.stack(color_tokens, dim=1)
            sequence_length = token_sequence.size(1)
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=text_emb.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )
            processed_tokens = self.sequence_processor(
                token_sequence,
                mask=causal_mask
            )
            next_color = self.color_regressor(processed_tokens[:, -1])
            predictions.append(next_color)

            if self.training and targets is not None:
                current_color = targets[:, step * 3:(step + 1) * 3]
            else:
                current_color = next_color

        return torch.cat(predictions, dim=-1)
