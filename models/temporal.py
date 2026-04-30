"""Temporal modeling with Bidirectional LSTM."""

import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """BiLSTM for capturing temporal inconsistencies across frames."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence of frame features.

        Args:
            x: (B, T, input_dim) temporal feature sequence.

        Returns:
            (B, output_dim) aggregated temporal representation.
        """
        output, (h_n, _) = self.lstm(x)  # output: (B, T, hidden*dirs)

        # Use the final hidden states from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * 2, B, hidden)
            fwd = h_n[-2]  # last layer forward
            bwd = h_n[-1]  # last layer backward
            combined = torch.cat([fwd, bwd], dim=1)  # (B, hidden*2)
        else:
            combined = h_n[-1]  # (B, hidden)

        return self.layer_norm(combined)
