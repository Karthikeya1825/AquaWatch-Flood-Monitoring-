"""
PyTorch LSTM Architecture for Rain Prediction Time-Series Modeling.
Processes multi-day sliding sequences of meteorological variables to predict
next-day precipitation probability.
"""

import torch
import torch.nn as nn


class RainLSTM(nn.Module):
    def __init__(
        self,
        num_features: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, num_features)
        out, (hn, cn) = self.lstm(x)
        # Use final time step hidden representation
        last_step = out[:, -1, :]  # (B, hidden_size)
        logits = self.classifier(last_step)  # (B, 1)
        return logits.squeeze(-1)
