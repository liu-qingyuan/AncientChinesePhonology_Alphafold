from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, num_classes: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
