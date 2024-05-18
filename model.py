import torch
import torch.nn as nn


class Odevity(nn.Module):
    def __init__(self):
        super(Odevity, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)