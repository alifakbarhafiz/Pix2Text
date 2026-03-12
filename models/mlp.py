import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in=256, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out)
        )

    def forward(self, x):
        return self.net(x)
