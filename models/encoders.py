"""MLP encoders (d=64). Owner: A."""
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x): return self.net(x)
