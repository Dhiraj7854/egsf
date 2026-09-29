"""BF Base Fusion (Step 3, Gate 0). Owner: A.

Two encoders + explicit instance gate alpha=softmax(W[h1;h2]), fused sum alpha_m h_m, linear clf.
Adam + wd + early stop on val-id. Train rho_corr in {0.6..0.9} x 5 seeds.
Gate 0 check: ID acc high, conflict acc drops >=10pts at 0.9, alpha_cue rises.
"""
import torch.nn as nn
class BaseFusion(nn.Module):
    def __init__(self, in1: int, in2: int, d: int = 64, K: int = 4):
        super().__init__()
        from .encoders import Encoder
        self.e1, self.e2 = Encoder(in1, d), Encoder(in2, d)
        self.gate = nn.Linear(2*d, 2)
        self.clf = nn.Linear(d, K)
    def forward(self, x1, x2):
        import torch, torch.nn.functional as F
        h1, h2 = self.e1(x1), self.e2(x2)
        alpha = F.softmax(self.gate(torch.cat([h1, h2], -1)), -1)
        h = alpha[:, :1]*h1 + alpha[:, 1:]*h2
        return self.clf(h), alpha
