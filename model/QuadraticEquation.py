import torch
import torch.nn as nn


class QuadraticEquation(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, dtype=torch.float, device=device), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1, dtype=torch.float, device=device), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1, dtype=torch.float, device=device), requires_grad=True)

    def forward(self, X):
        return self.a * X * X + self.b * X + self.c
