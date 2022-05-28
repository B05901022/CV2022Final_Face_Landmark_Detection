import torch.nn as nn
import torch

class MLP_2L(nn.Module):
    def __init__(self):
        super(MLP_2L, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(136,544),
            nn.BatchNorm1d(544),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(544,136))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out