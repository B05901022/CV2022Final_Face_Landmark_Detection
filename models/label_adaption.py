import torch.nn as nn
import torch

class MLP_2L(nn.Module):
    def __init__(self):
        super(MLP_2L, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(136,544),
            nn.Tanh())
        self.layer2 = nn.Sequential(
            nn.Linear(544,544),
            nn.Tanh())
        self.layer3 = nn.Sequential(
            nn.Linear(544,136),
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out