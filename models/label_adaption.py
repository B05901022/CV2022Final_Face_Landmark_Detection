import torch.nn as nn
import torch

class MLP_2L(nn.Module):
    def __init__(self, up_scale = 4):
        super(MLP_2L, self).__init__()
        self.up_scale = up_scale

        self.layer1 = nn.Sequential(
            nn.Linear(136,up_scale * 136, bias = True),
            nn.BatchNorm1d(up_scale * 136))
            
        self.layer2 = nn.Sequential(
            nn.Linear(up_scale * 136,136, bias = True))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out + x