import torch.nn as nn
from .loss_exp import *

loss_dict = {
    'L1' : nn.L1Loss(reduction='mean'),
    'MSE': nn.MSELoss(reduction='mean'),
    'Wing': WingLoss(),
    'AdaptWing': AdpativeWingLoss(),
    'SmoothL1': SmoothL1Loss(),
}

def loss_sel(model_name):
    return loss_dict[model_name]

