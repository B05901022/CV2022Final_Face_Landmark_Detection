import torch.nn as nn


loss_dict = {
    'L1' : nn.L1Loss(reduction='mean'),
    'MSE': nn.MSELoss(reduction='mean')
}

def loss_sel(model_name):
    return loss_dict[model_name]

