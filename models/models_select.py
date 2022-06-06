import torchvision
import torch.nn as nn
from .label_adaption import *
from .models_exp import *
from .mobilenext_v3_CMAB import *
from .mobilenet_v2_CBAM import *
from .models_mobilenet_v2_ca import *

models_dict = {
    'mobilenet_v2' : torchvision.models.mobilenet_v2,
    'mobilevit_v2' : MobileViTv2,
    'mobilenet_v2_ca' : MBV2_CA,
}

def model_sel(model_name, num_classes=68*2, cood_en=False):
    in_channels = 6 if cood_en else 3

    if model_name == 'mobilenet_v2':
        model = models_dict[model_name](num_classes=num_classes)
        model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=(1,1),
            bias=False,
            )
        nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out')
    else:
        model = models_dict[model_name](in_channels=in_channels, num_classes=num_classes)
    return  model

adapt_dict = {
    'MLP_2L' : MLP_2L,
}

def adapt_sel(model_name, up_scale = 4):
    return adapt_dict[model_name](up_scale = up_scale)

fc_dict = {

}

def fc_sel(fc_name = None):
    return models_dict[fc_name]
