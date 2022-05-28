import torchvision
from .label_adaption import *
models_dict = {
    'mobilenet_v2' : torchvision.models.mobilenet_v2,
}

def model_sel(model_name, num_classes=68*2):
    return models_dict[model_name](num_classes=num_classes)

adapt_dict = {
    'MLP_2L' : MLP_2L,
}

def adapt_sel(model_name):
    return adapt_dict[model_name]()