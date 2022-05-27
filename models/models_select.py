#
# Import your models
#
import torchvision



#
# Add entrypoints of your models
#
models_dict = {
    'mobilenet_v2' : torchvision.models.mobilenet_v2,
}

def model_sel(model_name, num_classes=68*2):
    return models_dict[model_name](num_classes=num_classes)