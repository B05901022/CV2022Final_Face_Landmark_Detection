# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os
from torch.autograd import Variable
import numpy as np
import pickle
from torchvision import transforms
import os
from captum.attr import GradientShap
from captum.attr import visualization as viz
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import copy
from pathlib import Path

def model_transform(model):
    def backward_hook(module, grad_input, grad_output):
        return grad_input
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.saved_grad = m.register_backward_hook(backward_hook)
    return model

def visualization(model, test_image_path, devices, input_resolution = None, detect_target = "Default", save_img_path = "../CV_visualize/"):
    

    if input_resolution is None:
        img_size = 256
    else: 
        img_size = input_resolution

    transform = A.ReplayCompose(
        [A.geometric.resize.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()]
    )

    devices = devices[0] # Only use first GPU
    if devices != -1:
        device = f'cuda:{devices}'
        model.to(device)
    model.eval()
    
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = copy.deepcopy(img)
    t = transform(image=img)
    img = t['image']

    img = Variable(img).unsqueeze(0)
    if devices != -1:
        img = img.to(device)

    if detect_target == "FaceSilhouette":
        detect_target_list = [i for i in range(54)]
    elif detect_target == "Eyes":
        detect_target_list = [i for i in range(72,96)]
    elif detect_target == "Nose":
        detect_target_list = [i for i in range(54,72)]
    elif detect_target == "Mouth":
        detect_target_list = [i for i in range(96,136)]
    elif detect_target == "Default":
        detect_target_list = 0
    if detect_target != "Default":
        img = img.repeat((len(detect_target_list),1,1,1))

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')],
                                                       N=256)


    # --- Gradient Shap ---
    gradient_shap = GradientShap(model)
        
    # === baseline distribution ===
    rand_img_dist = torch.cat([img*0, img*1])
    
    
    attributions_gs = gradient_shap.attribute(img,
                                              n_samples=5,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=detect_target_list,
                                              )
    figure, axis = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.mean(dim=0).cpu().detach().numpy(), (1,2,0)),
                                                     #np.transpose(img.mean(dim=0).cpu().detach().numpy(), (1,2,0)),
                                                     orig_img,
                                                     ["original_image", "heat_map"],
                                                     ["all", "absolute_value"],
                                                     cmap=default_cmap,
                                                     show_colorbar=True,
                                                     use_pyplot=False,
                                                     titles=['Original', detect_target],
                                                     )

    figure.savefig(Path(save_img_path)/Path(f"{detect_target}.jpg"))

    del attributions_gs

    return
