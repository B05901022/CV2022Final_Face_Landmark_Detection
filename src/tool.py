# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from torch.autograd import Variable
import numpy as np
import pickle

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = True #False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def gen_result_data(model, path):
    transform = A.ReplayCompose(
        [A.geometric.resize.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()]
    )

    image_names = os.listdir(path)
    model.eval()
    i = 0

    with open('solution.txt', 'w') as f:

        for image_name in image_names:
            f.write(image_name + " ")

            image_path = os.path.join(path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = transform(image=img)
            img = t['image']
            img = Variable(img).unsqueeze(0)
            with torch.no_grad():
                out = model(img)

            kp_p = out.cpu().detach().numpy()[0]
            kp_p = (kp_p + 1) / 2 * 384

            data = ' '.join(map(str, kp_p))
            f.write(data + "\n")

            i = i + 1
            if (i % 100 == 0):
                print("%d images", i)


