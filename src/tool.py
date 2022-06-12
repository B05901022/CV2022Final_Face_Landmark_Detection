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
from torchvision import transforms
import os



def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = True #False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def gen_result_data(model, path, devices, input_resolution = None, use_shift = False):
    shift = [-6, -3, 0, 3, 6]

    if input_resolution is None:
        img_size = 256
    else: 
        img_size = input_resolution

    transform = A.ReplayCompose(
        [A.geometric.resize.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()]
    )
    if devices[0] != -1:
        devices_str = "cuda:" + str(devices[0])
    else:
        devices_str = "cpu"
    device = torch.device(devices_str)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in devices])
 
    # device = torch.device("cuda")
    # if len(devices) > 1:
    #     model = torch.nn.DataParallel(model, device_ids = [i for i in range(len(devices))])

    image_names = os.listdir(path)
    model.to(device)
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

            if (use_shift):
                shift_imgs = None
                for x_shift in shift:
                    for y_shift in shift:
                        temp = transforms.functional.affine(img = img.clone(), translate = [x_shift, y_shift], shear = 0, scale = 1, resample = False, fillcolor = [0, 0, 0], angle = 0)

                        if (shift_imgs is None):
                            shift_imgs = temp
                        else :
                            shift_imgs = torch.cat((shift_imgs, temp))
                shift_imgs = shift_imgs.view(25, 3, img_size, img_size)
                img = shift_imgs.to(device)
            else:
                img = Variable(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img)

                img_flip = img.clone().flip(dims=(-1,))
                out_flip = model(img_flip)
                out_flip = out_flip.view(-1,68,2)
                
                out = (out + model._flip_keypoints(out_flip).view(-1,68 * 2)) / 2
            if (use_shift):
                out = torch.sum(out, dim = 0) / img.shape[0]
                kp_p = out.cpu().detach().numpy()
            else:
                kp_p = out.cpu().detach().numpy()[0]

            kp_p = (kp_p + 1) / 2 * 384

            data = ' '.join(map(str, kp_p))
            f.write(data + "\n")

            i = i + 1
            if (i % 100 == 0):
                print("%d images" % (i))

            