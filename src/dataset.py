import os
import os.path as osp
import queue as Queue
import pickle
import threading
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .augs import RectangleBorderAugmentation
from PIL import Image
import copy

class FaceDataset(Dataset):
    def __init__(self, root_dir, is_train, is_coord_enhance = False, is_random_resize_crop = False, input_resolution = None, use_25shift = False):
        super(FaceDataset, self).__init__()

        #self.local_rank = local_rank
        self.is_train = is_train
        self.num_kps = 68
        self.use_25shift = use_25shift
        self.is_coord_enhance = is_coord_enhance
        self.shift = [-6, -3, 0, 3, 6]

        if input_resolution is None:
            self.input_size = 256
        else:
            self.input_size = input_resolution

        if is_random_resize_crop:
            transform_list = [
                # A.geometric.resize.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.augmentations.crops.transforms.RandomResizedCrop(self.input_size, self.input_size, always_apply=True)
            ]
        else:
            transform_list = [
                A.geometric.resize.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            ]

        if is_train:
            transform_list += \
                [
                    A.ColorJitter(brightness=0.8, contrast=0.5, p=0.5),
                    A.ToGray(p=0.1),
                    A.ISONoise(p=0.1),
                    A.MedianBlur(blur_limit=(1,7), p=0.1),
                    A.GaussianBlur(blur_limit=(1,7), p=0.1),
                    A.MotionBlur(blur_limit=(5,12), p=0.1),
                    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.05),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=40, interpolation=cv2.INTER_LINEAR, 
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    RectangleBorderAugmentation(limit=0.33, fill_value=0, p=0.2),
                    #A.Cutout(num_holes = 1, max_h_size = 64, max_w_size = 64),
                ]
        transform_list += \
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply = True),
                ToTensorV2(),
            ]
        self.transform = A.ReplayCompose(
            transform_list,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )


        
        
        self.root_dir = root_dir
        with open(osp.join(root_dir, 'annot.pkl'), 'rb') as f:
            annot = pickle.load(f)
            self.X, self.Y = annot
            print(len(self.X))
        # train_size = int(len(self.X)*0.99)
        train_size = len(self.X)

        #if local_rank==0:
        #    logging.info('data_transform_list:%s'%transform_list)
        flip_parts = ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
            [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
            [32, 36], [33, 35],
            [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
            [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56])
        self.flip_order = np.arange(self.num_kps)
        for pair in flip_parts:
            self.flip_order[pair[1]-1] = pair[0]-1
            self.flip_order[pair[0]-1] = pair[1]-1
        logging.info('len:%d'%len(self.X))
        print('!!!len:%d'%len(self.X))

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        image_path = os.path.join(self.root_dir, x)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(Image.open(image_path))

        if not(self.is_train):
            img_o = copy.deepcopy(img)

        label = y
        if self.transform is not None:
            t = self.transform(image=img, keypoints=label)
            flipped = False
            for trans in t["replay"]["transforms"]:
                if trans["__class_fullname__"].endswith('HorizontalFlip'):
                    if trans["applied"]:
                        flipped = True
            img = t['image']
            label = t['keypoints']
            label = np.array(label, dtype=np.float32)
            #print(img.shape)
            if flipped:
                #label[:, 0] = self.input_size - 1 - label[:, 0]  #already applied in horizantal flip aug
                label = label[self.flip_order,:]
            # label normalization 
            label /= (self.input_size/2) # to 0 ~ 2
            label -= 1.0 # to -1 ~ 1
            label = label.flatten()
            label = torch.tensor(label, dtype=torch.float32)

        if self.is_coord_enhance:
            end = self.input_size - 1
            x = torch.linspace(0, end, steps=self.input_size) / (self.input_size / 2) - 1
            y = torch.linspace(0, end, steps=self.input_size) / (self.input_size / 2) - 1
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            xx = xx.view(1, self.input_size, self.input_size)
            yy = yy.view(1, self.input_size, self.input_size)
            rr = torch.sqrt(xx * xx + yy * yy)
            img = torch.cat((img, xx, yy, rr), dim = 0)
        
        shift_imgs = None
        if (self.use_25shift):
            for x_shift in self.shift:
                for y_shift in self.shift:
                    temp = transforms.functional.affine(img = img.clone(), translate = [x_shift, y_shift], shear = 0, scale = 1, resample = False, fillcolor = [0, 0, 0], angle = 0)

                    if (shift_imgs is None):
                        shift_imgs = temp
                    else :
                        shift_imgs = torch.cat((shift_imgs, temp))
            shift_imgs = shift_imgs.view(25, 3, self.input_size, self.input_size)

        if self.is_train:
                return img, label
        elif self.use_25shift : 
            return img_o, shift_imgs, label,
        else:
            return img_o, img, label, 
    def __len__(self):
        return len(self.X)