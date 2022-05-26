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


class FaceDataset(Dataset):
    def __init__(self, root_dir, is_train):
        super(FaceDataset, self).__init__()

        #self.local_rank = local_rank
        self.is_train = is_train
        self.input_size = 256
        self.num_kps = 68
        

        transform_list = [
            A.geometric.resize.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True)
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
        return img, label

    def __len__(self):
        return len(self.X)