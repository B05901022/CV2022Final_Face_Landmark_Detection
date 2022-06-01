# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import argparse
from pathlib import Path
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
# import timm
import torchvision
from torchvision.utils import make_grid
from src.dataset import FaceDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import copy
import cv2
from models.models_select import model_sel, adapt_sel, fc_sel
from loss_function.loss_select import loss_sel
from src.tool import gen_result_data

def plot_keypoints_2(img, keypoints):
    img = copy.deepcopy(img)
    for y, x in keypoints:
        cv2.circle(img, (y, x), 5, (0, 0, 255), -1)
    return img

class Label_adaption(pl.LightningModule):
    def __init__(self, backbone, loss_func, up_scale, stage_1_model, lr, wd, beta1 = 0.9, beta2 = 0.999, momentum = 0.9):
        super().__init__()
        self.save_hyperparameters()
        backbone = adapt_sel(backbone, up_scale)
        self.backbone = backbone
        self.stage_1_model = stage_1_model
        self.stage_1_model.freeze()
        self.loss = loss_sel(loss_func)
        self.hard_mining = False
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.stage_1_model.eval()
        with torch.no_grad():
            out = self.stage_1_model(x)

        y_hat = self.backbone(out)
        if self.hard_mining:
            loss = torch.abs(y_hat - y) #(B,K)
            loss = torch.mean(loss, dim=1) #(B,)
            B = len(loss)
            S = int(B*0.5)
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:S]
            loss = torch.mean(loss) * 5.0
        else:
            loss = self.loss(y_hat, y) * 5.0
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_o, x, y = batch
        out = self.stage_1_model(x)
        y_hat = self.backbone(out)
        
        y = y.view(-1,68,2)
        y_hat = y_hat.view(-1,68,2)
        out = out.view(-1,68,2)

        kp_GT = y.cpu().detach().numpy()
        kp_p = y_hat.cpu().detach().numpy()
        
        dis = kp_GT - kp_p
        dis = np.sqrt(np.sum(np.power(dis, 2), 2))
        dis = np.mean(dis)
        NME = dis / 2 * 100
        
        # calculate loss
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)
        self.log('NME %', NME, on_step=True)
        
        
        # plot result on wandb
        if (batch_idx == 0):
            img_o = img_o.cpu().detach().numpy()[0]
            img = np.array(img_o).astype(np.uint8)
            img = cv2.resize(img, dsize=(256, 256))
            
            # keypoint transform
            kp_GT = y.cpu().detach().numpy()[0]
            kp_GT = (kp_GT + 1) / 2 * 256
            kp_GT = np.array(kp_GT).astype(np.int32)
            GT = plot_keypoints_2(img, kp_GT)

            kp_p = y_hat.cpu().detach().numpy()[0]
            kp_p = (kp_p + 1) / 2 * 256
            kp_p = np.array(kp_p).astype(np.int32)
            predict = plot_keypoints_2(img, kp_p)

            kp_b = out.cpu().detach().numpy()[0]
            kp_b = (kp_b + 1) / 2 * 256
            kp_b = np.array(kp_b).astype(np.int32)
            predict_b = plot_keypoints_2(img, kp_b)

            images = wandb.Image(GT, caption="GT")
            wandb.log({"GT": images})
            images = wandb.Image(predict, caption="after")
            wandb.log({"after": images})
            images = wandb.Image(predict_b, caption="before")
            wandb.log({"before": images})

    def test_step(self, batch, batch_idx):
        img_o, x, y = batch
        out = self.stage_1_model(x)
        y_hat = self.backbone(out)
        
        img_o = img_o.cpu().detach().numpy()
        
        y = y.view(-1,68,2)
        kp_GT = y.cpu().detach().numpy()
        kp_GT = (kp_GT + 1) / 2 * 256
        kp_GT = np.array(kp_GT).astype(np.int32)
        
        # before adaption
        out = out.view(-1,68,2)
        kp_b = out.cpu().detach().numpy()
        kp_b = (kp_b + 1) / 2 * 256
        kp_b = np.array(kp_b).astype(np.int32)
        
        # after adaption
        y_hat = y_hat.view(-1,68,2)
        kp_p = y_hat.cpu().detach().numpy()
        kp_p = (kp_p + 1) / 2 * 256
        kp_p = np.array(kp_p).astype(np.int32)
        
        imgs_gt, imgs_p, imgs_b = [], [], []
        for i in range(img_o.shape[0]):
            img = np.array(img_o[i]).astype(np.uint8)
            img = cv2.resize(img, dsize=(256, 256))
            kp1 = kp_GT[i]
            kp2 = kp_p[i]
            kp3 = kp_b[i]
            
            GT = plot_keypoints_2(img, kp1)
            predict = plot_keypoints_2(img, kp2)
            predict_b = plot_keypoints_2(img, kp3)
            imgs_gt.append(GT)
            imgs_p.append(predict)
            imgs_b.append(predict_b)
        
        # NME with normalized keypoint
        kp_GT = y.cpu().detach().numpy()
        kp_p = y_hat.cpu().detach().numpy()
        
        dis = kp_GT - kp_p
        dis = np.sqrt(np.sum(np.power(dis, 2), 2))
        dis = np.mean(dis, axis = 1)
        NME_list = dis / 2 * 100

        result = {
            'NME_list' : NME_list,
            'imgs_gt' : imgs_gt,
            'imgs_p' : imgs_p,
            'imgs_b' : imgs_b
        }
        
        return result
    
    def test_epoch_end(self, test_out):

        imgs_gt = []
        imgs_p = []
        imgs_b = []
        NME_list = []
        for result in test_out:
            NME_list.append(result['NME_list'])
            imgs_gt.append(result['imgs_gt'])
            imgs_p.append(result['imgs_p'])
            imgs_b.append(result['imgs_b'])
            
        NME_list = np.concatenate(NME_list, axis=0)
        epoch_NME = np.mean(NME_list)
        self.log('test_NME(%)', epoch_NME, on_epoch=True)

        imgs_gt = np.concatenate(imgs_gt, axis=0)
        imgs_p = np.concatenate(imgs_p, axis=0)
        imgs_b = np.concatenate(imgs_b, axis=0)
        
        temp = torch.from_numpy(imgs_gt[:16]).type(torch.ByteTensor)
        temp = temp.permute(0, 3, 1, 2)
        grid = make_grid(temp)
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().detach().numpy()
        images = wandb.Image(grid, caption="GT")
        wandb.log({"GT_test": [images]})
        
        temp = torch.from_numpy(imgs_b[:16]).type(torch.ByteTensor)
        temp = temp.permute(0, 3, 1, 2)
        grid = make_grid(temp)
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().detach().numpy()
        images = wandb.Image(grid, caption="Predict_before")
        wandb.log({"predict_before": [images]})
        
        temp = torch.from_numpy(imgs_p[:16]).type(torch.ByteTensor)
        temp = temp.permute(0, 3, 1, 2)
        grid = make_grid(temp)
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().detach().numpy()
        images = wandb.Image(grid, caption="Predict_after")
        wandb.log({"predict_after": [images]})

    def configure_optimizers(self):
        
        opt = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=self.momentum, weight_decay = self.wd)
        
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in [15, 25, 28] if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]

class FaceSynthetics(pl.LightningModule):
    def __init__(self, backbone, fc_extend, loss_func, lr, wd, beta1 = 0.9, beta2 = 0.999, momentum = 0.9):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = model_sel(backbone)
        self.fc = self.backbone.classifier
        if fc_extend :
            self.fc[1] = nn.Linear(self.fc[1].in_features, 68*3)
        self.backbone.classifier = nn.Identity()
        self.loss = loss_sel(loss_func)

        self.hard_mining = False
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        y = self.fc(y)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.fc(y_hat)
        if self.hard_mining:
            loss = torch.abs(y_hat - y) #(B,K)
            loss = torch.mean(loss, dim=1) #(B,)
            B = len(loss)
            S = int(B*0.5)
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:S]
            loss = torch.mean(loss) * 5.0
        else:
            loss = self.loss(y_hat, y) * 5.0
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_o, x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.fc(y_hat)
        
        y = y.view(-1,68,2)
        y_hat = y_hat.view(-1,68,2)
        
        kp_GT = y.cpu().detach().numpy()
        kp_p = y_hat.cpu().detach().numpy()
        
        dis = kp_GT - kp_p
        dis = np.sqrt(np.sum(np.power(dis, 2), 2))
        dis = np.mean(dis)
        NME = dis / 2 * 100
        
        # calculate loss
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)
        self.log('val_NME %', NME, on_step=True)
        
        
        # plot result on wandb
        if (batch_idx == 0):
            img_o = img_o.cpu().detach().numpy()[0]
            img = np.array(img_o).astype(np.uint8)
            img = cv2.resize(img, dsize=(256, 256))
            
            # keypoint transform
            kp_GT = y.cpu().detach().numpy()[0]
            kp_GT = (kp_GT + 1) / 2 * 256
            kp_GT = np.array(kp_GT).astype(np.int32)
            GT = plot_keypoints_2(img, kp_GT)

            kp_p = y_hat.cpu().detach().numpy()[0]
            kp_p = (kp_p + 1) / 2 * 256
            kp_p = np.array(kp_p).astype(np.int32)
            predict = plot_keypoints_2(img, kp_p)
            images = wandb.Image(GT, caption="GT")
            wandb.log({"GT": images})
            images = wandb.Image(predict, caption="predict")
            wandb.log({"predict": images})

    def test_step(self, batch, batch_idx):
        img_o, x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.fc(y_hat)
        
        img_o = img_o.cpu().detach().numpy()
        
        y = y.view(-1,68,2)
        kp_GT = y.cpu().detach().numpy()
        kp_GT = (kp_GT + 1) / 2 * 256
        kp_GT = np.array(kp_GT).astype(np.int32)
        
        y_hat = y_hat.view(-1,68,2)
        kp_p = y_hat.cpu().detach().numpy()
        kp_p = (kp_p + 1) / 2 * 256
        kp_p = np.array(kp_p).astype(np.int32)
        
        imgs_gt, imgs_p = [], []
        for i in range(img_o.shape[0]):
            img = np.array(img_o[i]).astype(np.uint8)
            img = cv2.resize(img, dsize=(256, 256))
            kp1 = kp_GT[i]
            kp2 = kp_p[i]
            
            GT = plot_keypoints_2(img, kp1)
            predict = plot_keypoints_2(img, kp2)
            imgs_gt.append(GT)
            imgs_p.append(predict)
        
        # NME with normalized keypoint
        kp_GT = y.cpu().detach().numpy()
        kp_p = y_hat.cpu().detach().numpy()
        
        dis = kp_GT - kp_p
        dis = np.sqrt(np.sum(np.power(dis, 2), 2))
        dis = np.mean(dis, axis = 1)
        NME_list = dis / 2 * 100

        result = {
            'NME_list' : NME_list,
            'imgs_gt' : imgs_gt,
            'imgs_p' : imgs_p
        }
        
        return result
    
    def test_epoch_end(self, test_out):

        imgs_gt = []
        imgs_p = []
        NME_list = []
        for result in test_out:
            NME_list.append(result['NME_list'])
            imgs_gt.append(result['imgs_gt'])
            imgs_p.append(result['imgs_p'])
            
        NME_list = np.concatenate(NME_list, axis=0)
        epoch_NME = np.mean(NME_list)
        self.log('test_NME(%)', epoch_NME, on_epoch=True)

        imgs_gt = np.concatenate(imgs_gt, axis=0)
        imgs_p = np.concatenate(imgs_p, axis=0)
        
        temp = torch.from_numpy(imgs_gt[:16]).type(torch.ByteTensor)
        temp = temp.permute(0, 3, 1, 2)
        grid = make_grid(temp)
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().detach().numpy()
        images = wandb.Image(grid, caption="GT")
        wandb.log({"GT_test": [images]})
        
        
        temp = torch.from_numpy(imgs_p[:16]).type(torch.ByteTensor)
        temp = temp.permute(0, 3, 1, 2)
        grid = make_grid(temp)
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().detach().numpy()
        images = wandb.Image(grid, caption="Predict")
        wandb.log({"predict_test": [images]})

    def configure_optimizers(self):
        
        # opt = torch.optim.Adam(self.parameters(), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.wd)

        opt = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=self.momentum, weight_decay = self.wd)

        def lr_step_func(epoch):
            return 0.1 ** len([m for m in [15, 25, 28] if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]


def main(hparams):

    # --- Fix random seed ---
    # fixed_seed(hparams.seed)
    pl.seed_everything(hparams.seed)
    # -- Logger instantiation ---
    logger = WandbLogger(project="cv_final~~~", name = hparams.exp_name, entity = 'cv_final')
    #wandb.init()
    #logger = TensorBoardLogger(hparams.log_path)
    # --- Trainer instantiation ---

    if hparams.train:
        # --- File creation ---
        Path(hparams.ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

        # --- Dataset 
        train_path = osp.join(hparams.dataset_path, 'synthetics_train')
        val_path = osp.join(hparams.dataset_path, 'aflw_val')
        train_set = FaceDataset(root_dir = train_path, is_train=True, is_coord_enhance = hparams.cood_en)
        val_set = FaceDataset(root_dir=val_path, is_train=False, is_coord_enhance = hparams.cood_en)

        train_loader = DataLoader(train_set, batch_size=hparams.bs, shuffle=True, num_workers=hparams.num_workers, pin_memory=True, prefetch_factor = 8)
        val_loader = DataLoader(val_set, batch_size=hparams.bs, shuffle=False)

        # --- Model instantiation ---
        model = FaceSynthetics(backbone=hparams.backbone, fc_extend = hparams.fc_extend, loss_func = hparams.loss, lr = hparams.lr, wd = hparams.wd, beta1 = hparams.beta1, beta2 = hparams.beta2, momentum = hparams.momentum)
        # --- Fit model to trainer ---
        checkpoint_callback = ModelCheckpoint(
            monitor = 'val_loss',
            dirpath = hparams.ckpt_path,
            filename = hparams.exp_name +'_'+ '{epoch:02d}-{val_loss:.4f}',
            save_top_k = 5,
            mode='min',
            save_weights_only=True,
            )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            devices = [hparams.gpu],
            accelerator="gpu",
            benchmark=True,
            logger = logger,
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=1,
            progress_bar_refresh_rate=2,
            max_epochs = hparams.epoch,
            profiler="simple",
            #limit_train_batches=0.1,
            #fast_dev_run=3,
        )
        trainer.fit(model, train_loader, val_loader)

    elif hparams.test:
        # --- File creation ---
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

        # --- Load model from checkpoint ---
        ckpt = osp.join(hparams.ckpt_path, hparams.ckpt_name)
        model = FaceSynthetics.load_from_checkpoint(ckpt)
        # --- Fit testing ---
        test_path = osp.join(hparams.dataset_path, 'aflw_val')
        test_set = FaceDataset(root_dir=test_path, is_train=False, is_coord_enhance = hparams.cood_en)
        test_loader = DataLoader(test_set, batch_size=hparams.bs, shuffle=False)

        trainer = pl.Trainer(
            devices = [hparams.gpu],
            accelerator="gpu",
            logger = logger,
        )
        trainer.test(model, dataloaders=test_loader)

    elif hparams.adaption_train:
        # --- File creation ---
        Path(hparams.ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

        # --- Dataset 
        train_path = osp.join(hparams.dataset_path, 'synthetics_train')
        val_path = osp.join(hparams.dataset_path, 'aflw_val')
        train_set = FaceDataset(root_dir = train_path, is_train=True, is_coord_enhance = hparams.cood_en)
        val_set = FaceDataset(root_dir=val_path, is_train=False, is_coord_enhance = hparams.cood_en)

        train_loader = DataLoader(train_set, batch_size=hparams.bs, shuffle=True, num_workers=hparams.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=hparams.bs, shuffle=False)

        # --- Model instantiation ---
        ckpt = osp.join(hparams.ckpt_path, hparams.ckpt_name)
        model_stage1 = FaceSynthetics.load_from_checkpoint(ckpt)
        #device = torch.device('cuda:{}'.format(hparams.gpu))
        #model_stage1 = model_stage1.to(device)
        model = Label_adaption(backbone=hparams.adap_backbone, loss_func = hparams.loss, up_scale = hparams.up_scale,stage_1_model = model_stage1, \
                lr = hparams.lr, wd = hparams.wd, beta1 = hparams.beta1, beta2 = hparams.beta2, momentum = hparams.momentum)
        
        # --- Fit model to trainer ---
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.ckpt_path,
            filename= hparams.exp_name +'_adaption_'+ '{epoch:02d}-{val_loss:.4f}',
            save_top_k=5,
            mode='min',
            save_weights_only=True,
            )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            devices = [hparams.gpu],
            accelerator="gpu",
            benchmark=True,
            logger = logger,
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=1,
            progress_bar_refresh_rate=1,
            max_epochs = hparams.epoch,
        )
        trainer.fit(model, train_loader, val_loader)
        
    elif hparams.adaption_test:
        # --- File creation ---
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)
        # --- Load model from checkpoint ---
        ckpt = osp.join(hparams.ckpt_path, hparams.ckpt_name)
        model = Label_adaption.load_from_checkpoint(ckpt)
        # --- Fit testing ---
        test_path = osp.join(hparams.dataset_path, 'aflw_val')
        test_set = FaceDataset(root_dir=test_path, is_train=False, is_coord_enhance = hparams.cood_en)
        test_loader = DataLoader(test_set, batch_size=hparams.bs, shuffle=False)

        trainer = pl.Trainer(
            devices = [hparams.gpu],
            accelerator="gpu",
            logger = logger,
        )
        trainer.test(model, dataloaders=test_loader)

    elif hparams.test_file:
        ckpt = osp.join(hparams.ckpt_path, hparams.ckpt_name)
        model = FaceSynthetics.load_from_checkpoint(ckpt)
        test_path = osp.join(hparams.dataset_path, 'aflw_test')
        gen_result_data(model = model, path = test_path)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # --- Path Arguments ---
    parser.add_argument('--dataset_path', help='File directory of dataset.', default='../data/', type=str)
    parser.add_argument('--ckpt_path', help='File directory to save checkpoint models.', default='./checkpoints/', type=str)
    parser.add_argument('--log_path', help='Experiment logging path.', default='./log/', type=str)

    # --- Logging Arguments ---
    parser.add_argument('--exp_name', help='Experiment name.', default='Exp_1', type=str)
    parser.add_argument('--ckpt_name', help='Checkpoint name.', default='default_ckpt', type=str)

    # --- Model select
    parser.add_argument('--backbone', default='mobilenet_v2', type=str)
    parser.add_argument('--adap_backbone', default='MLP_2L', type=str)
    parser.add_argument('--up_scale', help='Upscale for MLP', default=4, type=int)
    parser.add_argument('--fc_extend', help='Extend output dimension from 68*2 to 68*3', action='store_true')

    # --- Training Hyperparameters ---
    parser.add_argument('--epoch', help='Training epochs.', default=50, type=int)
    parser.add_argument('--loss', help='Loss function.', default='L1', type=str)
    parser.add_argument('--lr', help='Learning rate.', default=1e-2, type=float)
    parser.add_argument('--wd', help='Weight decay of optimizer', default=1e-5, type=float)
    parser.add_argument('--beta1', help='beta1 of Adam', default=0.9, type=float)
    parser.add_argument('--beta2', help='beta2 of Adam', default=0.999, type=float)
    parser.add_argument('--momentum', help='momentum', default=0.9, type=float)
    parser.add_argument('--bs', help='Batch size.', default=50, type=int)
    parser.add_argument('--cood_en', help='Append coordinate information', action='store_true')
    parser.add_argument('--seed', help='Random seed.', default=7, type=int)

    # --- GPU/CPU Arguments ---
    parser.add_argument('--num_workers', help='Number of workers', default=4, type=int)
    parser.add_argument('--gpu', help='Which GPU to be used (if none, specify as -1).', type=int, default = 3)

    # --- Mode ---
    parser.add_argument('--train', help='Run in train mode.', action='store_true')
    parser.add_argument('--test', help='Run in test mode.', action='store_true')
    parser.add_argument('--adaption_train', help='Run in test mode.', action='store_true')
    parser.add_argument('--adaption_test', help='Run in test mode.', action='store_true')
    parser.add_argument('--test_file', help='Generate testing data.', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
    wandb.finish()