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
import timm
import torchvision
from src.dataset import FaceDataset
from pytorch_lightning.loggers import WandbLogger
import wandb

class FaceSynthetics(pl.LightningModule):
    def __init__(self, backbone, lr, wd, beta1 = 0.9, beta2 = 0.999):
        super().__init__()
        self.save_hyperparameters()
        backbone = timm.create_model(backbone, num_classes=68*2)
        # backbone = torchvision.models.shufflenet_v2_x1_0(num_classes=68*2)
        self.backbone = backbone
        self.loss = nn.MSELoss(reduction='mean')
        self.hard_mining = False
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
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
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.wd)
        
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
    logger = WandbLogger(project="CV_final", name = hparams.exp_name)
    #logger = TensorBoardLogger(hparams.log_path)
    # --- Trainer instantiation ---

    if hparams.train:
        # --- File creation ---
        Path(hparams.ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

        # --- Dataset 
        train_path = osp.join(hparams.dataset_path, 'synthetics_train')
        val_path = osp.join(hparams.dataset_path, 'aflw_val')
        train_set = FaceDataset(root_dir = train_path, is_train=True)
        val_set = FaceDataset(root_dir=val_path, is_train=False)

        train_loader = DataLoader(train_set, batch_size=hparams.bs, shuffle=True, num_workers=hparams.num_workers, pin_memory=True, prefetch_factor = 8)
        val_loader = DataLoader(val_set, batch_size=hparams.bs, shuffle=False)

        # --- Model instantiation ---
        model = FaceSynthetics(backbone=hparams.backbone, lr = hparams.lr, wd = hparams.wd, beta1 = hparams.beta1, beta2 = hparams.beta2)
        # --- Fit model to trainer ---
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=5,
            mode='min',
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
        )
        trainer.fit(model, train_loader, val_loader)

    elif hparams.test:
        # --- File creation ---
        Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

        # --- Load model from checkpoint ---

        # --- Fit testing ---


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
    parser.add_argument('--backbone', default='mobilenetv3_small_075', type=str)

    # --- Training Hyperparameters ---
    parser.add_argument('--epoch', help='Training epochs.', default=150, type=int)
    parser.add_argument('--lr', help='Learning rate.', default=1e-2, type=float)
    parser.add_argument('--wd', help='Weight decay of optimizer', default=1e-5, type=float)
    parser.add_argument('--beta1', help='beta1 of Adam', default=0.9, type=float)
    parser.add_argument('--beta2', help='beta2 of Adam', default=0.999, type=float)
    parser.add_argument('--bs', help='Batch size.', default=50, type=int)
    parser.add_argument('--seed', help='Random seed.', default=7, type=int)
    # --- GPU/CPU Arguments ---
    parser.add_argument('--num_workers', help='Number of workers', default=4, type=int)
    parser.add_argument('--gpu', help='Which GPU to be used (if none, specify as -1).', type=int, default = 3)

    # --- Mode ---
    parser.add_argument('--train', help='Run in train mode.', action='store_true', default = True)
    parser.add_argument('--test', help='Run in test mode.', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    main(args)
    wandb.finish()