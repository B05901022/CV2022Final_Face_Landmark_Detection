# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
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
from src.dataset import FaceDataset
from src.tool import fixed_seed

class FaceSynthetics(pl.LightningModule):
	def __init__(self, backbone, lr, wd):
		super().__init__()
		self.save_hyperparameters()
		backbone = timm.create_model(backbone, num_classes=68*2)
		self.backbone = backbone
		self.loss = nn.L1Loss(reduction='mean')
		self.hard_mining = False
		self.lr = lr
		self.wd = wd

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
		opt = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=0.9, weight_decay = self.wd)
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
    logger = TensorBoardLogger(hparams.log_path)
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

        train_loader = DataLoader(train_set, batch_size=hparams.bs, shuffle=True, num_workers=hparams.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=hparams.bs, shuffle=False)

        # --- Model instantiation ---
        model = FaceSynthetics(backbone=hparams.backbone, lr = hparams.lr, wd = hparams.wd)
        # --- Fit model to trainer ---
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=hparams.ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=10,
            mode='min',
            )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            devices = hparams.gpu,
            accelerator="gpu",
            benchmark=True,
            logger = logger,
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=1,
            progress_bar_refresh_rate=1,
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
    parser.add_argument('--exp_name', help='Experiment name.', default='Default Experiment Name', type=str)
    parser.add_argument('--ckpt_name', help='Checkpoint name.', default='default_ckpt', type=str)

    # --- Model select
    parser.add_argument('--backbone', default='resnet50d', type=str)

    # --- Training Hyperparameters ---
    parser.add_argument('--epoch', help='Training epochs.', default=100, type=int)
    parser.add_argument('--lr', help='Learning rate.', default=0.001, type=float)
    parser.add_argument('--wd', help='Weight decay of optimizer', default=1e-5, type=float)
    parser.add_argument('--bs', help='Batch size.', default=32, type=int)
    parser.add_argument('--seed', help='Random seed.', default=7, type=int)

    # --- GPU/CPU Arguments ---
    parser.add_argument('--num_workers', help='Number of workers', default=3, type=int)
    parser.add_argument('--gpu', help='Which GPU to be used (if none, specify as -1).', type=int, default = -1)

    # --- Mode ---
    parser.add_argument('--train', help='Run in train mode.', action='store_true', default = True)
    parser.add_argument('--test', help='Run in test mode.', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    main(args)