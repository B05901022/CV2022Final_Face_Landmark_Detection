# -*- coding: utf-8 -*-
from pathlib import Path
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from main import FaceSynthetics
from src.dataset import FaceDataset

MODEL_CKPT_PATH = './checkpoints/'
MODEL_CKPTS = [
	'Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch_epoch=222-val_loss=0.0241.ckpt',
	'Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch_epoch=239-val_loss=0.0241.ckpt',
	#'Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch_epoch=233-val_loss=0.0241.ckpt',
]

def uniform_soup(model_path, model_list):

	models = [Path(model_path)/Path(i) for i in model_list]

	default_model = FaceSynthetics.load_from_checkpoint(models[0])
	model_dict = default_model.backbone.state_dict()
	model_dict2 = default_model.fc.state_dict()
	soups = {key:[] for key in model_dict}
	soups2 = {key:[] for key in model_dict2}

	for i, model_i in enumerate(models):
		new_model = FaceSynthetics.load_from_checkpoint(model_i)
		backbone = new_model.backbone.state_dict()
		for k, v in backbone.items():
			soups[k].append(v)
		fc = new_model.fc.state_dict()
		for k, v in fc.items():
			soups2[k].append(v)
	soups = {k:(torch.sum(torch.stack(v), axis=0)/len(v)).type_as(v[0]) for k,v in soups.items() if len(v) != 0}
	soups2 = {k:(torch.sum(torch.stack(v), axis=0)/len(v)).type_as(v[0]) for k,v in soups2.items() if len(v) != 0}
	model_dict.update(soups)
	model_dict2.update(soups2)
	default_model.backbone.load_state_dict(model_dict)
	default_model.fc.load_state_dict(model_dict2)

	# --- Dataset 
	train_path = Path('../data/')/Path('synthetics_train')
	val_path = Path('../data/')/Path('aflw_val')
	train_set = FaceDataset(root_dir = train_path, is_train=True, is_coord_enhance = False,  is_random_resize_crop = False, input_resolution=384)
	val_set = FaceDataset(root_dir=val_path, is_train=False, is_coord_enhance = False, input_resolution=384)

	train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

	trainer = pl.Trainer(fast_dev_run=True)
	wandb.init()
	trainer.fit(default_model, train_loader, val_loader)
	trainer.save_checkpoint(Path(model_path)/Path('merged.ckpt'), weights_only=True)
	wandb.finish()

	return

if __name__ == "__main__":
	uniform_soup(MODEL_CKPT_PATH, MODEL_CKPTS)