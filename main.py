# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from src.tool import fixed_seed

def main(hparams):
	
	# --- Fix random seed ---
	fixed_seed(hparams.seed)

	# -- Logger instantiation ---

	# --- Trainer instantiation ---

	if hparams.train:
		# --- File creation ---
		Path(hparams.ckpt_path).mkdir(parents=True, exist_ok=True)
		Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

		# --- Model instantiation ---

		# --- Fit model to trainer ---

		# --- Fit testing ---

	elif hparams.test:
		# --- File creation ---
		Path(hparams.log_path).mkdir(parents=True, exist_ok=True)

		# --- Load model from checkpoint ---

		# --- Fit testing ---


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# --- Path Arguments ---
	parser.add_argument('--dataset_path', help='File directory of dataset.', default='./dataset/', type=str)
	parser.add_argument('--ckpt_path', help='File directory to save checkpoint models.', default='./checkpoints/', type=str)
	parser.add_argument('--log_path', help='Experiment logging path.', default='./log/', type=str)

	# --- Logging Arguments ---
	parser.add_argument('--exp_name', help='Experiment name.', default='Default Experiment Name', type=str)
	parser.add_argument('--ckpt_name', help='Checkpoint name.', default='default_ckpt', type=str)

	# --- Training Hyperparameters ---
	parser.add_argument('--epoch', help='Training epochs.', default=100, type=int)
	parser.add_argument('--lr', help='Learning rate.', default=0.001, type=float)
	parser.add_argument('--wd', help='Weight decay of optimizer', default=1e-5, type=float)
	parser.add_argument('--bs', help='Batch size.', default=32, type=int)
	parser.add_argument('--seed', help='Random seed.', default=7, type=int)

	# --- GPU/CPU Arguments ---
	parser.add_argument('--num_workers', help='Number of workers', default=1, type=int)
	parser.add_argument('--gpu', help='Which GPU to be used (if none, specify as -1).', type=int)

	# --- Mode ---
	parser.add_argument('--train', help='Run in train mode.', action='store_true')
	parser.add_argument('--test', help='Run in test mode.', action='store_true')