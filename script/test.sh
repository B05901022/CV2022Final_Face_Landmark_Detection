#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/               # where is the dataset
ckpt_path=./checkpoints/            # Path to save checkpoint
log_path=./log/                     # Path to save log

# --- Logging Arguments ---
exp_name=UniformSoup #Exp_MobilenetB_L1_FullRes_240EPOCH_NOSCH     # Name for wand running
ckpt_name=merged.ckpt #Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch_epoch=222-val_loss=0.0241.ckpt     # name of check point (Used when testing)

# --- Training Hyperparameters ---
epoch=30                         
loss=L1
lr=0.01
wd=0.00001
beta1=0.9                           # if use Adam, beta would be used
beta2=0.999
momentum=0.9                        # For SGD
bs=32                               # Batch size
seed=7

# --- GPU/CPU Arguments ---
num_workers=4
gpu="0,1"                           # Which gpu you want to use. -1 for cpu

backbone=mobilenet_v2               # Model backbone, mobilevit_v2, mobilenet_v2

wandb login

# --- Available Flags ---
# --cood_en          # Coordconv
# --use_shift      # use 25 shifted images

python main.py \
  --backbone ${backbone} \
  --dataset_path ${dataset_path} \
  --ckpt_path ${ckpt_path} \
  --log_path ${log_path} \
  --exp_name ${exp_name} \
  --ckpt_name ${ckpt_name} \
  --epoch ${epoch} \
  --loss ${loss} \
  --lr ${lr} \
  --wd ${wd} \
  --beta1 ${beta1} \
  --beta2 ${beta2} \
  --momentum ${momentum} \
  --bs ${bs} \
  --seed ${seed} \
  --num_workers ${num_workers} \
  --gpu ${gpu} \
  --test ${test} 