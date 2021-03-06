#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/               # where is the dataset
ckpt_path=./checkpoints/            # Path to save checkpoint
log_path=./log/                     # Path to save log

# --- Logging Arguments ---
exp_name=Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch        # Name for wand running
ckpt_name=default_ckpt              # name of check point (Used when testing)

# --- Training Hyperparameters ---
epoch=240
loss=L1                             # L1, Wing, AdaptWing, SmoothL1, GNLL
lr=0.01
wd=0.00001
beta1=0.9                           # if use Adam, beta would be used
beta2=0.999
momentum=0.9                        # For SGD
bs=32                               # Batch size
seed=7                              # Random seed

# --- SWA setting
swa_epoch_start=0.8                 
annealing_epochs=10
swa_lrs=0.01

# --- GPU/CPU Arguments ---
num_workers=4
gpu="0,1"                             # Which gpu you want to use. -1 for cpu


backbone=mobilenet_v2_b               # Model backbone, mobilevit_v2, mobilenet_v2, mobilenet_v2_ca, mobilenet_v2_lk, mobilenet_v2_b, 

# --- Available Flags ---
# --use_sam   # Use sam 
# --cood_en   # Coordconv
# --use_swa   # use swa
# --lr_nosch  # Don't use learning rate scheduler

wandb login
echo ${test}

python main.py \
  --backbone ${backbone}\
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
  --train ${train}\
  --lr_nosch \
  #--use_swa \
  #--swa_epoch_start ${swa_epoch_start}\
  #--annealing_epochs ${annealing_epochs}\
  #--swa_lrs ${swa_lrs}\
  #--use_sam \
