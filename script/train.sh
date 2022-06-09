#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/
ckpt_path=./checkpoints/
log_path=./log/

# --- Logging Arguments ---
exp_name=Exp_25_mbv2_CA_L1_SWA                     # Name for wand running
ckpt_name=default_ckpt              # name of check point (Used when testing)

# --- Training Hyperparameters ---
epoch=30
loss=L1                           # Wing, AdaptWing, SmoothL1
lr=0.01
wd=0.00001
beta1=0.9                           # if use Adam, beta would be used
beta2=0.999
momentum=0.9                        # For SGD
bs=50                               # Batch size
seed=7
swa_epoch_start=0.8
annealing_epochs=10
swa_lrs=0.01
# --- GPU/CPU Arguments ---
num_workers=4
gpu="3"                               # Which gpu you want to use


backbone=mobilenet_v2_ca               # Model backbone   , mobilevit_v2, mobilenet_v2, mobilenet_v2_ca

# --- Available Flags ---
# --use_sam
# --cood_en # Coordinate enhancement

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
  --swa_epoch_start ${swa_epoch_start}\
  --annealing_epochs ${annealing_epochs}\
  --swa_lrs ${swa_lrs}\
