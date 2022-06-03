#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/
ckpt_path=./checkpoints/
log_path=./log/

# --- Logging Arguments ---
exp_name=Exp_11_mbvitv2_0.75_L1_rand_cutout_64_1                      # Name for wand running
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
cood_en=false                       # Coordinate enhancement, 
seed=7
# --- GPU/CPU Arguments ---
num_workers=4
gpu=3                               # Which gpu you want to use


backbone=mobilevit_v2               # Model backbone   , mobilevit_v2, mobilenet_v2

wandb login
echo ${test}

if ${cood_en}; then
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
    --cood_en ${cood_en}\
    
else
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
    --train ${train}
    
fi


