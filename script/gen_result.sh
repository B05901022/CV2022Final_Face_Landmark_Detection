#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/
ckpt_path=./checkpoints/
log_path=./log/

# --- Logging Arguments ---
exp_name=UniformSoup #Exp_31_mbv2b_L1_fullres_240ep_nosch                      # Name for wand running
ckpt_name=merged.ckpt #Exp_31_mbnetv2b_L1_rand_fullres_240ep_nosch_epoch=222-val_loss=0.0241.ckpt              # name of check point (Used when testing)

# --- Training Hyperparameters ---
epoch=30                         
lr=0.01
loss=L1
wd=0.00001
beta1=0.9                           # if use Adam, beta would be used
beta2=0.999
momentum=0.9                        # For SGD
bs=32                               # Batch size
cood_en=false                       # Coordinate enhancement, 
seed=7
# --- GPU/CPU Arguments ---
num_workers=4
gpu="0,1"                               # Which gpu you want to use


backbone=mobilenet_v2               # Model backbone

# --- Available Flags ---
# ----use_shift # use 25 shifted images

wandb login

if ${cood_en}; then
  python main.py \
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
    --test_file ${test_file}\
    --cood_en ${cood_en}\
    
else
  python main.py \
    --dataset_path ${dataset_path} \
    --ckpt_path ${ckpt_path} \
    --log_path ${log_path} \
    --exp_name ${exp_name} \
    --ckpt_name ${ckpt_name} \
    --epoch ${epoch} \
    --lr ${lr} \
    --wd ${wd} \
    --beta1 ${beta1} \
    --beta2 ${beta2} \
    --momentum ${momentum} \
    --bs ${bs} \
    --seed ${seed} \
    --num_workers ${num_workers} \
    --gpu ${gpu} \
    --test_file ${test_file}
    
fi


