#!/bin/bash

# --- Path Arguments ---
dataset_path=../data/
ckpt_path=./checkpoints/
log_path=./log/

# --- Logging Arguments ---
exp_name=Exp_2_adap_test                      # Name for wand running
ckpt_name=Exp_2_adap_train_adaption_epoch=20-val_loss=0.0279.ckpt              # name of check point (Used when testing)

# --- Training Hyperparameters ---
epoch=30    
loss=L1                     
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


backbone=mobilenet_v2               # Model backbone
adap_backbone=MLP_2L
up_scale=4

wandb login


if ${cood_en}; then
  python main.py \
    --backbone ${backbone}\
    --up_scale ${up_scale} \
    --adap_backbone ${adap_backbone}\
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
    --adaption_test ${adaption_test} \
    --cood_en ${cood_en}\

else
  python main.py \
    --backbone ${backbone}\
    --up_scale ${up_scale} \
    --adap_backbone ${adap_backbone}\
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
    --adaption_test ${adaption_test}\

fi
