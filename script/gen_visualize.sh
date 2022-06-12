#!/bin/bash

# --- Detect Target ---
# 0: Default (x coordinate of first landmark)
# 1: Face Silhouette (54-points)
# 2: Eyes (24-points)
# 3: Nose (18-points)
# 4: Mouth (40-points)
detect_target=0

# --- Path Arguments ---
ckpt_path=./checkpoints/
test_image_path=../data/aflw_val/image00013.jpg

# --- Logging Arguments ---
exp_name=Exp_27_mbv2_L1_fullres_60ep_nosch_vis                     # Name for wand running
ckpt_name=Exp_27_mbnetv2_L1_rand_fullres_60ep_nosch_epoch=59-val_loss=0.0253.ckpt              # name of check point (Used when testing)

# --- GPU/CPU Arguments ---
gpu="0"                               # Which gpu you want to use

# --- Available Flags ---
# ----use_shift # use 25 shifted images

wandb login

python main.py \
  --visualize \
  --test_image_path $test_image_path \
  --exp_name $exp_name \
  --ckpt_path $ckpt_path \
  --ckpt_name $ckpt_name \
  --gpu $gpu \
  --detect_target $detect_target \