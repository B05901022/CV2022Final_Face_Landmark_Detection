#!/bin/bash

# --- Detect Target ---
# 0: Default (x coordinate of first landmark)
# 1: Face Silhouette (54-points)
# 2: Eyes (24-points)
# 3: Nose (18-points)
# 4: Mouth (40-points)
detect_target=0                                 # set your detect target (0, 1, 2, 3, or 4)

# --- Path Arguments ---
ckpt_path=./checkpoints/                        # checkpoints path
test_image_path=../data/aflw_val/image00013.jpg # fetch image path (one image that you want to visualize)
save_img_path=../                  # save image path (path that you want to save result image)

# --- Logging Arguments ---
exp_name=Exp_27_mbv2_L1_fullres_60ep_nosch_vis                                           # Name for wand running
ckpt_name=merged.ckpt        # name of check point (Used when testing)

# --- GPU/CPU Arguments ---
gpu="-1"                                         # Which gpu you want to use, -1 for cpu

# --- Available Flags ---
# ----use_shift # use 25 shifted images

mkdir -p $save_img_path                         # create save image folder

wandb login                                     # wandb

# run main.py
python main.py \
  --visualize \
  --test_image_path $test_image_path \
  --exp_name $exp_name \
  --ckpt_path $ckpt_path \
  --ckpt_name $ckpt_name \
  --gpu $gpu \
  --detect_target $detect_target \
  --save_img_path $save_img_path \