#!/bin/bash

# --- Path Arguments ---
ckpt_path=./checkpoints/
test_image_path=../data/synthetics_train/000009.jpg # synthetics_train/000009.jpg, aflw_val/image00013.jpg, TTL/TTL.jpg
save_img_path=../CV_visualize/train/result.jpg

# --- Logging Arguments ---
exp_name=Exp_30_mbnetv2b_inference_one                    # Name for wand running
ckpt_name=Exp_30_mbnetv2b_L1_rand_fullres_180ep_nosch_epoch174-val_loss0.0246.ckpt              # name of check point (Used when testing)

# --- GPU/CPU Arguments ---
gpu="3"                               # Which gpu you want to use


wandb login

python main.py \
  --inf_one \
  --test_image_path $test_image_path \
  --exp_name $exp_name \
  --ckpt_path $ckpt_path \
  --ckpt_name $ckpt_name \
  --gpu $gpu \
  --save_img_path $save_img_path \