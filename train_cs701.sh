#!/bin/bash
EPOCH_TIME=1000
OUT_DIR='./model_out'
CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
DATA_DIR='datasets/cs701_224'
LEARNING_RATE=0.001
IMG_SIZE=224
BATCH_SIZE=24

echo "start train model"
python train_cs701.py --dataset cs701 --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE
