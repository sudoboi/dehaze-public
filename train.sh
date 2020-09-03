#!/bin/bash

IMG_PATH="/media/harshan01/Seagate Backup Plus Drive/Sally_CV/image-dehazing/Dataset/Reside-V0/OTS/hazy" #"../SOTS/outdoor/hazy/" #"../dataset/train/imgs"
LABEL_PATH="/media/harshan01/Seagate Backup Plus Drive/Sally_CV/image-dehazing/Dataset/Reside-V0/OTS/clear" #"../SOTS/outdoor/gt/" #"../dataset/train/labels"

MODEL_LOAD_NAME="default-model"
MODEL_SAVE_NAME="reside-v0-train-1"

BATCH_SIZE=2
EPOCHS=1 #25

python train.py \
    "$IMG_PATH" \
    "$LABEL_PATH" \
    --save-name "$MODEL_SAVE_NAME" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --cache-ds

# python train.py \
#   "$IMG_PATH" \
#   "$LABEL_PATH" \
#   --load-name "$MODEL_LOAD_NAME" \
#   --save-name "$MODEL_SAVE_NAME" \
#   --batch-size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --cache-ds
