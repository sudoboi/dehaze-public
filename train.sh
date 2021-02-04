#!/bin/bash

IMG_PATH="/media/harshan01/Seagate Backup Plus Drive/Sally_CV/image-dehazing/Dataset/Reside-V0/OTS/hazy" #"../SOTS/outdoor/hazy/" #"../dataset/train/imgs"
LABEL_PATH="/media/harshan01/Seagate Backup Plus Drive/Sally_CV/image-dehazing/Dataset/Reside-V0/OTS/clear" #"../SOTS/outdoor/gt/" #"../dataset/train/labels"

MODEL_LOAD_NAME="None" # "None" (default) implies no model checkpoint loading is done. Use "default-model" as fallback
MODEL_SAVE_NAME="reside-v0-train-1"

BATCH_SIZE=2
EPOCHS_P1=1 # Phase 1 epochs: 25 (default)
EPOCHS_P2=1 # Phase 2 epochs: 25 (default)

python train.py \
    "$IMG_PATH" \
    "$LABEL_PATH" \
    --load-name "$MODEL_LOAD_NAME" \
    --save-name "$MODEL_SAVE_NAME" \
    --batch-size $BATCH_SIZE \
    --epochs_p1 $EPOCHS_P1 \
    --epochs_p2 $EPOCHS_P2 \
    --cache-ds
