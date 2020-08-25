#!/bin/bash

IMG_PATH="../SOTS/outdoor/hazy/" #"../dataset/train/imgs"
LABEL_PATH="../SOTS/outdoor/gt/" #"../dataset/train/labels"

MODEL_LOAD_NAME="default-model"
MODEL_SAVE_NAME="default-model"

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
