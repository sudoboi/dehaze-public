#!/bin/bash

IMG_PATH="../SOTS/outdoor/hazy/" #"../dataset/imgs"
LABEL_PATH="../SOTS/outdoor/gt/" #"../dataset/labels"

MODEL_LOAD_PATH="./saved-models/default-model.h5"
MODEL_SAVE_PATH="./saved-models/default-model.h5"

BATCH_SIZE=1
EPOCHS=25

python train.py \
    $IMG_PATH \
    $LABEL_PATH \
    --save-path $MODEL_SAVE_PATH \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --cache-ds

# python train.py \
#   $IMG_PATH \
#   $LABEL_PATH \
#   --load-path $MODEL_LOAD_PATH \
#   --save-path $MODEL_SAVE_PATH \
#   --batch-size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --cache-ds
