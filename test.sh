#!/bin/bash

IMG_PATH="../SOTS/outdoor/hazy/" #"../dataset/test/imgs"
LABEL_PATH="../SOTS/outdoor/gt/" #"../dataset/test/labels"

MODEL_LOAD_NAME="default-model"

BATCH_SIZE=1

python test.py \
    "$IMG_PATH" \
    "$LABEL_PATH" \
    --save-name "$MODEL_SAVE_NAME" \
    --batch-size $BATCH_SIZE \
    --cache-ds
