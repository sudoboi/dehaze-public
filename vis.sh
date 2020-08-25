#!/bin/bash

IMG_PATH="../SOTS_vis/" #"../dataset/test/imgs"

MODEL_LOAD_NAME="default-model"

python vis.py \
    "$IMG_PATH" \
    --load-name "$MODEL_LOAD_NAME"
    