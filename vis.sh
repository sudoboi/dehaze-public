#!/bin/bash

IMG_PATH="/content/Dataset/test"

MODEL_LOAD_NAME="Reside-Beta-subset-train-1"

python vis.py \
    "$IMG_PATH" \
    --load-name "$MODEL_LOAD_NAME"
    