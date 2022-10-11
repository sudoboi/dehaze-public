#!/bin/bash

IMG_PATH="/content/Dataset/Reside-Beta-subset/test/hazy_dark/part1"

MODEL_LOAD_NAME="Reside-Beta-subset-train-8"

python vis.py \
    "$IMG_PATH" \
    --load-name "$MODEL_LOAD_NAME"
    