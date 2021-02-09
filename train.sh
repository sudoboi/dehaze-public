#!/bin/bash

IMG1_PATH="/content/Dataset/Reside-Beta-subset/hazy/part1"
IMG2_PATH="/content/Dataset/Reside-Beta-subset/hazy_dark/part1"
LABEL_PATH="/content/Dataset/Reside-Beta-subset/clear/clear"

MODEL_LOAD_NAME="None" # "None" (default) implies no model checkpoint loading is done. Use "default-model" as fallback
MODEL_SAVE_NAME="Reside-Beta-subset-train-1"

BATCH_SIZE=2
EPOCHS_P1=25 # Phase 1 epochs: 25 (default)
EPOCHS_P2=50 # Phase 2 epochs: 25 (default)

python train.py \
    "$IMG1_PATH" \
    "$IMG2_PATH" \
    "$LABEL_PATH" \
    --load-name "$MODEL_LOAD_NAME" \
    --save-name "$MODEL_SAVE_NAME" \
    --batch-size $BATCH_SIZE \
    --epochs_p1 $EPOCHS_P1 \
    --epochs_p2 $EPOCHS_P2 \
    --cache-ds
