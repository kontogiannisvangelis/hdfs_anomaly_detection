#!/bin/bash

LEARNING_RATES=(0.00005 0.0001)
TRAIN_BATCH_SIZES=(32 64)
LORA_RANKS=(16 32)

for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${TRAIN_BATCH_SIZES[@]}"; do
    for rank in "${LORA_RANKS[@]}"; do

      echo "--------------------------"
      echo "LR=$lr  BS=$bs  RANK=$rank" 
      echo "-------------------------"

      alpha=$((rank * 2))
      python3 train.py \
        --LEARNING_RATE $lr \
        --TRAIN_BATCH_SIZE $bs \
        --LORA_RANK $rank \
        --LORA_ALPHA $alpha

    done
  done
done
