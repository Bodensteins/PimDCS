#!/bin/bash

DATA_PATH="/home/zhouheng/datasets/susy.npz"
DATASET_NAME="susy"

python ./online_deep_learning.py \
  --data $DATA_PATH \
  --dataset-name $DATASET_NAME \
  -j 4 \
  --batch-size 1 \
  --lr 0.01 \
  --beta 0.9 \
  --s 0.2 \
  --hidden-layers 8 \
  --hidden-width 100 \
  --freeze-threshold 0.005 \
  --feature-size 18 \
  --classes 2 \
  --test-interval 100000
