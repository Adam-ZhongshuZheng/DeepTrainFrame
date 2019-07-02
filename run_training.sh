#!/bin/bash
set -e # Exit the script if an error happens


maxeps=100
lr=1e-2
batch_size=64
flod=1

# -r means to read the saved checkpoint model

python3 train.py  --max_epoch $maxeps --lr $lr --batch_size $batch_size -f $flod #-r
# python train.py  --max_epoch 100 --lr 1e-2 --batch_size 12 -f 1 #-r

