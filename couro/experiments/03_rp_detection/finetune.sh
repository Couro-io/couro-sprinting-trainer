#!/bin/bash

python train.py \
    --workers 8 \
    --device cpu \
    --batch-size 16 \
    --data ./config/01_init.yaml \
    --img 640 640 \
    --cfg ./config/training/rp_detection.yaml \
    --weights ./models/baseline/yolov7_training.pt \
    --name 01_rp_detection \
    --hyp ./hyp/01_hyp.scratch.rp.yml