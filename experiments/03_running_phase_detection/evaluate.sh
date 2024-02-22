#!/bin/bash
python test.py \
    --device cpu \
    --batch-size 16 \
    --data ./config/01_init.yaml \
    --weights ./runs/train/01_rp_detection6/weights/best.pt \
    --name 01_rp_detection_eval \
