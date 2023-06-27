# Run baseline yolov7
python3 ./couro/experiments/baseline/inference.py

:'
# Fine-tuning for running phase detection
python3 ./couro/experiments/running_phase_detection/train.py \
    --epochs 100 \
    --workers 4 \ 
    --device 0 \ 
    --batch-size 16 \
    --data 

# Fine-tuning for action recognition

'