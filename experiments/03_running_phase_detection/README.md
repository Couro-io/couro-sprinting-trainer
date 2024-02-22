# Experiment 3: Running Phase Detection

## Experiment objectives
We aim to fine-tune foundational computer vision models for recognizing running phases.

## Usage
First, load the virtual environment:

```Python
source venv/bin/activate
```

### Model training
The `./finetune.sh` script performs fine tuning, using YOLO v7 as the baseline model.

### Model evaluation
The `./evaluate.sh` script uses a validation set to determine model metrics.
