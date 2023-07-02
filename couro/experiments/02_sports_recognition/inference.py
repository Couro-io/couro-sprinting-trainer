"""
inference.py

Run Action Recognition inference on a video file and print the results.
"""

import sys
sys.path.append('./mmaction2/')


import torch
from torchvision.io import read_video
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer, inference_recognizer

def load_model(config_file:str, checkpoint_file:str):
    """Load PyTorch model from config and checkpoint files."""
    model = build_model(config_file)
    checkpoint = load_checkpoint(model, checkpoint_file)
    model.eval()
    
    return model

def process_video_data(model):
    """Process Video data for inference."""
    video_frames, _, _ = read_video(video_file)
    transform = Compose(model.cfg.data.test.pipeline)
    input_data = dict(img_group=[video_frames])
    input_data = transform(input_data)
    
    return input_data
    
def run_inference(model, input_data):
    """Run inference from model"""
    with torch.no_grad():
        result = inference_recognizer(model, input_data)

    labels = model.cfg.data.test.pipeline[1]['label_file']
    action_labels = [labels[i] for i in result.argmax(axis=1)]
    action_scores = result.max(axis=1)

    for label, score in zip(action_labels, action_scores):
        print(f"Action: {label}, Score: {score}")
        
if __name__ == "__main__":
    config_file = ""
    checkpoint_file = ""
    video_file = 'path_to_video_file.mp4'  # Provide the path to the video file

    model = load_model(config_file, checkpoint_file)
    input_data = process_video_data(model)
    run_inference(model, input_data)
