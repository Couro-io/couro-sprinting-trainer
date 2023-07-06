"""
inference.py

Run Action Recognition inference on a video file and print the results.
"""

import sys
sys.path.append('./mmaction2/')


import torch
from torchvision.io import read_video
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

def load_model(config_file:str, checkpoint_file:str):
    """Load PyTorch model from config and checkpoint files."""
    model = init_recognizer(config_file, checkpoint_file, device='cpu')
    
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
    config_file = "./mmaction2/configs/recognition/csn/ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py"
    checkpoint_file = "./couro/experiments/02_sports_recognition/checkpoints/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth"
    video_file = './test_cases/_aJOs5B9T-Q.mp4'  # Provide the path to the video file

    model = load_model(config_file, checkpoint_file)
    pred_results = inference_recognizer(model, video_file)
