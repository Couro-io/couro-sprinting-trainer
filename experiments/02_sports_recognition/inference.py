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
        
if __name__ == "__main__":
    config_file = "./mmaction2/configs/recognition/csn/ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py"
    checkpoint_file = "./couro/experiments/02_sports_recognition/checkpoints/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth"
    video_file = './test_cases/_aJOs5B9T-Q.mp4'  

    model = init_recognizer(config_file, checkpoint_file, device='cpu')    
    pred_results = inference_recognizer(model, video_file)
    print(pred_results)
