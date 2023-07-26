"""
Evaluate model performance.
"""

import os
import sys
import tempfile
from urllib.parse import unquote
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

class FrameDataset(Dataset):
    """Dataloader for video data."""
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return frame, idx
    
def load_model(device, model_path:str="./yolov7/yolov7-w6-pose.pt"):
    """Default model is yolo v7 pose estimation"""
    model = torch.load(model_path, map_location=device)['model']
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

def run_inference(image, model):
    """Runs inference on a single image."""
    scale_percent = 60 # percent of original size, you can adjust this as needed
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # if the image is a PyTorch tensor, convert it to a numpy array
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # resize the image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    image = letterbox(image, 1920, stride=64, auto=True)[0] 
    image = transforms.ToTensor()(image) 
    if torch.cuda.is_available():
        image = image.half().to(device)
    image = image.unsqueeze(0) 
    with torch.no_grad():
        output, _ = model(image)
    return output, image

def draw_keypoints(output, image, model):  
    """Draws keypoints on the image."""
    output = non_max_suppression_kpt(output, 
                                        0.25, # Confidence Threshold
                                        0.65, # IoU Threshold
                                        nc=model.yaml['nc'], # Number of Classes
                                        nkpt=model.yaml['nkpt'], # Number of Keypoints
                                        kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg

def load_video(file_path):
    """Loads a video file and returns the frames and fps."""
    cap = cv2.VideoCapture(file_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def process_frame_batch(args):
    """Processes a batch of frames."""
    frame, idx, model = args
    output, processed_frame = run_inference(frame, model)
    processed_frame = draw_keypoints(output, processed_frame, model)  # Pass the model here
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    return processed_frame

if __name__ == "__main__":
    test_file = "./tests/CaVa73_230528_LJ3_400.mov"
    weights = "./runs/train/01_rp_detection6/weights/best.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    model = load_model(device, weights)    
        
    frames, fps = load_video(test_file)
        
    frame_dataset = FrameDataset(frames)
    dataloader = DataLoader(frame_dataset, batch_size=batch_size)

    processed_frames = []
    
    for frame in frames:
        output, processed_frame = run_inference(frame, model)
        print(output)
        print(type(output))
        print(processed_frame)
        print(type(processed_frame))
        break