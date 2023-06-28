"""
Inference.py
This script is used to run yolo v7 video inference on a video file and upload the results to S3 for a single user. 
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
import boto3

sys.path.append('./yolov7/')
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

def draw_keypoints(output, image, model):  # Add a model parameter here
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

def pose_estimation_video(s3_url, batch_size=8, num_processes=8):
    """Runs pose estimation on a video file and uploads the results to S3."""
    bucket_name, object_key = extract_bucket_and_key(s3_url)
    temp_file_path = download_file_from_s3(bucket_name, object_key)
    
    frames, fps = load_video(temp_file_path)
        
    frame_dataset = FrameDataset(frames)
    dataloader = DataLoader(frame_dataset, batch_size=batch_size)

    processed_frames = []
    
    with mp.Pool(processes=num_processes) as pool:
        for frame_batch, idx_batch in tqdm(dataloader):
            processed_batch = pool.map(process_frame_batch, [(frame, idx, model) for frame, idx in zip(frame_batch, idx_batch)])
            processed_frames.extend(processed_batch)

    write_video_to_s3(processed_frames, fps, bucket_name, object_key)

def extract_bucket_and_key(s3_url):
    """Extracts the bucket name and object key from an S3 URL."""
    parts = s3_url.split('//')[1].split('/')[0].split('.')
    bucket_name = parts[0]
    object_key = unquote('/'.join(s3_url.split('//')[1].split('/')[1:]))
    return bucket_name, object_key

def download_file_from_s3(bucket_name, object_key):
    """Downloads a file from S3 and returns the local file path."""
    s3 = boto3.client('s3')
    temp_file_path = tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_key)[1]).name
    s3.download_file(bucket_name, object_key, temp_file_path)
    return temp_file_path

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

def write_video_to_s3(frames, fps, bucket_name, object_key):
    """Writes a video file to S3."""
    temp_file_path = tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_key)[1]).name

    if object_key.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif object_key.endswith(".mov"):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:
        raise ValueError("Unsupported file extension.")

    frame_height, frame_width, _ = frames[0].shape
    out = cv2.VideoWriter(temp_file_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)
    out.release()

    base_name = os.path.splitext(object_key)[0]
    prediction_object_key = f"{base_name}_prediction.mp4"

    s3 = boto3.client('s3')
    s3.upload_file(temp_file_path, bucket_name, prediction_object_key)

    os.remove(temp_file_path)

if __name__ == "__main__":
    test_mov = "https://pose-estimation-db.s3.us-west-1.amazonaws.com/testuser%40test.com/CaVa73_230528_LJ3_400.mov"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    pose_estimation_video(test_mov)
