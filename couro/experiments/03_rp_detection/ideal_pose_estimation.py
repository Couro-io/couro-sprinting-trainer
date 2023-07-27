"""
Evaluate model performance.
"""

import os
import yaml
from tqdm import tqdm
from pprint import pprint
import sys
import tempfile
from urllib.parse import unquote
from tqdm import tqdm
import argparse

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils.datasets import letterbox, create_dataloader
from utils.general import non_max_suppression_kpt, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_images, output_to_target, plot_study_txt
from models.experimental import attempt_load
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel

from preprocessing import get_files_with_annotations

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
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--single-cls', default='False', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    
    # 0. Initialize test case
    test_frames_dir = "./data/processed/ipe/images"
    weights = "./runs/train/01_rp_detection6/weights/best.pt"
    device = select_device(opt.device, batch_size=opt.batch_size)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    model.eval()
    half_precision=True
    grid_size = max(int(model.stride.max()), 32)  
    img_size = check_img_size(opt.img_size, s=grid_size)
    
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    save_hybrid = False
    if half:
        model.half()
    
    data_file = "./config/01_init.yaml"
    with open(data_file) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    
    nc = int(data['nc'])
    dataloader = create_dataloader(path=test_frames_dir, imgsz=img_size, batch_size=opt.batch_size, stride=grid_size, opt=opt, pad=0.5, rect=True, prefix='')[0]
    
    # 1. Predict the running phases
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            out, train_out = model(img)
            
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, \
                                      conf_thres=0.25, \
                                      iou_thres=0.65, \
                                      labels=lb, \
                                      multi_label=True)
            print(targets)
            print(out)

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            pprint(boxes)
        
    
    # 2. Calculate joint angles for every frame
    
    # 3. Save them with true and predicted labels
    
    