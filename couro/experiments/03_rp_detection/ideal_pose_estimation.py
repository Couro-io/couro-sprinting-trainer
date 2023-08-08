"""
Evaluate model performance.
"""

import os
import yaml
from tqdm import tqdm
from pprint import pprint
import sys
import math
import tempfile
import random
from urllib.parse import unquote
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

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

import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend for GUI display
import matplotlib.pyplot as plt

class FrameDataset(Dataset):
    """Dataloader for video data."""
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return frame, idx
    
def load_model(device, half, weights = "./runs/train/01_rp_detection6/weights/best.pt"):
    """Default model is yolo v7 pose estimation"""
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.eval()
    
    if half:
        model.half()
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

def get_xy_coords(kpts, steps=3):
    """
    """
    num_kpts = len(kpts) // steps
    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            return x_coord, y_coord

def process_frame_batch(args):
    """Processes a batch of frames."""
    frame, idx, model = args
    output, image = run_inference(frame, model)
    nms_output = non_max_suppression_kpt(output, 
                                            0.25, 
                                            0.65, 
                                            nc=model.yaml['nc'], 
                                            nkpt=model.yaml['nkpt'], 
                                            kpt_label=True)
    with torch.no_grad():
        kp_output = output_to_keypoint(nms_output)
    
    img = draw_keypoints(output, image, model)
    #plt.figure(figsize=(8,8))
    #plt.axis('off')
    #plt.imshow(img)
    #plt.show()

    kpt_xy_coords = list()    
    for idx in range(kp_output.shape[0]):
        x_coord, y_coord = get_xy_coords(kp_output[idx, 7:].T)
        kpt_xy_coords.append((x_coord, y_coord))
    return kpt_xy_coords

def draw_predicted_bbox(image, bbox_predictions):
    for bbox in bbox_predictions:
        batch_id, class_id, x, y, w, h, conf = bbox

        # Convert coordinates to integers
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        # Optionally, you can add text to label the bounding boxes with confidence or class_id
        # text = f"{conf:.2f}"  # Example: Show confidence with two decimal places
        # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

colors = [
    [255, 0, 0],     # Red
    [0, 255, 0],     # Green
    [0, 0, 255],     # Blue
    [255, 255, 0],   # Yellow
    [255, 0, 255],   # Magenta
    [0, 255, 255],   # Cyan
]

def plot_one_box(box, img, label=None, color=None, line_thickness=None):
    # Function to draw a single bounding box on the image
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    color = color or [random.randint(0, 255) for _ in range(3)]

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)

    return img

def plot_single_image(images, targets, paths=None, names=None, max_size=640, line_thickness=3):
    # Plot individual images with bounding boxes

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    annotated_images = []

    for i, img in enumerate(images):
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)

        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  
                    boxes[[0, 2]] *= w  
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  
                    boxes *= scale_factor

            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, img, label=label, color=color, line_thickness=line_thickness)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 chars
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=line_thickness - 1)[0]
            cv2.putText(img, label, (5, t_size[1] + 5), 0, line_thickness / 3, [220, 220, 220], thickness=line_thickness - 1,
                        lineType=cv2.LINE_AA)

        annotated_images.append(img)

    return annotated_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--single-cls', default='False', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    
    # 0. Initialize test case
    test_frames_dir = "./data/processed/ipe/images"
    device = select_device(opt.device, batch_size=opt.batch_size)
    half_precision=True
    half = device.type != 'cpu' and half_precision  
    model = load_model(device, half)

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    grid_size = max(int(model.stride.max()), 32)  
    img_size = check_img_size(opt.img_size, s=grid_size)

    save_hybrid = False

    data_file = "./config/01_init.yaml"
    with open(data_file) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    
    nc = int(data['nc'])
    dataloader = create_dataloader(path=test_frames_dir, imgsz=img_size, batch_size=opt.batch_size, stride=grid_size, opt=opt, pad=0.5, rect=True, prefix='')[0]
    
    ###########################################################################
    # 1. STAGE 1: Predict the running phases
    ###########################################################################

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
            boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
        pprint(boxes)

        annot_images = plot_single_image(img, targets=targets)
        print(annot_images)
        for i in annot_images:
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.imshow(i)
            plt.show()
        
        break
    """
    ###########################################################################
    # 2. STAGE 2: Run pose estimation
    ###########################################################################

    model_path="./models/baseline/yolov7-w6-pose.pt"
    model = torch.load(model_path, map_location=device)['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
        
    frames = list()
    for file in os.listdir(test_frames_dir):
        frames.append(cv2.imread(os.path.join(test_frames_dir, file)))
        
    frame_dataset = FrameDataset(frames)
    dataloader = DataLoader(frame_dataset, batch_size=opt.batch_size)
        
    pe_outputs = []
    
    with mp.Pool(processes=8) as pool:
        for frame_batch, idx_batch in tqdm(dataloader):
            processed_batch = pool.map(process_frame_batch, [(frame, idx, model) for frame, idx in zip(frame_batch, idx_batch)])
            pe_outputs.extend(processed_batch)
    print(pe_outputs)
                            
    # 3. Calculate joint angles for every frame
    
    
    # 4. Save them with true and predicted labels
    """

    