import sys
sys.path.append('./yolov7/')
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    print("Successful import from inference.py")
