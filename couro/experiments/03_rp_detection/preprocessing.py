"""
Data preprocessing for the RP detection experiment.
"""
import sys
from pprint import pprint
import os
import random
from typing import List, Tuple

import numpy as np
import torch

sys.path.append('./visualize')
from visualize import get_cvat_bbox

def calc_x_y_center(xtl:float, ytl:float, xbr:float, ybr:float) -> Tuple[float, float]:
    """
    Calculate the center of a bounding box.
    """
    x_center = (xbr + xtl) / 2
    y_center = (ybr + ytl) / 2
    return x_center, y_center

def extract_label(xml_path:str):
    """
    labels should be named for the corresponding video_img_num combination with the following format:
    class x_center y_center width height
    """
    pass

def extract_img():
    """
    """
    pass


    
def create_img_label_dataset():
    """
    """
    pass

if __name__ == '__main__':
    xml_path = "./annotations/train/annotations 1.xml"
    pprint(get_cvat_bbox(xml_path))

    
    


    