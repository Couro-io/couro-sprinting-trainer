"""
Data preprocessing for the RP detection experiment.
"""
import sys
import shutil
from pprint import pprint
from datetime import datetime
import uuid
import logging
import os
import random
from typing import List, Tuple
import json
import PIL
from sklearn.model_selection import train_test_split

import numpy as np
import torch

sys.path.append('./visualize')
from visualize import get_cvat_bbox, remove_img_with_no_annot

label_encoding_path ='./data/processed/label_dict.json'
with open(label_encoding_path, 'r') as file:
    LABEL_ENCODING_DICT = json.load(file)

def calc_x_y_center(xtl:float, ytl:float, xbr:float, ybr:float) -> Tuple[float, float]:
    """Calculate the center of a bounding box."""
    x_center = (float(xbr) + float(xtl)) / 2
    y_center = (float(ybr) + float(ytl)) / 2
    return x_center, y_center

def get_key_from_value(dictionary, value) -> str:
    """Get the key from a dictionary based on a value"""
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def format_yolo_label_data(annotation:dict) -> str:
    """Format the annotation data for the YOLO model."""
    x_center, y_center = calc_x_y_center(annotation['xtl'], annotation['ytl'], annotation['xbr'], annotation['ybr'])
    width = annotation['width']
    height = annotation['height']
    label = get_key_from_value(LABEL_ENCODING_DICT, annotation['label'])
    formatted_annotation = f"{label} {x_center} {y_center} {width} {height}"
    return formatted_annotation

def extract_label(data: dict) -> dict:
    """
    labels should be named for the corresponding video_img_num combination with the following format:
    class x_center y_center width height
    """
    
    label_data = dict()
    for image_filename, annotations in data.items():
        for annotation in annotations:
            formatted_annotation = format_yolo_label_data(annotation, LABEL_ENCODING_DICT)
            label_data[image_filename] = formatted_annotation
                
    return label_data

def save_label_data(label_data:dict, video_name:str, path_to_save:str):
    """"""
    for image_filename, annotations in label_data.items():
        label_filename = os.path.join(path_to_save, f"{video_name}_{image_filename}.txt")
        with open(label_filename, 'w') as file:
            for annotation in annotations:
                file.write(annotation + '\n')

def create_new_image_name_map(filtered_filenames:list, video_name:str, path_to_orig_imgs:str, path_to_new_imgs:str) -> dict:
    """Create new image name mapping for the filtered images."""

    image_name_map = dict()
    video_dir = os.path.join(path_to_orig_imgs, video_name)    
    for orig_imgname in os.listdir(video_dir):
        if orig_imgname in filtered_filenames:
            original_path = os.path.join(video_dir, orig_imgname)
            new_path = os.path.join(path_to_new_imgs, 'images', f"{video_name}_{orig_imgname}")
            image_name_map[original_path] = new_path
    
    return image_name_map
    
def rename_image_file(image_name_map: dict):
    """Rename the image files using the filtered image names."""
    for orig_path, new_path in image_name_map.items():
        image = PIL.Image.open(orig_path)
        image.save(new_path)    

def get_files_with_annotations(path_to_annotations:str, \
        path_to_imgs:str, \
        label_encoding_path:str='./data/processed/label_dict.json', \
        data_split:str='./data/processed/train/') -> List[str]:
    """"""
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"
    
    label_filenames = list()
    img_filenames = list()
    for xml in os.listdir(path_to_annotations):
        xml_path = os.path.join(path_to_annotations, xml)
        label_map = get_cvat_bbox(xml_path)
        filtered_map = remove_img_with_no_annot(label_map, logname)
        label_data = extract_label(filtered_map)
        image_name_map = create_new_image_name_map(filtered_filenames=label_data.keys(), \
                                                   video_name=xml.split('.')[0], \
                                                   path_to_orig_imgs="", \
                                                   path_to_new_imgs="")
        
    
        
    return label_filenames, img_filenames

def split_train_test_files(train_ratio:float=0.8, parent_dir:str='./data/processed/train'):
    """
    """
    test_label_dir = './data/processed/test/labels'
    test_images_dir = './data/processed/test/images'
    
    label_files = os.listdir(os.path.join(parent_dir, 'labels'))
    image_files = os.listdir(os.path.join(parent_dir, 'images'))
    
    _, test_labels, _, test_images = train_test_split(
        image_files, label_files, train_size=train_ratio, random_state=42
    )
    
    for filename in test_labels:
        shutil.move(os.path.join(parent_dir, 'labels', filename), test_label_dir)
    for filename in test_images:
        shutil.move(os.path.join(parent_dir, 'images', filename), test_images_dir)

if __name__ == '__main__':
    
    
        
        
    label_filenames, img_filenames = get_files_with_annotations(path_to_annotations="./annotations/validation/", \
        path_to_imgs="./data/raw/validation/", \
        data_split="./data/processed/validation/")
    
    print(len(label_filenames))
    print(len(img_filenames))
    
    label_filenames, img_filenames = get_files_with_annotations(path_to_annotations="./annotations/train/", \
        path_to_imgs="./data/raw/train/", \
        data_split="./data/processed/train/")
    
    print(len(label_filenames))
    print(len(img_filenames))
    
    split_train_test_files()
    

    
    


    