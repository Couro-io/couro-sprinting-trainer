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

def calc_x_y_center(xtl:float, ytl:float, xbr:float, ybr:float) -> Tuple[float, float]:
    """
    Calculate the center of a bounding box.
    """
    x_center = (float(xbr) + float(xtl)) / 2
    y_center = (float(ybr) + float(ytl)) / 2
    return x_center, y_center

def get_key_from_value(dictionary, value):
    """Get the key from a dictionary based on a value"""
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def extract_label(data: dict, save_prefix:str, label_encoding_path:str='./data/processed/label_dict.json.json', data_split:str='./data/processed/train/'):
    """
    labels should be named for the corresponding video_img_num combination with the following format:
    class x_center y_center width height
    """
    
    with open(label_encoding_path, 'r') as file:
        label_encoding_dict = json.load(file)
    
    image_filenames = []
    for image_filename, annotations in data.items():
        image_filenames.append(image_filename)
        filename_without_extension = image_filename.split('.')[0]
        new_filename = f"{save_prefix}_{filename_without_extension}.txt"  
        with open(os.path.join(data_split, 'labels', new_filename), 'w') as file:
            for annotation in annotations:
                x_center, y_center = calc_x_y_center(annotation['xtl'], annotation['ytl'], annotation['xbr'], annotation['ybr'])
                width = annotation['width']
                height = annotation['height']
                label = get_key_from_value(label_encoding_dict, annotation['label'])
                
                formatted_annotation = f"{label} {x_center} {y_center} {width} {height}"
                
                file.write(formatted_annotation + '\n')
    return image_filenames
    
def get_files_with_annotations(path_to_annotations:str, \
        path_to_imgs:str, \
        label_encoding_path:str='./data/processed/label_dict.json', \
        data_split:str='./data/processed/train/') -> List[str]:
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"
    
    label_filenames = list()
    img_filenames = list()
    for xml in os.listdir(path_to_annotations):
        xml_path = os.path.join(path_to_annotations, xml)
        label_map = get_cvat_bbox(xml_path)
        filtered_map = remove_img_with_no_annot(label_map, logname)
        label_filenames.extend(extract_label(filtered_map, xml.split('.')[0], label_encoding_path, data_split))
    
        for root, dirs, files in os.walk(path_to_imgs):
            for directory in dirs:
                if directory == xml.split('.')[0]:
                    video_dir = os.path.join(root, directory)    
                    for img in os.listdir(video_dir):
                        if img in label_filenames:
                            new_filename = f"{directory}_{img}"
                            image = PIL.Image.open(os.path.join(video_dir, img))
                            image.save(os.path.join(data_split, 'images', new_filename))
                            img_filenames.append(new_filename)
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
    

    
    


    