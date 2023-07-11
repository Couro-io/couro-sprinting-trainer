"""
Data preprocessing for the RP detection experiment.
"""
import sys
from pprint import pprint
from datetime import datetime
import uuid
import logging
import os
import random
from typing import List, Tuple
import json
import PIL

import numpy as np
import torch

sys.path.append('./visualize')
from visualize import get_cvat_bbox, remove_img_with_no_annot

def get_files_with_annotations(path_to_annotations:str) -> List[str]:
    """
    """
    pass

def create_data_splits(image_files:list, label_files:list, train_ratio:float=0.70, test_ratio:float=0.15, val_ratio:float=0.15):
    """
    """
    # Combine image and label file names into tuples
    data = list(zip(image_files, label_files))
    
    # Shuffle the data randomly
    random.shuffle(data)
    
    # Calculate the number of samples for each split
    total_samples = len(data)
    train_samples = round(total_samples * train_ratio)
    test_samples = round(total_samples * test_ratio)
    val_samples = round(total_samples * val_ratio)
    
    # Split the data into train, test, and validation sets
    train_data = data[:train_samples]
    test_data = data[train_samples:train_samples + test_samples]
    val_data = data[train_samples + test_samples:]
    
    # Unzip the data to obtain separate image and label lists for each split
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)
    val_images, val_labels = zip(*val_data)
    
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def calc_x_y_center(xtl:float, ytl:float, xbr:float, ybr:float) -> Tuple[float, float]:
    """
    Calculate the center of a bounding box.
    """
    x_center = (xbr + xtl) / 2
    y_center = (ybr + ytl) / 2
    return x_center, y_center

def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def extract_label(data: dict, save_prefix:str, label_encoding_path:str='./data/processed.json', data_split:str='./data/processed/train/'):
    """
    labels should be named for the corresponding video_img_num combination with the following format:
    class x_center y_center width height
    """
    
    with open(label_encoding_path, 'r') as file:
        label_encoding_dict = json.load(file)
    
    for image_filename, annotations in data.items():
        new_filename = f"{save_prefix}_{image_filename.split('.')[0]}.txt"
        with open(os.path.join(data_split, new_filename), 'w') as file:
            for annotation in annotations:
                x_center, y_center = calc_x_y_center(annotation['xtl'], annotation['ytl'], annotation['xbr'], annotation['ybr'])
                width = annotation['width']
                height = annotation['height']
                label = get_key_from_value(label_encoding_dict, annotation['label'])
                
                # Format the annotation as "class x_center y_center width height"
                formatted_annotation = f"{label} {x_center} {y_center} {width} {height}"
                
                # Write the formatted annotation to the file
                file.write(formatted_annotation + '\n')
                print(new_filename)
                print(formatted_annotation)

def rename_img(img_name:str, save_prefix:str, input_data:str="./data/raw/", output_data_split:str="./data/processed/train/"):
    """
    """
    image = PIL.Image.open(os.path.join(data_split, img_name))
    new_filename = f"{save_prefix}_{img_name.split('.')[0]}.jpg"
    
    
def create_img_label_dataset():
    """
    """
    pass

if __name__ == '__main__':
    xml_path = "./annotations/train/annotations 1.xml"
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"
    label_dict = get_cvat_bbox(xml_path)
    annot_dict = remove_img_with_no_annot(label_dict, logname)
    pprint(annot_dict)

    
    


    