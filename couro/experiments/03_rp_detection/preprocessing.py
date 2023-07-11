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
    x_center = (float(xbr) + float(xtl)) / 2
    y_center = (float(ybr) + float(ytl)) / 2
    return x_center, y_center

def get_key_from_value(dictionary, value):
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
        path_to_img:str, \
        save_prefix:str, \
        label_encoding_path:str='./data/processed/label_dict.json', \
        data_split:str='./data/processed/train/') -> List[str]:
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"
    
    image_filenames = []
    
    for xml in os.listdir(path_to_annotations):
        xml_path = os.path.join(path_to_annotations, xml)
        label_map = get_cvat_bbox(xml_path)
        filtered_map = remove_img_with_no_annot(label_map, logname)
        image_filenames.extend(extract_label(filtered_map, save_prefix, label_encoding_path, data_split))
    print(len(image_filenames))
    
    tmp_new_filenames = list()    
    for img in os.listdir(path_to_img):
        if img in image_filenames:
            new_filename = f"{save_prefix}_{img}"
            image = PIL.Image.open(os.path.join(path_to_img, img))
            image.save(os.path.join(data_split, 'images', new_filename))
            tmp_new_filenames.append(new_filename)
    print(len(tmp_new_filenames))

if __name__ == '__main__':
    """
    xml_path = "./annotations/train/annotations 1.xml"
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"
    label_dict = get_cvat_bbox(xml_path)
    annot_dict = remove_img_with_no_annot(label_dict, logname)
    pprint(annot_dict)
    """
    
    get_files_with_annotations(path_to_annotations="./annotations/validation/", path_to_img="./data/raw/validation/video 3", save_prefix="video3", data_split="./data/processed/validation/")
    

    
    


    