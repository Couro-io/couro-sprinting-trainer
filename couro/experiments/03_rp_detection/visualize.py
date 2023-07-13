"""
Visualize the images and CVAT annotations both before and after data preprocessing.
"""

import os
import sys
import uuid
import logging
from pprint import pprint
from datetime import datetime
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns

from lxml import etree
from tqdm import tqdm

import cv2

##############################################################################################
# HELPER FUNCTIONS
##############################################################################################

def create_log_file(filename:str, data:list):
    """Creates a log file"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'./logs/{filename}_empty_keys.log',
        filemode='w'
    )

    # Save the list of empty keys as a log file
    logging.info("Empty keys: %s", data)

##############################################################################################
# Functions for CVAT annotations
##############################################################################################

def get_cvat_bbox(xml_path:str) -> dict:
    """
    Uses an annotation file (xml) from CVAT to create a dictionary mapping the original image name and the bounding box coordinates.
    """
    
    root = etree.parse(xml_path).getroot()

    label_map = {}
    for image_elem in root.findall('image'):
        file_name = image_elem.get('name')
        boxes = []
        for box_elem in image_elem.findall('box'):
            box_attr = {
                'label': box_elem.get('label'),
                'xtl': box_elem.get('xtl'),
                'ytl': box_elem.get('ytl'),
                'xbr': box_elem.get('xbr'),
                'ybr': box_elem.get('ybr'),
            }
            box_attr['img_width'] = image_elem.get('width')
            box_attr['img_height'] = image_elem.get('height')
            boxes.append(box_attr)
        label_map[file_name] = boxes
            
    return label_map

def remove_img_with_no_annot(label_map:dict, log_name:str) -> dict:
    """Remove images with no annotations from the label map."""
    filtered_dict = {key: value for key, value in label_map.items() if value}
    empty_keys = [key for key, value in label_map.items() if not value]
    create_log_file(log_name, empty_keys)
    return filtered_dict

def draw_cvat_bbox_on_img(label_map: dict, image_path: str):
    """
    Draws bounding boxes on an image based on the provided label map.
    Saves the annotated image with the bounding boxes.
    """
    for image_filename, boxes in label_map.items():
        image = cv2.imread(os.path.join(image_path, image_filename))

        for box in boxes:
            label = box['label']
            xtl, ytl, xbr, ybr = map(float, [box['xtl'], box['ytl'], box['xbr'], box['ybr']])
            x, y, w, h = int(xtl), int(ytl), int(xbr - xtl), int(ybr - ytl)

            # Draw the bounding box on the image
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

        # Save the annotated image
        save_dir = os.path.join("./data/qa/original/")
        save_path = os.path.join(save_dir, f"{image_filename.split('.')[0]}_CVAT_annotations.jpg")
        print(f"Image with CVAT bounding box saved as: {save_path}")
        cv2.imwrite(save_path, image)
        
##############################################################################################
# Functions for COCO annotations
##############################################################################################

def generate_sample(annotation_list:list, image_list:list, num_samples:int=5):
    """Generate a smaller sample of the dataset for testing purposes. Default is 5 images/annotation files."""
    N = len(annotation_list)
    sample_idx = random.sample(range(N), num_samples)
    sampled_annotation_list = sorted([annotation_list[i] for i in sample_idx])
    sampled_image_list = sorted([image_list[i] for i in sample_idx])
    return sampled_annotation_list, sampled_image_list

def get_coco_bboxes(image_path, bboxes, color=(0, 0, 255), thickness=2):
    """Draw the bounding boxes on the image using the COCO format."""
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    for bbox in bboxes:
        bbox = [float(element) for element in bbox]
        class_id, x_center, y_center, width, height = bbox

        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        x2 = int(x_center + (width / 2))
        y2 = int(y_center + (height / 2))

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    image_name = os.path.splitext(os.path.basename(image_path))[0] 
    output_path = f'./data/qa/processed/{image_name}_with_bboxes.jpg'
    cv2.imwrite(output_path, img)
    print(f"Image with COCO bounding boxes saved as: {output_path}")
    
def load_coco_annot_file(file_path:str) -> list:
    """Load the COCO annotation file as a nested list per bbox object"""
    try:        
        with open(file_path, 'r') as file:
            print(file_path)
            file_contents = []
            for line in file:
                line_elements = line.strip().split()
                if len(line_elements) > 0:
                    file_contents.append(line_elements)
            return file_contents
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []
    
def draw_preprocess_bbox_on_img(annotation_path:str='./data/processed/validation/labels', 
                                image_path:str='./data/processed/validation/images', 
                                num_img=None):
    """
    Draw the bbox on the image using the COCO format.
    """
    all_annotation_list = sorted(os.listdir(annotation_path))
    all_img_list = sorted(os.listdir(image_path))
    if num_img is not None:
        sampled_annotation_list, sampled_image_list = generate_sample(all_annotation_list, all_img_list, num_img)
    else:
        sampled_annotation_list = all_annotation_list
        sampled_image_list = all_img_list
    for i, annotation in enumerate(sampled_annotation_list):
        if annotation.endswith('.DS_Store'):
            print("Skipping .DS_Store file.")
            continue
        img = sampled_image_list[i]
        
        full_annotation_path = os.path.join(annotation_path, annotation)
        full_img_path = os.path.join(image_path, img)
        
        bbox = load_coco_annot_file(full_annotation_path)
        get_coco_bboxes(full_img_path, bbox)
        
##############################################################################################
# Functions for label distributions
##############################################################################################
        
def get_single_video_label_count(label_map: dict) -> pd.DataFrame:
    """Get the label distribution from a single annotation file."""
    data = []
    for filename, boxes in label_map.items():
        for box in boxes:
            label = box['label']
            data.append({'filename': filename, 'label': label})
    df = pd.DataFrame(data)
    return df['label'].value_counts().reset_index()
        
def draw_barplot(df: pd.DataFrame, title:str='', save_png_name:str='test.png'):
    """Draw a barplot given a dataframe of the label distribution."""
    order = ['Initial Swing', 'Midswing', 'Terminal Swing',  'Initial Contact', 'Midstance', 'Terminal Stance']
    plt.figure(figsize=(11, 8)) 
    ax = sns.barplot(x='label', y='count',
                 data=df,
                 order=order,
                 errwidth=0)
    ax.bar_label(ax.containers[0])
    ax.set_xticklabels(order)
    plt.title(f"{title} label distribution", loc='left')
    plt.savefig(f"./results/{save_png_name}", dpi=300, bbox_inches='tight')
    
def get_multi_video_label_count(annotation_dir:str):
    """Get the label distribution from multiple annotation files."""
    annotation_list = os.listdir(annotation_dir)
    df_list = list()
    current_date = datetime.now()
    unique_id = uuid.uuid4()
    logname = f"{current_date.strftime('%Y%m%d%H%M%S')}_{unique_id}"

    for xml in annotation_list:
        tmp = os.path.splitext(os.path.basename(xml))[0]
        xml_path = os.path.join(annotation_dir, tmp)
        label_map = get_cvat_bbox(xml_path)
        label_map = remove_img_with_no_annot(label_map, log_name=logname)
        data = []
        for filename, boxes in label_map.items():
            for box in boxes:
                label = box['label']
                data.append({'filename': filename, 'label': label})
        df = pd.DataFrame(data)
        df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    return df['label'].value_counts().to_frame().reset_index()

def return_video1_pipe():
    """This is a test pipeline for a single video."""
    xml_path = './annotations/validation/video1.xml'
    image_path = './data/raw/validation/video1'
    label_map = get_cvat_bbox(xml_path)
    label_map = remove_img_with_no_annot(label_map, log_name='video1_log.txt')
    
    draw_cvat_bbox_on_img(label_map, image_path)
    df = get_single_video_label_count(label_map)
    draw_barplot(df, title='video 1')
    
def get_train_val_label_dist():
    """Get the label distribution for the training and validation set."""
    df = get_multi_video_label_count(annotation_dir='./annotations/train/')
    draw_barplot(df, title='Training set')
    
    df = get_multi_video_label_count(annotation_dir='./annotations/validation/')
    draw_barplot(df, title='Validation set')
            
            
if __name__ == "__main__":
    #########################################################################################
    # Draw the bbox using the original CVAT annotation
    #########################################################################################
    xml_path = './annotations/validation/video3.xml'
    image_path = './data/raw/validation/video3'
    label_map = get_cvat_bbox(xml_path)
    label_map = remove_img_with_no_annot(label_map, log_name='video3_log.txt')
    draw_cvat_bbox_on_img(label_map, image_path)
    
    #########################################################################################
    # Draw the bbox on the image using the COCO format.
    #########################################################################################
    draw_preprocess_bbox_on_img(annotation_path='./data/processed/validation/labels', 
                                image_path='./data/processed/validation/images')
    