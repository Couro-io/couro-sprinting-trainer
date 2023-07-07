"""
Visualize annotation and predictions.
"""
import os
import sys
from pprint import pprint
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from lxml import etree
from tqdm import tqdm

import cv2

def get_cvat_bbox(xml_path:str) -> dict:
    """
    Uses an annotation file (xml) from CVAT to create a dictionary mapping the original image name and the bounding box coordinates."""
    root = etree.parse(xml_path).getroot()
    previous_image = None  

    label_map = dict()
    for element in root.iter():
        if element.tag == 'image':  
            previous_image = element  
        elif element.tag == 'box' and previous_image is not None:
            element.attrib['width'] = previous_image.attrib['width']
            element.attrib['height'] = previous_image.attrib['height']
            label_map[previous_image.attrib['name']] = element.attrib
            previous_image = None  
            
    return label_map

def draw_bbox_on_img(label_map:dict, image_path:str):
    """Returns an image with the bounding boxes drawn on it."""
    for key, value in label_map.items():
        image_filename = key
        label = value['label']
        xtl, ytl, xbr, ybr = map(float, [value['xtl'], value['ytl'], value['xbr'], value['ybr']])
        
        image = cv2.imread(os.path.join(image_path, image_filename))
        mask = np.zeros_like(image)
        x, y, w, h = int(xtl), int(ytl), int(xbr - xtl), int(ybr - ytl)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        result = cv2.bitwise_and(image, mask)
        alpha = 0.6
        annotated_image = cv2.addWeighted(image, 1 - alpha, result, alpha, 0)

        color = (0, 255, 0)  
        thickness = 2
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
        
        save_path = f"./annotations/qa/{image_filename.split('.')[0]}/{image_filename.split('.')[0]}_annotations.jpg"
        cv2.imwrite(save_path, annotated_image)
        
def get_single_video_label_count(label_map: dict) -> pd.DataFrame:
    """
    Get the label distribution from a single annotation file.
    """
    df = pd.DataFrame.from_dict(label_map, orient='index')
    return df['label'].value_counts().to_frame().reset_index()
        
def draw_barplot(df: pd.DataFrame, title:str=''):
    """
    Draw a barplot given a dataframe of the label distribution.
    """
    ax = sns.barplot(x='label', y='count',
                 data=df,
                 errwidth=0)
    ax.bar_label(ax.containers[0])
    plt.title(f"{title} label distribution", loc='left')
    plt.show()
    
def get_multi_video_label_count(annotation_dir:str):
    """
    Get the label distribution from multiple annotation files.
    """
    annotation_list = os.listdir(annotation_dir)
    df_list = list()
    
    for xml in annotation_list:
        xml_path = os.path.join(annotation_dir, xml)
        label_map = get_cvat_bbox(xml_path)
        df_list.append(pd.DataFrame.from_dict(label_map, orient='index'))
    
    df = pd.concat(df_list, ignore_index=True)
    return df['label'].value_counts().to_frame().reset_index()

def return_video1_pipe():
    """
    This is a test pipeline for a single video.
    """
    xml_path = './annotations/train/annotations 1.xml'
    image_path = './data/train/video 1/'
    label_map = get_cvat_bbox(xml_path)
    
    draw_bbox_on_img(label_map, image_path)
    df = get_single_video_label_count(label_map)
    draw_barplot(df, title='video 1')
            
            
if __name__ == "__main__":
    df = get_multi_video_label_count(annotation_dir='./annotations/validation/')
    draw_barplot(df, title='Validation set')