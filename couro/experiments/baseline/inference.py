import os
import sys
import tempfile
sys.path.append('./yolov7/')
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from urllib.parse import unquote
from tqdm import tqdm

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2
import numpy as np

import sagemaker
import boto3

def load_model(device, model_path:str="./yolov7/yolov7-w6-pose.pt"):
    """Default model is yolo v7 pose estimation"""
    model = torch.load(model_path, map_location=device)['model']
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

def run_inference(image):
    image = letterbox(image, 1920, stride=64, auto=True)[0] 
    image = transforms.ToTensor()(image) 
    if torch.cuda.is_available():
      image = image.half().to(device)
    image = image.unsqueeze(0) 
    with torch.no_grad():
      output, _ = model(image)
    return output, image

def draw_keypoints(output, image):
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

def pose_estimation_video(s3_url):
    bucket_name, object_key = extract_bucket_and_key(s3_url)
    temp_file_path = download_file_from_s3(bucket_name, object_key)
    
    folder_name = object_key.split('/')[-1].split('.')[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    frames = load_video(temp_file_path)

    processed_frames = []
    for idx, frame in tqdm(enumerate(frames), total=len(frames)):
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)
        cv2.imshow('Pose estimation', processed_frame)
        
        output_path = f'./{folder_name}/{folder_name}_{idx}.jpg'
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, processed_frame)
        print(f'Saved processed frame {idx} as {output_path}')
    
    cv2.destroyAllWindows()

    write_video_to_s3(processed_frames, bucket_name, object_key)

def extract_bucket_and_key(s3_url):
    parts = s3_url.split('//')[1].split('/')[0].split('.')
    bucket_name = parts[0]
    object_key = unquote('/'.join(s3_url.split('//')[1].split('/')[1:]))
    return bucket_name, object_key

def download_file_from_s3(bucket_name, object_key):
    s3 = boto3.client('s3')
    temp_file_path = tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_key)[1]).name
    s3.download_file(bucket_name, object_key, temp_file_path)
    return temp_file_path

def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def process_frame(frame, height:int=608, width:int=608):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output, processed_frame = run_inference(frame)
    processed_frame = draw_keypoints(output, processed_frame)
    processed_frame = cv2.resize(processed_frame, (height, width))
    return processed_frame

def write_video_to_s3(frames, bucket_name, object_key):
    temp_file_path = tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_key)[1]).name

    if object_key.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif object_key.endswith(".mov"):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:
        raise ValueError("Unsupported file extension.")

    frame_height, frame_width, _ = frames[0].shape
    out = cv2.VideoWriter(temp_file_path, fourcc, 30.0, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)
    out.release()

    s3 = boto3.client('s3')
    s3.upload_file(temp_file_path, bucket_name, object_key)

    os.remove(temp_file_path)
    
def frames_to_video(frames_directory, output_path):
    # Get a list of frame filenames in the directory
    frame_filenames = sorted(os.listdir(frames_directory))

    # Prepare video writer
    frame = cv2.imread(os.path.join(frames_directory, frame_filenames[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Write frames to video
    for i in range(len(frame_filenames)):
        frame_path = os.path.join(frames_directory, frame_filenames[i])
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer
    out.release()

if __name__ == "__main__":
    test_mov = "https://pose-estimation-db.s3.us-west-1.amazonaws.com/testuser%40test.com/CaVa73_230528_LJ3_400.mov"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    pose_estimation_video(test_mov)
    
    #test_dir = "./CaVa73_230528_LJ3_400"
    #output_path = "./CaVa73_230528_LJ3_400.mp4"
    #frames_to_video(test_dir, output_path)
    
