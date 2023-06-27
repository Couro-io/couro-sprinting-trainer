import os
import sys
import tempfile
sys.path.append('./yolov7/')
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from urllib.parse import unquote

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2
import numpy as np

import sagemaker
import boto3

def load_model(device, model_path:str="./yolov7/yolov7-w6-pose.pt"):
    model = torch.load(model_path, map_location=device)['model']
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
      image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
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

def pose_estimation_video(filename, obj):
    cap = cv2.VideoCapture()
    cap.open(filename)

    # VideoWriter for saving the video
    if filename.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif filename.endswith(".mov"):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:
        raise ValueError("Unsupported file extension.")
    out = cv2.VideoWriter(f"{os.path.splitext(filename)[0]}_output{os.path.splitext(filename)[1]}", fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(frame)
            frame = draw_keypoints(output, frame)
            frame = cv2.resize(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def load_video_from_s3(s3_url):
    # Extract bucket name and object key from the S3 URL
    bucket_name = s3_url.split('//')[1].split('/')[0].split('.')[0]
    object_key = unquote('/'.join(s3_url.split('//')[1].split('/')[1:]))

    # Create a boto3 S3 client
    s3 = boto3.client('s3')

    # Get the file as a stream from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_stream = response['Body']

    # Convert the stream to a numpy array
    file_bytes = file_stream.read()
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    return object_key, file_array

def write_video_to_s3(video_frames, bucket_name, object_key, output_format='.mp4'):
    # Create a temporary file to save the video locally
    temp_filename = tempfile.NamedTemporaryFile(suffix=output_format).name

    # Create a VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, 30.0, (video_width, video_height))

    # Write the video frames to the VideoWriter
    for frame in video_frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()

    # Upload the video file to S3
    s3 = boto3.client('s3')
    s3.upload_file(temp_filename, bucket_name, object_key + output_format)

    # Remove the temporary file
    os.remove(temp_filename)

if __name__ == "__main__":
    test_mov = "https://pose-estimation-db.s3.us-west-1.amazonaws.com/testuser%40test.com/test.mov"
    filename, obj = load_video_from_s3(test_mov)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    pose_estimation_video(filename, obj)