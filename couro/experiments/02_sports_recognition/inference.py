import torch
from torchvision.io import read_video
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer, inference_recognizer


