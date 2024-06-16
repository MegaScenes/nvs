import cv2
import numpy as np
import random
import os, sys 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
import re
import math
from datetime import datetime
import pytz
from torch.utils.data import Dataset
from PIL import Image
import warnings

submodule_path = ( "/share/phoenix/nfs05/S8/gc492/scene_gen/Depth-Anything" )
assert os.path.exists(submodule_path)
sys.path.insert(0, submodule_path)
import depth_anything.dpt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

import ipdb


def load_depth_model():
    # load depth model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder = 'vitl' # can also be 'vitb' or 'vitl'
        depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).cuda().eval()
        dtransform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        return depth_model, dtransform


def invert_depth(depth_map):
    inv = depth_map.clone()
    # disparity_max = 1000
    disparity_min = 0.001
    # inv[inv > disparity_max] = disparity_max
    inv[inv < disparity_min] = disparity_min
    inv = 1.0 / inv
    return inv

    