from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler

import glob
from tqdm import tqdm
from .data_helpers import *


def make_tranforms(image_transforms):
    if isinstance(image_transforms, ListConfig):
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    #image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        #print("self.captions, return paths, postprocess, default caption: ", self.captions, return_paths, postprocess, default_caption) #None False None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        #self.paths = []
        # for e in ext:
        #     self.paths.extend(sorted(list(self.root_dir.rglob(f"*img_gt.{e}"))))


        img_path = '/share/phoenix/nfs05/S8/gc492/nerfw/nerfw/results/phototourism/training_data'
        imgs = glob.glob( os.path.join(img_path, "*.png") ) # (512, 512, 3)
        #imgs = sorted(imgs, key=extract_number) # this line extracts nfs05 instead of 000.png :(
        imgs = [s.split("/")[-1].split('.png')[0] for s in imgs] # leave only 000, 001...2639
        imgs = sorted(imgs, key=lambda s: int(s)) 
        imgs = [os.path.join(img_path, i+'.png') for i in imgs]
        self.extrinsics = np.load( os.path.join(img_path, "extrinsics.npy") )

        # processed_images = []
        # processed_ext = []
        # skip = 11
        # i = 0
        # while i < len(imgs):
        #    #print("outer i: ", i)
        #    if (i // skip + 1) % 3 == 2:  # 3 to get the second of every third iteration; multiple of 3 to skip more
        #        #print("inner i: ", i)
        #        processed_images.extend(imgs[i:i+skip])
        #        processed_ext.extend(self.extrinsics[i:i+skip])
        #    i += skip  # Move to the next set of 11 images
   
        # self.extrinsics = processed_ext
        # imgs = processed_images

        imgs = imgs[11:22] + imgs[1331:1342]
        self.extrinsics = np.concatenate((self.extrinsics[11:22], self.extrinsics[1331:1342]))
        print("len dataset: ", len(imgs))
        assert len(imgs) == len(self.extrinsics)

        
        self.intrinsics = np.load( os.path.join(img_path, "intrinsics.npy") )
        self.intrinsics = np_to_torchfloat(self.intrinsics) # 3x3 matrix, take focal
        #self.imgs = [load_img(i) for i in imgs] 
        self.imgs = []
        focals = self.intrinsics[0,0], self.intrinsics[1,1]
        
        for i in imgs:
            im, K = load_img_and_intrinsics(img_path=i, target_res=256, camera_params=focals)
            self.imgs.append(im)
        self.intrinsics = np_to_torchfloat(K)

        
        

        #self.pose_embeddings = []
        self.poss, self.dirs = [], []
        with tqdm(self.extrinsics) as pbar:
            for idx, ext in enumerate(pbar):
                pbar.set_description("Files loaded: {}/{}".format(idx, len(self.extrinsics)))
                rot = ext[:3,:3]
                trans = ext[:3,3]
                #print("ext: ", ext, rot, trans)
                poses = create_pose_embedding(rot, trans, self.intrinsics, pos_enc=False, target_res=256)
                #self.pose_embeddings.append(poses)
                self.poss.append(poses[0])
                self.dirs.append(poses[1])
        #print("pose shape: ", self.pose_embeddings[0][0].shape, self.pose_embeddings[0][0].min(), self.pose_embeddings[0][0].max(), self.pose_embeddings[0][1].min(), self.pose_embeddings[0][1].max()) # 512,512,54
        # print("extrinsic: ", self.extrinsics[0])
        self.poss = torch.stack(self.poss)
        self.dirs = torch.stack(self.dirs)
        #print("min max: ", poss.min(), poss.max(), poss.mean(), dirs.min(), dirs.max(), dirs.mean())
        #self.pose_embeddings = normalize_poses(self.pose_embeddings, to_zero_one=False)

        #self.poss = normalize_tensor(self.poss, to_zero_one=False)
        #self.dirs = normalize_tensor(self.dirs, to_zero_one=False)


        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.imgs)

    def __getitem__(self, index):
        data = {}

        target_img = self.imgs[index]#.numpy()
        rays_pos = self.poss[index]#.numpy()
        rays_dir = self.dirs[index]#.numpy()

        target_img = target_img/127.5-1.0

        #im = self.process_im(Image.fromarray(target_img)) 
        data["image_target"] = target_img#.permute(2,0,1)
        #print("im shapes: ", target_img.shape, target_img.min(), target_img.max())

        #rays_pos = self.process_im(rays_pos)
        #rays_dir = self.process_im(rays_dir)
        rays_pos = posenc_nerf(rays_pos, min_deg=0, max_deg=4) #15 self.pos_enc[0]
        rays_dir = posenc_nerf(rays_dir, min_deg=0, max_deg=4) #8 self.pos_enc[1]
        pose = torch.cat((rays_pos, rays_dir), dim=-1)
        data["image_cond"] = pose #.permute(2,0,1) #, rays_dir
        #print("cond shapes: ", rays_pos.shape, rays_pos.permute(2,0,1).shape)

        data["txt"] = self.default_caption

        return data

    def process_im(self, im):
        #im = im.convert("RGB")
        return self.tform(im)

