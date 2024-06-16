import json
import cv2
from PIL import Image
import numpy as np
import random
import os 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sys import getsizeof
import glob
from .data_helpers import *
from .depth_helpers import * 
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import ipdb
import math
import hashlib

class PairedDataset(Dataset):
    def __init__(self, target_res=256, split='train', pose_cond='warp_plus_pose'):
        
        # warp_plus_pose: scale translation based on 20th quantile of depth
        # inpainting: only use warped depth as pose condition, no additional pose condition for crossattn
        # zeronvs_baseline: no warped depth
        self.pose_cond = pose_cond
        self.split = split
        self.scaleclip = 1e-7
        self.target_res = target_res
        
        data_path =  'data/megascenes'
        self.highres_warped_path = join(data_path, 'warped_images')
        self.resized_image_path = join(data_path, 'resized_images')

        
        if split == 'train':
            with open('data/splits/trainsplit.pkl', 'rb') as f: 
                paired_images = pickle.load(f) # path1, path2, dict1, dict2, scale for path1 (dict keys: extrinsics, intrinsics)

        else:
            with open('data/splits/testsplit.pkl', 'rb') as f: 
                paired_images = pickle.load(f)
                # first 10000 are used for validation
                # 10000: are used for testing 
                paired_images = paired_images[10000:] 
        
        self.paired_images = paired_images


    def __len__(self):
        return len(self.paired_images)


    def convert_path_to_your_path(self, old_path):
        '''
        path 1, path2 in self.paired_images are absolute paths that should be converted to your local paths
        format of path1, path2: '/share/phoenix/nfs05/S8/jt664/WikiSFM/data/main/images/{category}/commons/{subcategory}/0/pictures/{imgname.png}'
        replace '/share/phoenix/nfs05/S8/jt664/WikiSFM/data/main/images/' with new image path

        if images in new image path are not resized to 256x256, resize them here
        '''
        return self.local_img_path + old_path.split('main/images/')[-1]


    def img_path_to_warped_path(self, path1, path2):
        '''
        get warped image given path1, path2;
        we used hashing to format the warped image
        '''
        cat = path1.split('main/images/')[-1].split('/commons')[0]
        subcat = path1.split('commons/')[-1].split('/0/pictures')[0]
        fname = path1.split('/')[-1][:-4]
        subcat2 = path2.split('commons/')[-1].split('/0/pictures')[0]
        fname2 = path2.split('/')[-1][:-4]
        warpname = join( cat, 'commons', subcat, "imgname:", fname, 'to', subcat2, "imgname:", fname2 )
        warpname = warpname.replace('/', '_')
        hash_object = hashlib.sha1(warpname.encode())
        hex_dig = hash_object.hexdigest() 
        return hex_dig+".png"


    def __getitem__(self, idx):

        if len(self.paired_images[idx])==4:
            path1, path2, dict1, dict2 = self.paired_images[idx]
        else:
            path1, path2, dict1, dict2, scales = self.paired_images[idx]
        try:
            img_ref = np.array(Image.open(self.convert_path_to_your_path(path1))) /127.5-1.0
            img_target = np.array(Image.open(self.convert_path_to_your_path(path2))) /127.5-1.0 # HxWx3
        except Exception as error:
            print("exception when loading image: ", error, path1, path2)
            img_ref = np.zeros((256,256,3)) -1.0
            img_target = np.zeros((256,256,3)) -1.0


        if self.pose_cond not in ['zeronvs_baseline'] or self.split!='train':
            warpname = self.img_path_to_warped_path(path1, path2)
            try:
                high_warped_depth = Image.open(join(self.highres_warped_path, warpname)) # original warped image with high resolution
                warped_depth = resize_with_padding(high_warped_depth, 32, black=False) /127.5-1.0 
                high_warped_depth = resize_with_padding(high_warped_depth, 256, black=False) /127.5-1.0 
            except Exception as error:
                print("exception when loading warped depth:", error, path1, path2, warpname)
                warped_depth = np.zeros((32,32,3)) -1.0
                high_warped_depth = np.zeros((256,256,3)) -1.0

        if self.pose_cond == 'inpainting':
            retdict = dict(image_target=img_target, image_ref=img_ref, warped_depth=warped_depth, highwarp=high_warped_depth)
            if self.split!='train':
                return retdict, idx
            return retdict


        if self.pose_cond == 'sdinpaint':
            mask = np.zeros_like(high_warped_depth)
            mask[np.all(high_warped_depth == [-1,-1,-1], axis=-1)] = [255,255,255]
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).convert("L")).astype(np.float32)/255.0
            mask = torch.tensor(mask).unsqueeze(0)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            #masked_image = torch.tensor(img_target).permute(2,0,1) * (mask < 0.5)
            masked_image = np.array(Image.open(warpname))
            masked_image[np.all(masked_image == [0,0,0], axis=-1)] = [127,127,127]
            masked_image = resize_with_padding(Image.fromarray(masked_image), 256, black=False) /127.5-1.0 
            #ipdb.set_trace()
            retdict = dict(image_target=img_target,  image_ref=img_ref, highwarp=high_warped_depth, mask=mask, masked_image=masked_image, txt="photograph of a beautiful scene, highest quality settings")
            if self.split!='train':
                return retdict, idx
            return retdict


        ext_ref = np.linalg.inv(extrinsics_to_matrix(dict1['extrinsics']))
        ext_target = np.linalg.inv(extrinsics_to_matrix(dict2['extrinsics'])) # using c2w, following zeronvs


        if self.pose_cond == 'zero123':
            tref = ext_ref[:3, -1] / np.clip(scales, a_min=self.scaleclip, a_max=None)
            ttarget = ext_target[:3, -1] / np.clip(scales, a_min=self.scaleclip, a_max=None)
            theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(tref[None, :])
            theta_target, azimuth_target, z_target = cartesian_to_spherical(ttarget[None, :])
            
            d_theta = theta_target - theta_cond
            d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
            d_z = z_target - z_cond
            
            d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()]).float()

            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=d_T, highwarp=high_warped_depth)
            if self.split!='train':
                return retdict, idx
            return retdict

        fov = torch.tensor(fov_from_intrinsics(dict2['intrinsics'])) # target fov, invariant to resizing
        rel_pose = np.linalg.inv(ext_ref) @ ext_target # 4x4

        # if self.pose_cond in ['warp_plus_pose', 'zeronvs_baseline']:
        #     depth_ref = np.load( self.imgname_to_depthname(path1) ) # HxW
        #     scales = np.quantile( depth_ref[::8, ::8].reshape(-1), q=0.2 ) 
        rel_pose[:3, -1] /= np.clip(scales, a_min=self.scaleclip, a_max=None) # scales preprocessed for faster data loading

        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        
        
        if self.pose_cond == 'warp_plus_pose':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose, warped_depth=warped_depth, highwarp=high_warped_depth)
        elif self.pose_cond == 'zeronvs_baseline':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose) # , highwarp=high_warped_depth
            if self.split!='train':
                retdict['highwarp'] = high_warped_depth
    
        if self.split!='train':
            return retdict, idx
        return retdict


