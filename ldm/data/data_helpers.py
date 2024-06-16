import cv2
import numpy as np
import random
import os 
import torch
from tqdm import tqdm
from sys import getsizeof
import glob
import re
import math

from torch.utils.data import Dataset

import ipdb

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # np array, HxWx3
    return np_to_torchfloat(img)

def load_img_and_transform(img_path, target_res):
    '''
    load and resize img then center crop to target_res x target_res (e.g. 512x512)
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # np array, HxWx3
    h, w = img.shape[:2]
    
    # determine which dimension is shorter for resizing image 
    if h < w:
        new_h = target_res
        ratio = new_h / h
        new_w = int(w * ratio)
    else:
        new_w = target_res
        ratio = new_w / w
        new_h = int(h * ratio)

    resized_img = cv2.resize(img, (new_w, new_h)) # resize is w then h (very confusing)

    # center crop
    x1 = (new_w - target_res) // 2
    y1 = (new_h - target_res) // 2
    x2 = x1 + target_res
    y2 = y1 + target_res
    cropped_img = resized_img[y1:y2, x1:x2]

    #print("img shape: ", cropped_img.shape, img.shape)
    assert cropped_img.shape[0] == target_res and cropped_img.shape[1] == target_res

    return np_to_torchfloat(cropped_img)


def params_to_K(params):
    assert len(params)==4 #focal length in x direction, focal length in y direction, principal point x coordinate, principal point y coordinate
    fx, fy, cx, cy = params
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def load_img_and_intrinsics(img_path, target_res, camera_params):
    '''
    1. load and resize img then center crop to target_res x target_res (e.g. 512x512)
    2. scale focal length and principal point by how much img is resized (e.g. if you take a (640,480) image with focal length 1000 and resize it to (320,240) (scaling by 0.5), then then focal length scales to 500)
    3. for center crop, principal point will just be in the center of the new image
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # np array, HxWx3
    h, w = img.shape[:2]
    
    # determine which dimension is shorter for resizing image 
    if h < w:
        new_h = target_res
        ratio = new_h / h
        new_w = int(w * ratio)
    else:
        new_w = target_res
        ratio = new_w / w
        new_h = int(h * ratio)

    resized_img = cv2.resize(img, (new_w, new_h)) # resize is w then h (very confusing)
    fx, fy = camera_params[0]*ratio, camera_params[1]*ratio # scale focal

    # center crop
    x1 = (new_w - target_res) // 2
    y1 = (new_h - target_res) // 2
    x2 = x1 + target_res
    y2 = y1 + target_res
    cropped_img = resized_img[y1:y2, x1:x2]

    #print("img shape: ", cropped_img.shape, img.shape)
    assert cropped_img.shape[0] == target_res and cropped_img.shape[1] == target_res
    
    cx, cy = target_res/2, target_res/2 # final principal point location
    
    K = params_to_K( [fx, fy, cx, cy] )

    return np_to_torchfloat(cropped_img), K #np_to_torchfloat(K)

def np_to_torchfloat(arr):
    assert type(arr) is np.ndarray 
    return torch.from_numpy(arr).float()

def normalize_tensor(tensor, to_zero_one, print_ratio=True, manual=None):
    
    if manual is None:
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
    else:
        min_val, max_val = manual
    
    # [0,1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    if print_ratio:
        print("normalized ratio -- min val, max val: ", min_val, max_val)

    if to_zero_one:
        return normalized_tensor
    else:
        # [-1,1]
        return 2 * normalized_tensor - 1
    
def normalize_poses(poses, to_zero_one=True, manual=None):
    assert len(poses[0])==2, 'each item in pose is a tuple with pos and dir'
    tmp_poses = []
    for p in poses:
        tmp_poses.append(p[0])
        tmp_poses.append(p[1])
    assert len(tmp_poses) == (len(poses)*2)

    tmp_poses = torch.stack(tmp_poses)
    #print("prev pose: ", self.pose_embeddings.shape, self.pose_embeddings.min(), self.pose_embeddings.max())

    tmp_poses = normalize_tensor(tmp_poses, to_zero_one=to_zero_one, manual=manual)
    #print("new pose: ", tmp_poses.shape, tmp_poses.min(), tmp_poses.max(), tmp_poses.mean())

    # change back to tuple pairs
    new_poses = []
    i=0
    while i < len(tmp_poses):
        new_poses.append( (tmp_poses[i], tmp_poses[i+1]) )
        i+=2
    
    return new_poses


    
def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF.""" 
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]) 
    xb = torch.reshape(
        (x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1]) 
    emb = torch.sin(torch.cat([xb, xb + np.pi / 2.], axis=-1)) 
    return torch.cat([x, emb], axis=-1)

def create_pose_embedding(rot_mat, trans_mat, K, pos_enc=False, target_res=512):
    ray_pos = -1 * rot_mat.T @ trans_mat
    ray_pos = np.tile(ray_pos, (target_res, target_res, 1))
    #ray_dir = rot_mat.T @ np.linalg.inv(K) @ np.array([x, y, 1]).T

    y, x = np.meshgrid(np.arange(target_res), np.arange(target_res))
    homogeneous_coords = np.stack((x, y, np.ones_like(x)), axis=-1)
    # Reshape to (height * width, 3), compute ray directions, and reshape back to (height, width, 3)
    ray_dir = (rot_mat.T @ np.linalg.inv(K) @ homogeneous_coords.reshape(-1, 3).T).T.reshape(target_res, target_res, 3)

    ray_pos = np_to_torchfloat(ray_pos)
    ray_dir = np_to_torchfloat(ray_dir)

    if pos_enc:
        pose_emb_pos = posenc_nerf(ray_pos, min_deg=0, max_deg=15)
        pose_emb_dir = posenc_nerf(ray_dir, min_deg=0, max_deg=8)
        pose_emb = torch.cat((pose_emb_pos, pose_emb_dir), dim=-1)
        return pose_emb
    else:
        return ray_pos, ray_dir

# def extract_number(s): # wrong function
#     return int(re.search(r'(\d+)', s).group(1))

def rotation_matrix_to_euler_angles(R):
    assert R.shape == (3, 3), "Matrix must be 3x3"
    
    # Calculate yaw (z-axis rotation)
    yaw = math.atan2(R[1, 0], R[0, 0])
    
    # Calculate pitch (y-axis rotation)
    pitch = math.asin(-R[2, 0])
    
    # Calculate roll (x-axis rotation)
    roll = math.atan2(R[2, 1], R[2, 2])
    
    # Convert the Euler angles from radians to degrees
    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)
    
    return yaw_deg, pitch_deg, roll_deg


def euler_angles_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    # Convert the Euler angles from degrees to radians
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    # Create the rotation matrix for the yaw (Z-axis rotation)
    R_z = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Create the rotation matrix for the pitch (Y-axis rotation)
    R_y = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])

    # Create the rotation matrix for the roll (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad), math.cos(roll_rad)]
    ])

    # Combine the rotation matrices in the order Z-Y-X
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2*(qy**2 + qz**2)
    R[0, 1] = 2*(qx*qy - qz*qw)
    R[0, 2] = 2*(qx*qz + qy*qw)
    R[1, 0] = 2*(qx*qy + qz*qw)
    R[1, 1] = 1 - 2*(qx**2 + qz**2)
    R[1, 2] = 2*(qy*qz - qx*qw)
    R[2, 0] = 2*(qx*qz - qy*qw)
    R[2, 1] = 2*(qy*qz + qx*qw)
    R[2, 2] = 1 - 2*(qx**2 + qy**2)
    
    return R

def read_imgtxt(txt_path):
    imgf = []
    count = 0
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            if count==0:
                img_info = [line]
                count+=1
            else:
                img_info.append(line)
                imgf.append(img_info)
                count=0
    return imgf




def rotation_z(azi):
    """Compute the rotation matrices for a batch of z-axis (azimuth) rotations."""
    azi = np.radians(azi)  # Convert to radians
    cos_azi = np.cos(azi)
    sin_azi = np.sin(azi)
    
    batch_size = len(azi)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = cos_azi
    R[:, 0, 1] = -sin_azi
    R[:, 1, 0] = sin_azi
    R[:, 1, 1] = cos_azi
    R[:, 2, 2] = 1
    return R

def rotation_y(ele):
    """Compute the rotation matrices for a batch of y-axis (elevation) rotations."""
    ele = np.radians(ele)  # Convert to radians
    cos_ele = np.cos(ele)
    sin_ele = np.sin(ele)
    
    batch_size = len(ele)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = cos_ele
    R[:, 0, 2] = sin_ele
    R[:, 1, 1] = 1
    R[:, 2, 0] = -sin_ele
    R[:, 2, 2] = cos_ele
    return R

def rotation_x(rol):
    """Compute the rotation matrices for a batch of x-axis (roll) rotations."""
    rol = np.radians(rol)  # Convert to radians
    cos_rol = np.cos(rol)
    sin_rol = np.sin(rol)
    
    batch_size = len(rol)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1] = cos_rol
    R[:, 1, 2] = -sin_rol
    R[:, 2, 1] = sin_rol
    R[:, 2, 2] = cos_rol
    return R

def angles_to_matrix(angles, axis):
    """Compute the rotation matrix from euler angles for a mini-batch"""

    # Convert angles from degrees to radians
    #angles = np.radians(angles)


    if axis=='z':
        return rotation_z(angles)    

    if axis=='y':
        return rotation_y(angles)    
    if axis=='x':
        return rotation_x(angles)    
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (np.cos(rol) * np.cos(azi) - np.sin(rol) * np.cos(ele) * np.sin(azi))[:, np.newaxis]
    element2 = (np.sin(rol) * np.cos(azi) + np.cos(rol) * np.cos(ele) * np.sin(azi))[:, np.newaxis]
    element3 = (np.sin(ele) * np.sin(azi))[:, np.newaxis]
    element4 = (-np.cos(rol) * np.sin(azi) - np.sin(rol) * np.cos(ele) * np.cos(azi))[:, np.newaxis]
    element5 = (-np.sin(rol) * np.sin(azi) + np.cos(rol) * np.cos(ele) * np.cos(azi))[:, np.newaxis]
    element6 = (np.sin(ele) * np.cos(azi))[:, np.newaxis]
    element7 = (np.sin(rol) * np.sin(ele))[:, np.newaxis]
    element8 = (-np.cos(rol) * np.sin(ele))[:, np.newaxis]
    element9 = (np.cos(ele))[:, np.newaxis]
    return np.concatenate((element1, element2, element3, element4, element5, element6, element7, element8, element9), axis=1)


def rotation_z(azi):
    """Compute the rotation matrices for a batch of z-axis (azimuth) rotations."""
    azi = np.radians(azi)  # Convert to radians
    cos_azi = np.cos(azi)
    sin_azi = np.sin(azi)
    
    batch_size = len(azi)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = cos_azi
    R[:, 0, 1] = -sin_azi
    R[:, 1, 0] = sin_azi
    R[:, 1, 1] = cos_azi
    R[:, 2, 2] = 1
    return R

def rotation_y(ele):
    """Compute the rotation matrices for a batch of y-axis (elevation) rotations."""
    ele = np.radians(ele)  # Convert to radians
    cos_ele = np.cos(ele)
    sin_ele = np.sin(ele)
    
    batch_size = len(ele)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = cos_ele
    R[:, 0, 2] = sin_ele
    R[:, 1, 1] = 1
    R[:, 2, 0] = -sin_ele
    R[:, 2, 2] = cos_ele
    return R

def rotation_x(rol):
    """Compute the rotation matrices for a batch of x-axis (roll) rotations."""
    rol = np.radians(rol)  # Convert to radians
    cos_rol = np.cos(rol)
    sin_rol = np.sin(rol)
    
    batch_size = len(rol)
    
    R = np.zeros((batch_size, 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1] = cos_rol
    R[:, 1, 2] = -sin_rol
    R[:, 2, 1] = sin_rol
    R[:, 2, 2] = cos_rol
    return R

def angles_to_matrix(angles, axis):
    """Compute the rotation matrix from euler angles for a mini-batch"""

    # Convert angles from degrees to radians
    #angles = np.radians(angles)


    if axis=='z':
        return rotation_z(angles)    

    if axis=='y':
        return rotation_y(angles)    
    if axis=='x':
        return rotation_x(angles)    
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (np.cos(rol) * np.cos(azi) - np.sin(rol) * np.cos(ele) * np.sin(azi))[:, np.newaxis]
    element2 = (np.sin(rol) * np.cos(azi) + np.cos(rol) * np.cos(ele) * np.sin(azi))[:, np.newaxis]
    element3 = (np.sin(ele) * np.sin(azi))[:, np.newaxis]
    element4 = (-np.cos(rol) * np.sin(azi) - np.sin(rol) * np.cos(ele) * np.cos(azi))[:, np.newaxis]
    element5 = (-np.sin(rol) * np.sin(azi) + np.cos(rol) * np.cos(ele) * np.cos(azi))[:, np.newaxis]
    element6 = (np.sin(ele) * np.cos(azi))[:, np.newaxis]
    element7 = (np.sin(rol) * np.sin(ele))[:, np.newaxis]
    element8 = (-np.cos(rol) * np.sin(ele))[:, np.newaxis]
    element9 = (np.cos(ele))[:, np.newaxis]
    return np.concatenate((element1, element2, element3, element4, element5, element6, element7, element8, element9), axis=1)
