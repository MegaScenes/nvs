import cv2
import numpy as np
import random
import os 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sys import getsizeof
import glob
import re
import math
from datetime import datetime
import pytz
from torch.utils.data import Dataset
from PIL import Image
import torchvision

import ipdb

def save_images_as_grid(imgs, fixed_height=256, spacing=5, max_per_row=5):
    """
    Save a grid of images with a maximum number of images per row.

    :param imgs: List of NumPy images
    :param fixed_height: Fixed height for each image in the grid
    :param spacing: Space between images in pixels
    :param max_per_row: Maximum number of images per row
    """
    row_widths = []
    row_images = []
    current_row = []

    from PIL import Image
    # Process images and organize them into rows
    for np_img in imgs:
        img = Image.fromarray(np_img)
        aspect_ratio = img.width / img.height
        new_width = int(fixed_height * aspect_ratio)
        resized_img = img.resize((new_width, fixed_height))

        if len(current_row) < max_per_row:
            current_row.append(resized_img)
        else:
            row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
            row_images.append(current_row)
            current_row = [resized_img]

    # Add last row
    if current_row:
        row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
        row_images.append(current_row)

    total_width = max(row_widths)
    total_height = fixed_height * len(row_images) + spacing * (len(row_images) - 1)

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (total_width, total_height), color='white')

    # Paste each resized image into the grid
    y_offset = 0
    for row in row_images:
        x_offset = 0
        for img in row:
            grid_img.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
        y_offset += fixed_height + spacing

    # Return the grid image
    return grid_img

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

def load_img_and_intrinsics(img_path, target_res, camera_params, return_focal=False):
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

    if return_focal:
        return np_to_torchfloat(cropped_img), K, fx, fy

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

def create_pose_embedding(rot_mat, trans_mat, K, pos_enc=False, target_res=256):
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


'''
use the following instead in the future; current functions are zyx not xyz
from scipy.spatial.transform import Rotation as R
r = R.from_matrix(target_pose[:3,:3])
roll, pitch, yaw = r.as_euler('xyz', degrees=True) 
'''

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

def read_imgtxt(txt_path, read_point_info=True):
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
                if read_point_info:
                    img_info.append(line)
                imgf.append(img_info)
                count=0
    return imgf

def read_points3dtxt(txt_path):
    point_dict = {}
    count = 0
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            # line: '36068 -0.676148 0.318962 0.959069 51 32 25 0.15947 69 2935 24 7935 21 8053\n'
            vals = line.split()
            # X, Y, Z, R, G, B
            point_dict[int(vals[0])] = [float(vals[1]), float(vals[2]), float(vals[3]), int(vals[4]), int(vals[5]), int(vals[6])]
    
    return point_dict


def filename_to_image(img_txt):
    im_dict = {}
    for img in img_txt:
        img_info = img[0].split()
        filename = img_info[-1]
        im_dict[filename] = [float(item) for item in img_info[1:-2]] # QW, QX, QY, QZ, TX, TY, TZ
    return im_dict


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


def et_to_utc(filename):
    timestamp = filename.split('.')[0] # remove extension
    timestamp = timestamp.split(':')[0] + ":00:00" # only keep hour
    local_dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S") #datetime.strptime(frame["time"], "%Y-%m-%dT%H:%M:%S")
    local = pytz.timezone("America/New_York")    
    local_dt = local.localize(local_dt, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return str(utc_dt).split('+')[0] + ' +0000 UTC' # '1979-01-01 07:00:01 +0000 UTC' --> use as key to pandas df

def printinfo(*args):
    for a in args:
        print(a.shape, a.min(), a.max())

def clipval(value, min_value, max_value):
    return max(min_value, min(value, max_value))



# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# chatgpt, verify this later
def quaternion_to_euler(quaternion):
    """
    Convert a batch of quaternions to Euler angles
    """
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to a quaternion
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z])

# extrinsic matrix to spherical coordinates (in degrees)
def ext_to_sph(ext):
    if ext.ndim == 2:
        ext = ext.unsqueeze(0)
    trans = ext[:,:3,-1]
    r = torch.sqrt(torch.sum(trans**2, dim=-1))
    elevation = torch.arcsin( trans[:,2]/r )  / math.pi * 180.0 
    azimuth = torch.arctan2( trans[:,1], trans[:,0] ) / math.pi * 180.0 
    return elevation, azimuth, r

def compute_T(batch, precomputed_scale):
    batch_size = batch["cond_cam2world"].shape[0]
    precomputed_scale = precomputed_scale * torch.ones(
        size=(batch_size, 1),
        dtype=batch["cond_cam2world"].dtype,
        device=batch["cond_cam2world"].device,
    )

    scales = precomputed_scale

    # this functions returns E = inv(cond)@target (i.e. go from cond to target)
    # and scales the translation by 'scales'
    relative_transformations = get_relative_transformations(batch, scales)
    a = 0.5
    c = 10
    # we do a transformation of the translation. the intuition behind a and c is that
    # we want the transformation to be approximately linear for like 99% of batches
    # but when the translations are very large we want to smoothly ramp down to some
    # reasonable value rather than explode.

    # for small x, tanh(x)~~x, clamped at -1,1
    # c scales the input to the tanh function. A larger c makes the function behave linearly over a wider range of values (because it takes larger values of translation to get close to the saturation points of tanh).
    # so dividing by c before tanh and then multiplying by c almost makes the output the same, then a scales the result again
    # sanity check: the output of this line should be roughly 0.5x the original 
    relative_transformations[:, :3, -1] = (
        a * c * torch.tanh(relative_transformations[:, :3, -1] / c)
    )
    relative_transformations = torch.nan_to_num(
        relative_transformations, nan=0, posinf=c, neginf=-c
    )

    fov_rad = batch["fov_deg"] * np.pi / 180
    fov_enc = torch.stack(
        [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
    )
    relative_transformations = relative_transformations.reshape((batch_size, 16))

    #print("shapes: ", relative_transformations.shape, fov_enc.shape, fov_rad)

    T = torch.cat([relative_transformations, fov_enc], axis=1)

    return T
    
def get_relative_transformations(batch, scales):
    # this is the correct representation to use because it's invariant to
    # shift but not scale

    assert scales.ndim == 2

    target_cam2world = batch["target_cam2world"].detach().clone()
    cond_cam2world = batch["cond_cam2world"].detach().clone()

    batch_size = cond_cam2world.shape[0]

    relative_target_transformation = torch.linalg.inv(cond_cam2world) @ target_cam2world
    relative_target_transformation[:, :3, -1] /= torch.clip(scales, min=1e-2, max=None)

    assert relative_target_transformation.shape == (batch_size, 4, 4)
    return relative_target_transformation

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])

def extrinsics_to_matrix(ext):
    quat, trans = ext[0:4], ext[4:7]
    rot = quaternion_to_rotation_matrix(*quat)
    ext4x4 = np.eye(4)
    ext4x4[:3,:3] = rot
    ext4x4[:3,-1] = trans
    return ext4x4

def fxfy_from_intrinsics(intrinsics):
    parts = intrinsics
    model = parts[0]
    if model == 'SIMPLE_PINHOLE':
        _, width, height, focal_length, cx, cy = parts
        fx = fy = float(focal_length)
    elif model == 'PINHOLE':
        _, width, height, fx, fy, cx, cy = parts
        fx = float(fx)
        fy = float(fy)
    elif model == 'SIMPLE_RADIAL':
        _, width, height, focal_length, cx, cy, _ = parts  # Last part is radial distortion coefficient
        fx = fy = float(focal_length)

    return fx, fy

def fxfycxcy_from_intrinsics(intrinsics):
    parts = intrinsics
    model = parts[0]
    if model == 'SIMPLE_PINHOLE':
        _, width, height, focal_length, cx, cy = parts
        fx = fy = float(focal_length)
    elif model == 'PINHOLE':
        _, width, height, fx, fy, cx, cy = parts
        fx = float(fx)
        fy = float(fy)
    elif model == 'SIMPLE_RADIAL':
        _, width, height, focal_length, cx, cy, _ = parts  # Last part is radial distortion coefficient
        fx = fy = float(focal_length)

    return fx, fy, cx,cy

def fov_from_intrinsics(intrinsics):
    parts = intrinsics
    model = parts[0]
    if model == 'SIMPLE_PINHOLE':
        _, width, height, focal_length, cx, cy = parts
        fx = fy = float(focal_length)
    elif model == 'PINHOLE':
        _, width, height, fx, fy, cx, cy = parts
        fx = float(fx)
        fy = float(fy)
    elif model == 'SIMPLE_RADIAL':
        _, width, height, focal_length, cx, cy, _ = parts  # Last part is radial distortion coefficient
        fx = fy = float(focal_length)
        
    sensor_diagonal = math.sqrt(width**2 + height**2)
    diagonal_fov = 2 * math.atan(sensor_diagonal / (2 * fx)) # assuming fx = fy

    return diagonal_fov

def intrinsics_to_matrix(intrinsics):
    parts = intrinsics # .split()
    model = parts[0]

    # Initialize the intrinsic matrix as an identity matrix
    intrinsic_matrix = np.eye(3)

    if model == 'SIMPLE_PINHOLE':
        _, width, height, focal_length, cx, cy = parts
        fx = fy = float(focal_length)
    elif model == 'PINHOLE':
        _, width, height, fx, fy, cx, cy = parts
        fx = float(fx)
        fy = float(fy)
    elif model == 'SIMPLE_RADIAL':
        _, width, height, focal_length, cx, cy, _ = parts  # Last part is radial distortion coefficient
        fx = fy = float(focal_length)

    cx = float(cx)
    cy = float(cy)

    # Set the focal lengths and principal point
    intrinsic_matrix[0, 0] = fx  # fx
    intrinsic_matrix[1, 1] = fy  # fy
    intrinsic_matrix[0, 2] = cx  # cx
    intrinsic_matrix[1, 2] = cy  # cy

    return intrinsic_matrix



def resize_with_padding(img_path, target_size, black=False, grayscale=False, return_unpadded=False, returnpil=False, returnratio=False):
    """
    Resize an image to the target size, adding padding to the shorter side to maintain aspect ratio.

    :param img: img_path (str) or PIL image
    :param target_size: Tuple (width, height) specifying the target size
    :param  black: black or white background
    :return: PIL Image object of the resized image with padding
    """
    target_size = (target_size, target_size)
    # Open the image
    if isinstance(img_path,str):
        img = Image.open(img_path)
    else:
        img = img_path

    # Calculate the ratio of the target size and the original size
    width_ratio = target_size[0] / img.width
    height_ratio = target_size[1] / img.height
    ratio = min(width_ratio, height_ratio)

    # Calculate the new size keeping aspect ratio
    new_size = (int(img.width * ratio), int(img.height * ratio))
    if max(new_size)!=target_size[0]: # sometimes size is rounded down to 255
        x,y = new_size
        if x > y:
            new_size = (target_size[0], y)
        elif y > x:
            new_size = (x, target_size[0])
        else:
            new_size = (target_size[0],target_size[0])

    # Resize the image
    resized_img = img.resize(new_size, Image.LANCZOS)
    if return_unpadded:
        if returnpil:
            return resized_img
        return np.array(resized_img), ratio

    if grayscale:
        bg = 0 if black else 255
        new_img = Image.new("L", target_size, bg) # (H,W)
    else:
        bg = (0,0,0) if black else (255,255,255)
        new_img = Image.new("RGB", target_size, bg) # (H,W,3)

    # Get the position to paste the resized image
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)

    # Paste the resized image onto the white background
    new_img.paste(resized_img, paste_position)

    if returnpil:
        return new_img
    return np.array(new_img)


