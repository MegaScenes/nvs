import numpy as np 
import os, json, sys
import matplotlib.pyplot as plt

submodule_path = ( "/share/phoenix/nfs05/S8/gc492/scene_gen/Depth-Anything" )
assert os.path.exists(submodule_path)
sys.path.insert(0, submodule_path)
import depth_anything.dpt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
import warnings

from PIL import Image

def get_x_poses(num_steps=6, radius=1, endpoint=2, w2c=True):
    extrinsics = []

    for step in range(num_steps):

        x = -endpoint + endpoint*2/(num_steps-1)*step
        t = np.array([x, 0, 0]) # z=radius
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, 3] = -t
        if not w2c:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
        

    return extrinsics

def get_ccw_poses(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0,-1)):
    extrinsics = []
    a,b = endpoint
    for step in range(num_steps):
        theta = np.radians(rotation_angle) * step / num_steps

        # Rotation matrix (around the y-axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Translation vector
        x = a + (b - a) * step / (num_steps-1)
        #x = radius *xscale* np.cos(theta)
        t = np.array([x, 0, radius * zscale * np.sin(theta)])
        #print(t)

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics
    
def get_cw_poses(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0,1)):
    extrinsics = []
    a, b = endpoint
    for step in range(num_steps):
        # Negative theta for clockwise rotation
        theta = -np.radians(rotation_angle) * step / num_steps

        # Rotation matrix (around the y-axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Translation vector
        x = a + (b - a) * step / (num_steps - 1)
        t = np.array([x, 0, radius * zscale * np.sin(theta)])

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t  # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics
def get_orbit_poses(cwx=(0,1), ccwx=(0,-1)):
    orbitposes = []
    cwposes = get_cw_poses(endpoint=cwx)
    cwposes.reverse()
    ccwposes = get_ccw_poses(endpoint=ccwx)
    orbitposes.extend(cwposes[:-1])
    orbitposes.extend(ccwposes)
    return orbitposes
# def get_orbit_poses(num_steps=20, radius=1, total_angle=60, speed_factor=2, w2c=True):
#     """
#     Generate extrinsic matrices for a camera moving in a smooth orbit around an object,
#     with monotonically changing translation and adjustable speed.

#     :param num_steps: Number of steps in the camera path.
#     :param radius: Radius of the circle on which the camera moves.
#     :param total_angle: Total angle of the orbit in degrees (30 degrees each way from the origin).
#     :param speed_factor: Factor to control the speed of translation. Greater than 1 for faster, less than 1 for slower.
#     :return: List of extrinsic matrices (4x4 numpy arrays).
#     """
#     extrinsics = []
#     angle_increment = np.radians(total_angle) / (num_steps - 1)  # Adjusted angle step for each pose

#     for step in range(num_steps):
#         # Calculate the current angle
#         theta = -np.radians(total_angle / 2) + angle_increment * step

#         # Linear interpolation for x-coordinate
#         x = -radius + (2 * radius) * (step / (num_steps - 1))
#         x*=speed_factor

#         z = radius * np.cos(theta) - radius

#         t = np.array([x, 0, z])
#         #print(t)

#         # Rotation matrix (around the y-axis)
#         R = np.array([
#             [np.cos(theta), 0, np.sin(theta)],
#             [0, 1, 0],
#             [-np.sin(theta), 0, np.cos(theta)]
#         ])

#         # Create extrinsic matrix (4x4)
#         extrinsic_matrix = np.eye(4)
#         extrinsic_matrix[:3, :3] = R
#         extrinsic_matrix[:3, 3] = -t

#         if not w2c:
#             extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
#         extrinsics.append(extrinsic_matrix)

#     return extrinsics

    # if ccw:
    #     extrinsics = []
    #     theta_increment = np.radians(rotation_angle) / (num_steps * speed_factor)  # Increase speed_factor to slow down
    #     for step in range(num_steps):
    #         theta = theta_increment * step #  np.radians(rotation_angle) * step / num_steps
    
    #         # Rotation matrix (around the y-axis)
    #         R = np.array([
    #             [np.cos(theta), 0, np.sin(theta)],
    #             [0, 1, 0],
    #             [-np.sin(theta), 0, np.cos(theta)]
    #         ])
    
    #         # Translation vector
    #         t = np.array([radius * np.cos(theta), 0, radius * np.sin(theta)])
    
    #         # Create extrinsic matrix (4x4)
    #         extrinsic_matrix = np.eye(4)
    #         extrinsic_matrix[:3, :3] = R
    #         extrinsic_matrix[:3, 3] = -R @ t
    
    #         if c2w:
    #             extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    #         extrinsics.append(extrinsic_matrix)
    # else:
    #     extrinsics = []
    #     theta_increment = np.radians(rotation_angle) / (num_steps * speed_factor)  # Increase speed_factor to slow down
    #     for step in range(num_steps):
    #         theta = theta_increment * step #  np.radians(rotation_angle) * step / num_steps
    
    #         # Rotation matrix (around the y-axis)
    #         # R = np.array([
    #         #     [np.cos(theta), 0, np.sin(theta)],
    #         #     [0, 1, 0],
    #         #     [-np.sin(theta), 0, np.cos(theta)]
    #         # ])
    #         R = np.array([
    #             [np.cos(theta), 0, -np.sin(theta)],
    #             [0, 1, 0],
    #             [np.sin(theta), 0, np.cos(theta)]
    #         ])
    
    #         # Translation vector
    #         t = np.array([-radius * np.cos(theta), 0, radius * np.sin(theta)])
    
    #         # Create extrinsic matrix (4x4)
    #         extrinsic_matrix = np.eye(4)
    #         extrinsic_matrix[:3, :3] = R
    #         extrinsic_matrix[:3, 3] = -R @ t
    
    #         if c2w:
    #             extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    #         extrinsics.append(extrinsic_matrix)

    # return extrinsics

def get_front_facing_trans(num_frames, max_trans=2.0, c2w=True, z_div=2.0):
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /4.0 #* 3.0 / 4.0
        z_trans = -max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames)) / z_div

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                # [np.eye(3), np.array([x_trans, 0., 0.])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)[np.newaxis, :, :][0]

        if c2w:
            i_pose = np.linalg.inv(i_pose)
        output_poses.append(i_pose)

    return output_poses

def showbatch(imgs):
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    #for idx, img in enumerate(imgs):
    axs[0,0].imshow(imgs[0])
    axs[0,0].axis('off')  
    axs[0,1].imshow(imgs[1])
    axs[0,1].axis('off')  
    axs[0,2].imshow(imgs[2])
    axs[0,2].axis('off')  
    axs[1,0].imshow(imgs[3])
    axs[1,0].axis('off')  
    axs[1,1].imshow(imgs[4])
    axs[1,1].axis('off')  
    axs[1,2].imshow(imgs[5])
    axs[1,2].axis('off')  
    axs[2,0].imshow(imgs[6])
    axs[2,0].axis('off')  
    axs[2,1].imshow(imgs[7])
    axs[2,1].axis('off')  
    axs[2,2].imshow(imgs[8])
    axs[2,2].axis('off')  

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

# def invert_depth(depth_map):
#     inv = depth_map.copy()
#     disparity_max = 1000
#     disparity_min = 0.001
#     inv[inv > disparity_max] = disparity_max
#     inv[inv < disparity_min] = disparity_min
#     inv = 1.0 / inv
#     return inv

def invert_depth(depth_map):
    inv = depth_map.clone()
    # disparity_max = 1000
    disparity_min = 0.001
    # inv[inv > disparity_max] = disparity_max
    inv[inv < disparity_min] = disparity_min
    inv = 1.0 / inv
    return inv

# def save_images_as_grid(imgs, fixed_height=256, spacing=5):
#     """
#     Save a grid of images with the same height and a spacing between them, expanding horizontally.

#     :param imgs: List of NumPy images
#     :param save_path: Path to save the image
#     :param fixed_height: Fixed height for each image in the grid
#     :param spacing: Space between images in pixels
#     """
#     total_width = 0
#     resized_images = []

#     # Resize each image and calculate total width with spacing
#     for idx, np_img in enumerate(imgs):
#         # if idx == len(imgs)-1 or idx==len(imgs)-2: # for saving depth maps 
#         #     depth_map = np_img
#         #     normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
#         #     scaled_depth = (255 * normalized_depth).astype(np.uint8)
#         #     img = Image.fromarray(scaled_depth, 'L')  # 'L' mode for grayscale
#         # else:
#         img = Image.fromarray(np_img)
#         aspect_ratio = img.width / img.height
#         new_width = int(fixed_height * aspect_ratio)
#         resized_img = img.resize((new_width, fixed_height))
#         resized_images.append(resized_img)
#         total_width += new_width + spacing

#     total_width -= spacing  # Remove extra spacing at the end

#     # Create a new blank image with a white background
#     grid_img = Image.new('RGB', (total_width, fixed_height), color='white')

#     # Paste each resized image into the grid with spacing
#     x_offset = 0
#     for img in resized_images:
#         grid_img.paste(img, (x_offset, 0))
#         x_offset += img.width + spacing

#     # Save the grid image
#     return grid_img #.save(save_path)

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