from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import os 



def read_files(fpath):
    refimg = Image.open(f'{fpath}/refimg.png') # original size, run resize_with_padding function to get 256x256
    tarimg = Image.open(f'{fpath}/tarimg.png')
    refpose, tarpose = np.load(f'{fpath}/refpose.npy'), np.load(f'{fpath}/tarpose.npy') # extrinsic matrices, 4x4
    refints, tarints = np.load(f'{fpath}/refints.npy'), np.load(f'{fpath}/tarints.npy') # intrinsic matrices, 3x3
    warpedimg, warpedmask = np.load(f'{fpath}/warpedimg.npy'), np.load(f'{fpath}/warpedmask.npy') # warped mask is [-1,-1,-1] where pixels were not copied over
    
    mask = (warpedmask[:, :, 0] == -1) & (warpedmask[:, :, 1] == -1) & (warpedmask[:, :, 2] == -1)
    coordinates = np.where(mask)
    maskcoords = list(zip(coordinates[0], coordinates[1])) # list of (row, col) pairs where value==-1 (i.e. pixels were not copied over)

    showmultimgs([refimg, tarimg, warpedimg]) # comment out if not visualizing in jupyter notebook

    return refimg, tarimg, refpose, tarpose, refints, tarints, warpedimg, warpedmask, maskcoords


def showmultimgs(imgs):
    fig, axs = plt.subplots(1, len(imgs), figsize=(50, 50))
    for idx, img in enumerate(imgs):
        axs[idx].imshow(img)
        axs[idx].axis('off')  