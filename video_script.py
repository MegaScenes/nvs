import math
import numpy as np
import time
import torch, torchvision
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import cv2
from torchvision import transforms
import os, json, sys
import matplotlib.pyplot as plt
import glob, ipdb
from tqdm import tqdm
import yaml
import pandas as pd
import importlib
from datetime import datetime
import pickle
import hashlib
from os.path import join
import warnings


from dataloader.data_helpers import *
from dataloader.depth_helpers import *
from dataloader.util_3dphoto import unproject_depth, render_view, render_multiviews 

from accelerate import Accelerator
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config
from ldm.logger import ImageLogger
from accelerate.utils import set_seed

from dataloader.evalhelpers import *

set_seed(23)
torch.backends.cudnn.benchmark = True


def setup_model():
    expdir = args.exp_dir
    config_file = yaml.safe_load(open( join(expdir, 'config.yaml') ))
    train_configs = config_file.get('training', {})
    img_logger = ImageLogger(log_directory=expdir, log_images_kwargs=train_configs['log_images_kwargs'])
    model = instantiate_from_config(config_file['model']).eval()
    accelerator = Accelerator()

    resume_folder = 'latest' if args.resume == -1 else f'iter_{args.resume}'
    args.resume = int(open(os.path.join(args.exp_dir, 'latest/iteration.txt'), "r").read()) if args.resume == -1 else args.resume
    print("loading from iteration {}".format(args.resume))

    if args.ckpt_file:
        old_state = torch.load(join(args.exp_dir, resume_folder, 'zeronvs.ckpt'), map_location="cpu")["state_dict"]
        model.load_state_dict(old_state)

    model = accelerator.prepare( model )
    
    if not args.ckpt_file:
        accelerator.load_state(join(args.exp_dir, resume_folder))
    return model, img_logger

def setup_paths():
    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)
    orbitpath = join(savepath, 'orbit')
    spiralpath = join(savepath, 'spiral')
    os.makedirs(orbitpath, exist_ok=True)
    os.makedirs(spiralpath, exist_ok=True)
    
def setup_scene_and_poses():
    # load img as 256x256
    inputimg = Image.open(args.inputimg)
    refimg = resize_with_padding(inputimg, target_size=256, returnpil=True)
    refimg_nopad = resize_with_padding(inputimg, target_size=256, return_unpadded=True, returnpil=True)
    refimg_nopad.save(join(args.savepath, 'reference.png'))
    refimg.save(join(args.savepath, 'reference_padded.png'))
    refimg_nopad = np.array(refimg_nopad)
    refimg = np.array(refimg)/ 255. # 0-1 for unproject_depth input

    # get depth
    inputimg = np.array(inputimg)/ 255. # use original resolution for depth estimation, but resize depth to refimg shape
    depth_model, dtransform = load_depth_model()
    h, w = refimg_nopad.shape[:2]
    img = dtransform({'image': inputimg})['image']
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    with torch.no_grad():
        est_disparity = depth_model(img).cpu()
        est_disparity = F.interpolate(est_disparity[None], (h, w), mode='bicubic', align_corners=False)[0, 0] # bicubic
    depthmap = invert_depth(est_disparity)
  
    
    # setup camera
    fx = fy = 80
    sensor_diagonal = math.sqrt(w**2 + h**2)
    fov = 2 * math.atan(sensor_diagonal / (2 * fx))
    fov = torch.tensor(fov)
    ext1 = np.eye(4)
    ext2 = ext1.copy()
    K = np.eye(3)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = w/2, h/2 # cx,cy is w/2,h/2, respectively

    depthmap = depthmap.numpy()
    scales = np.quantile( depthmap.reshape(-1), q=0.2 ) 
    depthmap = depthmap / scales
    depthmap = depthmap.clip(0,100)


    # get mesh
    plypath = join(args.savepath, 'mesh.ply')
    mesh = unproject_depth(plypath, refimg_nopad/255., depthmap, K, np.linalg.inv(ext1), scale_factor=1.0, add_faces=True, prune_edge_faces=True) # takes C2W

    # save warps
    def savewarps(warps):
        warppath = join(args.savepath, 'xtrans/warped')
        os.makedirs(warppath, exist_ok=True)
        warpedimgs = [(w*255).astype(np.uint8) for w in warps]
        pilframes = [Image.fromarray(f) for f in warpedimgs]
        pilframes[0].save(join(warppath,f'warps.gif'), save_all=True, append_images=pilframes[1:], loop=0, duration=100)


    # setup poses 
        # orbit poses
    orbitposes = get_orbit_poses()
    orbitwarps, _  = render_multiviews(h, w, K, orbitposes, mesh)

    orbit_rel_poses = []
    for ext2 in orbitposes: # change this!!!
        refpose = np.linalg.inv(ext1) # convert to c2w
        rel_pose = np.linalg.inv(refpose) @ ext2 # 4x4
      
        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        orbit_rel_poses.append(rel_pose)

        # spiral poses
    spiralposes = get_front_facing_trans(num_frames=20, max_trans=3, z_div=2)
    spiralwarps, _  = render_multiviews(h, w, K, spiralposes, mesh)

    spiral_rel_poses = []
    for ext2 in spiralposes: # change this!!!
        refpose = np.linalg.inv(ext1) # convert to c2w
        rel_pose = np.linalg.inv(refpose) @ ext2 # 4x4
        
        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        spiral_rel_poses.append(rel_pose)

    #ipdb.set_trace()
    return refimg, refimg_nopad, orbitwarps, orbit_rel_poses, spiralwarps, spiral_rel_poses


def save_outputs(refimg, outputs, warps, posetype): # posetype == 'orbit' or 'spiral'
    # 1. save grids of 6 imgs per viewpoint
    # 2. save grids of multiview imgs (6 grids corresponding to 6 generations per viewpoint, each grid has 5x4 imgs)
    # 3. save grids of the warped images 
    # 4. save gifs 

    viewgridpath = join(args.savepath, posetype, 'per_view_generations')
    os.makedirs(viewgridpath, exist_ok=True)
    for i, view in enumerate(outputs):    
        save_images_as_grid(view, max_per_row=args.batch_size).save(join(viewgridpath, f'{i}.png'))

    # save warps
    warppath = join(args.savepath, posetype, 'warped')
    os.makedirs(warppath, exist_ok=True)
    warpedimgs = [(w*255).astype(np.uint8) for w in warps]
    for i, w in enumerate(warpedimgs):
        Image.fromarray(w).save(join(warppath, f'{i}.png'))
    save_images_as_grid(warpedimgs, max_per_row=5).save(join(warppath, f'warpgrid.png'))
    pilframes = [Image.fromarray(f) for f in warpedimgs]
    pilframes[0].save(join(warppath,f'warps.gif'), save_all=True, append_images=pilframes[1:], loop=0, duration=100)


    # saves batchsize number of videos
    grid5x4path = join(args.savepath, posetype, 'videos')
    os.makedirs(grid5x4path, exist_ok=True)


    # from each view, pick the most consistent one by calculating MSE between original image mask and new image mask
    best_frames = []
    for i in range(len(outputs)):
        frames_sameviews = outputs[i]
        warpimg = warpedimgs[i]
        nonzeromask = np.any(warpimg != 0, axis=-1)
        refpixels = warpimg[nonzeromask]
        mse = []
        for f in frames_sameviews:
            if f.shape[:2] != refimg.shape[:2]:
                f = np.array(Image.fromarray(f).resize((refimg.shape[1], refimg.shape[0])))
            f = f[nonzeromask]
            mse.append(np.mean((f-refpixels)**2))
        bestidx = np.argmin(mse)
        best_frames.append(frames_sameviews[bestidx])
    save_images_as_grid(best_frames, max_per_row=5).save(join(grid5x4path, f'best.png'))
    pilframes = [Image.fromarray(f) for f in best_frames]
    pilframes[0].save(join(grid5x4path,f'best.gif'), save_all=True, append_images=pilframes[1:], loop=0, duration=100)




def main():

    if not args.debug:
        model, img_logger = setup_model()
    setup_paths()
    refimg, refimg_nopad, orbitwarps, orbit_rel_poses, spiralwarps, spiral_rel_poses = setup_scene_and_poses()

    h,w = refimg_nopad.shape[:2]
    shortside = min(h,w)
    diff = math.ceil((256-shortside)/2) # round up to avoid padding
    end = 256-diff

    batchsize = args.batch_size
    refimg = resize_with_padding(Image.fromarray((refimg*255).astype(np.uint8)), target_size=256)/127.5-1
    outputs = []


    # orbit poses
    for i in range(0, len(orbit_rel_poses)): 
        warpedimg = (orbitwarps[i]*255).astype(np.uint8)
        lwarp = resize_with_padding(Image.fromarray(warpedimg), target_size=32)/127.5-1
        highwarp = torch.tensor(resize_with_padding(Image.fromarray(warpedimg), target_size=256)/127.5-1).permute(2,0,1).unsqueeze(0).repeat(batchsize,1,1,1)
        rel_pose = orbit_rel_poses[i].unsqueeze(0).repeat(batchsize,1)
        
        if i==0:
            newdataloader = {}

        dataloader = dict(image_target=[np.zeros((256,256,3))-1.0], image_ref=[refimg], warped_depth=[lwarp], rel_pose=rel_pose) # txt=[""*batchsize], 
        if args.zeronvs:
            del dataloader['warped_depth']
        if args.warponly:
            del dataloader['rel_pose']

        for k in dataloader.keys():
            if k not in ['rel_pose'] :
                dataloader[k] = torch.tensor(dataloader[k][0]).float().unsqueeze(0).repeat(batchsize,1,1,1)

        # concatenate each key to newdataloader if exists, else set first value
        for k in dataloader.keys():
            if k in newdataloader:
                newdataloader[k] = torch.cat([newdataloader[k], dataloader[k]])
            else:
                newdataloader[k] = dataloader[k]

        #print(i, dataloader['rel_pose'].shape, newdataloader['rel_pose'].shape)
        
        if (i%args.repeat==0 and i!=0) or i==len(orbit_rel_poses)-1:  
            out = img_logger.log_img(model, newdataloader, args.resume, split='test', foldername='360', returngrid='train', warpeddepth=highwarp, has_target=False, onlyretimg=True)
            npout = ((out.permute(0,2,3,1).cpu().numpy()+1)*127.5).astype(np.uint8)
            if w<h:
                cropped = npout[:,:,diff:end,...] # if h larger, then padding was added to width, so crop width
            else:
                cropped = npout[:,diff:end,...]

            # append to outputs every batchsize multiple index
            for j in range(0, len(cropped), batchsize):
                outputs.append(cropped[j:j+batchsize])

            newdataloader = {}
        
            
    save_outputs(refimg_nopad, outputs, orbitwarps, 'orbit')


    # spiral poses
    outputs = []
    for i in range(0, len(spiral_rel_poses)): 
        warpedimg = (spiralwarps[i]*255).astype(np.uint8)
        lwarp = resize_with_padding(Image.fromarray(warpedimg), target_size=32)/127.5-1
        highwarp = torch.tensor(resize_with_padding(Image.fromarray(warpedimg), target_size=256)/127.5-1).permute(2,0,1).unsqueeze(0).repeat(batchsize,1,1,1)
        rel_pose = spiral_rel_poses[i].unsqueeze(0).repeat(batchsize,1)

        if i==0:
            newdataloader = {}
        
        dataloader = dict(image_target=[np.zeros((256,256,3))-1.0], image_ref=[refimg], warped_depth=[lwarp], rel_pose=rel_pose)
        
        if args.zeronvs:
            del dataloader['warped_depth']
        if args.warponly:
            del dataloader['rel_pose']


        for k in dataloader.keys():
            if k not in ['rel_pose'] :
                dataloader[k] = torch.tensor(dataloader[k][0]).float().unsqueeze(0).repeat(batchsize,1,1,1)

        for k in dataloader.keys():
            if k in newdataloader:
                newdataloader[k] = torch.cat([newdataloader[k], dataloader[k]])
            else:
                newdataloader[k] = dataloader[k]


        if (i%args.repeat==0 and i!=0) or i==len(spiral_rel_poses)-1:  
            out = img_logger.log_img(model, newdataloader, args.resume, split='test', foldername='360', returngrid='train', warpeddepth=highwarp, has_target=False, onlyretimg=True)
            npout = ((out.permute(0,2,3,1).cpu().numpy()+1)*127.5).astype(np.uint8)
            if w<h:
                cropped = npout[:,:,diff:end,...] # if h larger, then padding was added to width, so crop width
            else:
                cropped = npout[:,diff:end,...]

            # append to outputs every batchsize multiple index
            for j in range(0, len(cropped), batchsize):
                outputs.append(cropped[j:j+batchsize])

            newdataloader = {}
        
    save_outputs(refimg_nopad, outputs, spiralwarps, 'spiral')
        




if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="Directory for logging. Should include 'specs.yaml'",
    )
    arg_parser.add_argument(
        "--resume", "-r", required=True, type=int,
        help="continue from previous saved logs, integer value",
    )

    arg_parser.add_argument("--debug", "-d", action='store_true')

    arg_parser.add_argument("--savepath", "-s", required=True)
    arg_parser.add_argument("--inputimg", "-i", required=True)
    
    arg_parser.add_argument("--zeronvs", "-z", action='store_true')
    arg_parser.add_argument("--warponly", "-w", action='store_true')

    arg_parser.add_argument("--ckpt_file", action='store_true', help='if checkpoint file is .ckpt instead of safetensors')

    arg_parser.add_argument("--batch_size", "-b", default=9, type=int, help='effective batch size is batch_size*repeat; lower either if OOM')
    arg_parser.add_argument("--repeat", default=10, type=int, help='number of generations for each camera position')

    args = arg_parser.parse_args()

    print(args.savepath)

    main()