import os

import numpy as np
import torch
import torchvision
from PIL import Image
import random
import ipdb


class ImageLogger:
    def __init__(self, log_directory, no_recon_cond=True, only_sample_output=False, grid_input_sample=False, batch_frequency=2000, max_images=80, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_directory = log_directory
        self.no_recon_cond = no_recon_cond
        self.only_sample_output = only_sample_output

        self.grid_input_sample = grid_input_sample

    def log_local(self, split, images, epoch, foldername, returngrid, warpeddepth, has_target, returnimg, savegrid):
        if split == 'test':
            if foldername is None:
                root = os.path.join(self.log_directory, "testing_log", split)
            else:
                root = os.path.join(self.log_directory, "testing_log", foldername)
        else:
            root = os.path.join(self.log_directory, "image_log", split)
        
        if self.grid_input_sample or returngrid=='test':
            #first_two = torch.stack( (images['conditioning'][0], images['inputs'][0]) )  
            #grid = torch.cat( (first_two, images['samples']) )
            grid = torch.cat( ( images['conditioning'][0].unsqueeze(0), images['samples'],
                                images['conditioning'][0].unsqueeze(0), images["samples_cfg_scale_3.0"]
            ) )
            grid = torchvision.utils.make_grid(grid, nrow=len(images['samples'])+1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            if returngrid=='test':
                return [grid]

            filename = "grid-{}.png".format(epoch)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            try:
                Image.fromarray(grid).save(path)
            except:
                pass

        else:
            if savegrid:
                for k in images:
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    filename = "{}-e-{}.png".format(k, epoch)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    try:
                        Image.fromarray(grid).save(path)
                    except:
                        pass
            
            #ipdb.set_trace()
            if returngrid == 'train':
                grids = []
                batch = images['inputs'].shape[0]
                for idx in range(batch):

                    if has_target and warpeddepth is None:
                        stack = (images['conditioning'][idx], images['inputs'][idx], images["samples_cfg_scale_3.0"][idx])
                    elif has_target and warpeddepth is not None:
                        stack = (images['conditioning'][idx], images['inputs'][idx], warpeddepth[idx], images["samples_cfg_scale_3.0"][idx])
                    elif not has_target and warpeddepth is None:
                        stack = (images['conditioning'][idx], images["samples_cfg_scale_3.0"][idx])
                    elif not has_target and warpeddepth is not None:
                        stack = (images['conditioning'][idx], warpeddepth[idx], images["samples_cfg_scale_3.0"][idx])

                    grid = torchvision.utils.make_grid( torch.stack( stack  )  )
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grids.append(grid)
                    #grids.append( [images['conditioning'][idx], images['inputs'][idx], images['samples'][idx]] )

                if savegrid:
                    try:
                        filename = "grids-e-{}.png".format(epoch)
                        combined_grids = np.vstack(grids)
                        Image.fromarray(combined_grids).save(os.path.join(root, filename))
                    except:
                        pass

                if returnimg:
                    return grids, images["samples_cfg_scale_3.0"]
                return grids

        

    def log_img(self, model, batch, epoch, split="train", foldername=None, returngrid=False, warpeddepth=None, has_target=False, returnimg=False, savegrid=True, onlyretimg=False):

        model.eval()
        with torch.no_grad():
            images = model.log_images(batch, no_recon_cond=self.no_recon_cond, only_sample_output=self.only_sample_output, split=split, **self.log_images_kwargs) 
        model.train()

        #ipdb.set_trace()

        if onlyretimg:
            if "samples_cfg_scale_3.0" in images:
                imgs = images["samples_cfg_scale_3.0"]
            elif "samples_cfg_scale_7.5" in images:
                imgs = images["samples_cfg_scale_7.5"] # or just change to f"samples_cfg_scale_{unconditional_guidance_scale}"
            return torch.clamp(imgs, -1., 1.)

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
                    

        if warpeddepth is None:
            if 'highwarp' in batch:
                warpeddepth = batch['highwarp'].permute(0,3,1,2).detach().cpu() 
        return self.log_local(split, images, epoch, foldername, returngrid, warpeddepth, has_target, returnimg, savegrid)


