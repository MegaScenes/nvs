import wandb
import os
import ipdb
from tqdm import tqdm
import yaml
import numpy as np
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

from ldm.util import instantiate_from_config

from dataloader import *
from ldm.logger import ImageLogger
from accelerate.utils import set_seed

from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont

set_seed(23)
torch.backends.cudnn.benchmark = True

def main():
    # load data
    config_file = yaml.safe_load(open(os.path.join(args.exp_dir, 'config.yaml')))
    train_configs = config_file.get('training', {})

    dataset = PairedDataset(pose_cond=train_configs['pose_cond'], split='train')
    print("size of dataset: ", len(dataset))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, persistent_workers=(args.workers!=0)
    )


    img_logger = ImageLogger(log_directory=args.exp_dir, log_images_kwargs=train_configs['log_images_kwargs'])
  
    # load model
    model = instantiate_from_config(config_file['model'])
    model.train()
    model.learning_rate = float(train_configs.get('learning_rate', 1e-4))
    gradient_accumulation_steps = train_configs.get('gradient_accumulation_steps', 1)
    log_freq = 500
    save_freq = log_freq*2
    total_iterations = 150000
    optimizer, scheduler = model.configure_optimizers()

    
    # setup Accelerate 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=gradient_accumulation_steps, log_with="wandb") 
    mode = 'online' if args.log else 'disabled'
    accelerator.init_trackers(
        project_name=train_configs['project'], config=config_file,
        init_kwargs={ "wandb": { "name": train_configs['exp_name'], 'mode': mode } } #online
    )
    if args.resume is None:
        old_state = torch.load(args.finetune, map_location="cpu") 
        if "state_dict" in old_state:
            old_state = old_state["state_dict"]

        # Check if we need to port weights from 4ch input to 8ch for concatenating x with condition
        # also check cc_projection layer
        in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
        new_state = model.state_dict()
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        if in_filters_current.shape != in_filters_load.shape:
            input_keys = [ "model.diffusion_model.input_blocks.0.0.weight", "model_ema.diffusion_modelinput_blocks00weight" ]
            for input_key in input_keys:
                if input_key not in old_state or input_key not in new_state:
                    continue
                input_weight = new_state[input_key]
                if input_weight.size() != old_state[input_key].size():
                    dim2 = old_state[input_key].shape[1]
                    print(f"Manual init: {input_key}")
                    input_weight.zero_()
                    input_weight[:, :dim2, :, :].copy_(old_state[input_key])
                    old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

        proj_key = "cc_projection.weight" 
        if proj_key in new_state and proj_key in old_state:
            in_proj_load = old_state[proj_key]
            in_proj_current = new_state[proj_key]
            if in_proj_current.shape != in_proj_load.shape:
                proj_weight = new_state[proj_key]
                if proj_weight.size() != old_state[proj_key].size():
                    print(f"Manual init: {proj_key}")
                    proj_weight.zero_()
                    dim2 = old_state[proj_key].shape[1]
                    proj_weight[:, :dim2].copy_(old_state[proj_key][:, :dim2])
                    old_state[proj_key] = torch.nn.parameter.Parameter(proj_weight)

        model.load_state_dict(old_state, strict=False)
     
    model, dataloader, optimizer, scheduler = accelerator.prepare( model, dataloader, optimizer, scheduler )
    if args.resume is not None:
        resume_folder = 'latest' if args.resume == -1 else f'iter_{args.resume}'
        args.resume = int(open(os.path.join(args.exp_dir, 'latest/iteration.txt'), "r").read()) if args.resume == -1 else args.resume
        print("loading from iteration {}".format(args.resume))
        accelerator.load_state(os.path.join(args.exp_dir, resume_folder))

    module = model.module if isinstance(model, DistributedDataParallel) else model
    starting_iter = args.resume or 0
    num_processes = accelerator.num_processes
    print("number of acc processes: ", accelerator.num_processes)

    # start training loop
    progress_bar = tqdm(initial=starting_iter, total=total_iterations, disable=not accelerator.is_main_process)
    global_step = starting_iter
    local_step = 0
   
    while True:
        progress_bar.set_description(f"Training step {global_step}")
        for _, batch in enumerate(dataloader):
            if local_step == 0: # log image in the beginning for sanity check and comparisons
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
                    accelerator.log( {"train_table":log_image_table(grid)} )
                    
            if ( (local_step % log_freq == 0 and local_step != 0) or global_step==total_iterations ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if local_step % save_freq == 0:
                        accelerator.save_state(os.path.join(args.exp_dir, f"iter_{global_step}"))
                    else:
                        accelerator.save_state(os.path.join(args.exp_dir, f"latest")) 
                        with open(os.path.join(args.exp_dir, "latest/iteration.txt"), "w") as f:
                            f.write(str(global_step))
                        
                    grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
                    accelerator.log( {"train_table":log_image_table(grid)} )

            with accelerator.accumulate(model):
                loss, loss_dict = model(batch)
                progress_bar.set_postfix(loss_dict)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()    
                optimizer.zero_grad()
                progress_bar.update(num_processes)
                global_step += num_processes
                local_step += 1

                lr = optimizer.param_groups[0]['lr']
                loss_dict.update({'lr': lr}) 

            if args.log and local_step%10==0 and accelerator.is_main_process:
                accelerator.log(loss_dict, step=global_step)

            if global_step >= total_iterations:
                accelerator.end_training()
                print("Training complete!")
                return

    


def log_image_table(grid, test=False):
    column = "cond/target/sample" if not test else "cond/samples"
    table = wandb.Table(columns=[column])
    for g in grid:
        table.add_data(wandb.Image(g))
    return table
    #wandb.log({"train_table":table}, commit=False)



if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="Directory for logging. Should include 'specs.yaml'",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None, type=int,
        help="continue from previous saved logs, integer value",
    )

    arg_parser.add_argument(
        "--finetune", "-f", default="configs/zeronvs_original/iter_0/zeronvs.ckpt", type=str, 
        help="pretrained model",
    )

    arg_parser.add_argument("--log", "-l", action='store_true', help='logs to wandb')
    arg_parser.add_argument("--batch_size", "-b", default=64, type=int)
    arg_parser.add_argument("--workers", "-w", default=8, type=int)

    args = arg_parser.parse_args()
   
    main()

    
