import wandb
import os
from os.path import join
import ipdb
from tqdm import tqdm
import yaml
import numpy as np
from datetime import timedelta
from datetime import datetime
import glob
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

from ldm.util import instantiate_from_config
from safetensors.torch import load_file

from dataloader import *
from ldm.logger import ImageLogger
from accelerate.utils import set_seed
from evaluation.recon_metrics import ReconMetric
from evaluation.gen_metrics import GenMetric

set_seed(42)
torch.backends.cudnn.benchmark = True


def main():
    # load data
    config_file = yaml.safe_load(open(os.path.join(args.exp_dir, 'config.yaml')))
    train_configs = config_file.get('training', {})
    dataset_name = args.dataset 
    if dataset_name == 'dtu':
        dataset = DTUDataset(pose_cond=train_configs['pose_cond'])
    elif dataset_name == 're10k':
        dataset = Re10kDataset(pose_cond=train_configs['pose_cond'])
    elif dataset_name == 'mipnerf':
        dataset = MipnerfDataset(pose_cond=train_configs['pose_cond'])
    elif dataset_name == 'megascenes':
        dataset = PairedDataset(pose_cond=train_configs['pose_cond'], split='test')
    else:
        print("check dataset")
        exit()

    print("size of dataset: ", len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        drop_last=False, shuffle=False, persistent_workers=False, pin_memory=False
    )

    img_logger = ImageLogger(log_directory=args.exp_dir, log_images_kwargs=train_configs['log_images_kwargs'])

    model = instantiate_from_config(config_file['model']).eval()  
    
    accelerator = Accelerator()
    if args.resume !=0: # if == 0: loading from pretrained checkpoints (e.g. zeronvs, zero123)
        model, dataloader = accelerator.prepare( model, dataloader )
  
    if args.resume is None: # runs evaluation on all checkpoints
        saved_checkpoints = glob.glob( join(args.exp_dir, 'iter*') )

        saved_checkpoints = sorted(saved_checkpoints, key=lambda x: int(x.split('iter_')[-1]))
        saved_checkpoints.reverse()
        saved_checkpoints =[saved_checkpoints[0]]
    else:
        saved_checkpoints = [join(args.exp_dir, f'iter_{args.resume}')]
    print(saved_checkpoints)

    savepath = join('quant_eval', args.savepath)
    os.makedirs(savepath, exist_ok=True)

    countgen = 0
    countdata = 0
    with torch.no_grad(): 
        for idx, ckpt in enumerate(saved_checkpoints):
            if args.save_generations:
                os.makedirs(join(savepath, 'generations'), exist_ok=True)
            if args.save_data:
                os.makedirs(join(savepath, 'refimgs'), exist_ok=True)
                os.makedirs(join(savepath, 'tarimgs'), exist_ok=True)
                os.makedirs(join(savepath, 'masks'), exist_ok=True)

            
            resume = int(ckpt.split('iter_')[-1])
            print("loading from iteration {}".format(resume))
            if args.resume !=0:
                accelerator.load_state(ckpt)
            else:
                old_state = torch.load(join(ckpt, args.released_ckpt), map_location="cpu")
                if "state_dict" in old_state:
                    old_state = old_state["state_dict"]
                model.load_state_dict(old_state)
                model, dataloader = accelerator.prepare( model, dataloader )

            kid_subset = min(1000, len(dataset)-1) # 1000 is default
            print("kid subset: ", kid_subset)
            reconmetric = ReconMetric(device='cuda')
            genmetric = GenMetric(device='cuda', kid_subset=kid_subset)

            for ii, batch in enumerate(tqdm(dataloader)):
                # images should be in range [-1,1] and in format (B, H, W, C), permuted later

                batch, dataidx = batch

                refimg = batch['image_ref'].cuda().float()
                tarimg = batch['image_target'].cuda().float()
                mask = (batch['highwarp'].float().cuda()+1)/2 # [-1,1]->[0,1], zeros are pixels without information
                pred = img_logger.log_img(model, batch, resume, split='test', returngrid='train', warpeddepth=None, onlyretimg=True).permute(0,2,3,1) # from chw to hwc, in range -1,1

                if args.save_generations:
                    for i in range(pred.shape[0]):
                        if dataidx[i]%args.savefreq==0:
                            predimg = ((pred[i].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                            Image.fromarray(predimg).save(join(savepath, 'generations', f'{dataidx[i]}.png'))
                        countgen += 1
                if args.save_data:
                    for i in range(pred.shape[0]):
                        if dataidx[i]%args.savefreq==0:
                            ref = ((refimg[i].cpu().numpy()+1)/2*255).astype(np.uint8)
                            Image.fromarray(ref).save(join(savepath, 'refimgs', f'{dataidx[i]}.png'))
                            tar = ((tarimg[i].cpu().numpy()+1)/2*255).astype(np.uint8)
                            Image.fromarray(tar).save(join(savepath, 'tarimgs', f'{dataidx[i]}.png'))
                            m = ((mask[i].cpu().numpy())*255).astype(np.uint8)
                            Image.fromarray(m).save(join(savepath, 'masks', f'{dataidx[i]}.png'))
                            countdata += 1

                pred = pred.cuda().float() 
                _ = reconmetric.update(refimg, tarimg, mask, pred)
                _ = genmetric.update(refimg, tarimg, mask, pred)

            reconresult = reconmetric.compute()
            genresult = genmetric.compute()

            genresult['fid'] = genresult['fid'].cpu().item()
            genresult['kid'] = (genresult['kid'][0].cpu().item(), genresult['kid'][1].cpu().item())
            genresult['masked_fid'] = genresult['masked_fid'].cpu().item()
            genresult['masked_kid'] = (genresult['masked_kid'][0].cpu().item(), genresult['masked_kid'][1].cpu().item())
    
           
            with open(join(savepath, 'reconmetrics.txt'), 'w') as file:
                json.dump(reconresult, file, indent=4)
            with open(join(savepath, 'genmetrics.txt'), 'w') as file:
                json.dump(genresult, file, indent=4)

           

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

    arg_parser.add_argument("--released_ckpt", type=str) # 'zeronvs.ckpt' or 'zero123-xl.ckpt'

    arg_parser.add_argument("--dataset", default='megascenes', type=str)
    arg_parser.add_argument("--savepath", "-s", required=True, type=str)
    arg_parser.add_argument("--save_generations", default=True, help='output generated images')
    arg_parser.add_argument("--save_data", action='store_true', help='output reference and target images and masks')

    arg_parser.add_argument("--savefreq", default=10, type=int, help='save every n-th image')
    arg_parser.add_argument("--batch_size", "-b", default=1, type=int)
    arg_parser.add_argument("--workers", "-w", default=0, type=int)

    args = arg_parser.parse_args()
   
    main()

    # example usage: python test.py -e configs/colmap/warping_only/baseline/ -r 52000 -b 88 -w 4 -s quanteval/40k/warponly_52000 --save_generations

    
