import os
import tqdm
import torch
import lpips
import numpy as np
import os.path as osp
from PIL import Image
from collections import defaultdict
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from evaluation.recon_metrics import _quantize


class GenMetric():
    def __init__(self, quantize=False, masked=False, kid_subset=1000, device='cpu'):
        self.quantize = quantize
        self.masked = masked
        self.fid = FrechetInceptionDistance(normalize=False).to(device)
        self.masked_fid = FrechetInceptionDistance(normalize=False).to(device)
        self.kid = KernelInceptionDistance(normalize=False, subset_size=kid_subset).to(device)
        self.masked_kid = KernelInceptionDistance(normalize=False, subset_size=kid_subset).to(device)
        # self.is_gtr = InceptionScore().to(device)
        # self.is_pred = InceptionScore().to(device)
        # self.masked_is_pred = InceptionScore().to(device)
        self.cnt = 0
        
    def reset(self):
        self.fid.reset()
        self.masked_fid.reset()
        self.kid.reset()
        self.masked_kid.reset()
        # self.is_gtr.reset()
        # self.is_pred.reset()
        # self.masked_is_pred.reset()
        self.cnt = 0
        
    def compute(self, aggregate=None):
        fid = self.fid.compute()
        masked_fid = self.masked_fid.compute()
        kid = self.kid.compute()
        masked_kid = self.masked_kid.compute()
        # is_pred = self.is_pred.compute()
        # masked_is_pred = self.masked_is_pred.compute()
        # is_gtr = self.is_gtr.compute()
        return {
            "fid": fid,
            "kid": kid, 
            # "is_pred": is_pred,
            "masked_fid": masked_fid,
            "masked_kid": masked_kid, 
            # "masked_is_pred": masked_is_pred,
            # "is_gtr": is_gtr,
            "cnt": self.cnt
        } 
        
    def update(self, ref, gt_view, mask, pred_view):
        """ Compute generative metrics such as FID, KID, and IS.

        Args:
            ref (numpy array): _description_
            trg (_type_): _description_
            mask (_type_): _description_
            pred (_type_): Predicted views.
            quantize (bool, optional): Whether to quantize. Defaults to False.
            
        """
        gt_view = _quantize(
            gt_view, return_type="int").permute((0, 3, 1, 2))
        pred_view = _quantize(
            pred_view, return_type="int").permute((0, 3, 1, 2))
        mask = mask.permute(0, 3, 1, 2).to(torch.uint8)
        masked_pred_view = mask * gt_view + (1 - mask) * pred_view
        
        # TODO: make sure it's uint8.
        # TODO: check the range of the image.
        # TODO: mask=0 means there exist points from depth warping. 

        #import ipdb; ipdb.set_trace()
        self.fid.update(gt_view, real=True)
        self.masked_fid.update(gt_view, real=True)
        self.fid.update(pred_view, real=False)
        self.masked_fid.update(masked_pred_view, real=False)
        
        self.kid.update(gt_view, real=True)
        self.masked_kid.update(gt_view, real=True)
        self.kid.update(pred_view, real=False)
        self.masked_kid.update(masked_pred_view, real=False)
        
        # self.is_gtr.update(gt_view)
        # self.is_pred.update(pred_view)
        # self.masked_is_pred.update(masked_pred_view)
        self.cnt += int(gt_view.shape[0])
        
        # NOTE: there is no intermediate outputs
        return
    
    
if __name__ == "__main__":
    pred_path = "evaluation/test_data/example_pairs_pred/"
    gtr_path = "evaluation/test_data/example_pairs/"
    def make_loader(batch_size: int = 512):
        while True:
            for f in os.listdir(gtr_path):
                refimg = torch.from_numpy(
                    np.array(Image.open(osp.join(gtr_path, f, "refimg.png")))
                )[None, ...].repeat(batch_size, 1, 1, 1)
                tarimg = torch.from_numpy(
                    np.array(Image.open(osp.join(gtr_path, f, "tarimg.png")))
                )[None, ...].repeat(batch_size, 1, 1, 1)
                mask = torch.from_numpy(
                    np.array(Image.open(osp.join(gtr_path, f, "tarimg.png")))
                )[None, ..., :1].repeat(batch_size, 1, 1, 1)
                pred = torch.from_numpy(
                    np.array(Image.open(osp.join(pred_path, f, "outimg.png")))
                )[None, ...].repeat(batch_size, 1, 1, 1)
                refimg = refimg.cuda()
                tarimg = tarimg.cuda()
                mask = mask.cuda()
                pred = pred.cuda()
                print(refimg.shape, tarimg.shape, mask.shape, pred.shape)
                yield refimg, tarimg, mask, pred
    
    batch_size = 64
    ttl = 10_000 // batch_size + 10
    pbar = tqdm.tqdm(total=ttl)
    with torch.no_grad(): 
        loader = make_loader(batch_size)
        metric = GenMetric(device='cuda:0')
        cnt = 0
        for refimg, tarimg, mask, pred in loader:
            out = metric.update(refimg, tarimg, mask, pred)
            if cnt > ttl:
                break
            cnt += 1
            pbar.update(1)
        print("Computing the metrics...")
        result = metric.compute()
    print(result)
            