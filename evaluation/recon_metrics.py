import os
import sys
import torch
import numpy as np
from torch import nn
import os.path as osp
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


def _quantize(t, return_type="float"):
    # t is in range [-1, 1]
    t = (t+1)/2
    t = torch.clip(t, 0, 1)
    t = t * 255.0
    t = t.to(torch.uint8)
    if return_type == "int":
        return t
    else:
        t = t.to(torch.float32)
        t = t / 127.5 - 1
        return t


class ReconMetric():
    def __init__(self, quantize=False, masked=False, device="cuda"):
        self.quantize = quantize
        self.masked = masked
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        
        self.mask_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.mask_psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.mask_ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        
        # Storage for the values 
        self.cnt = 0
        
    def reset(self):
        self.lpips.reset()
        self.psnr.reset()
        self.ssim.reset()
        self.mask_lpips.reset()
        self.mask_psnr.reset()
        self.mask_ssim.reset()
        self.cnt = 0
    
    def compute(self, aggregation="mean"):
        psnr = self.psnr.compute()
        ssim = self.ssim.compute()
        lpips = self.lpips.compute()
        mask_psnr = self.mask_psnr.compute()
        mask_ssim = self.mask_ssim.compute()
        mask_lpips = self.mask_lpips.compute()
        return {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "mask_psnr": mask_psnr.item(),
            "mask_ssim": mask_ssim.item(),
            "mask_lpips": mask_lpips.item(),
            "cnt": self.cnt
        } 
            
    def update(self, ref, gt_view, mask, pred_view):
        """ Compute reconstruction metrics.

        Args:
            ref (numpy array): _description_
            trg (_type_): _description_
            mask (_type_): _description_
            pred (_type_): Predicted views.
            
        Borrow from 
        https://github.com/kylesargent/ZeroNVS/blob/main/threestudio/systems/base.py#L51
        """
        if self.quantize:
            gt_view = _quantize(gt_view, return_type="float")
            pred_view = _quantize(pred_view, return_type="float")
        gt_view = gt_view.permute(0, 3, 1, 2)
        pred_view = pred_view.permute(0, 3, 1, 2)

        # Range: [-1, 1]
        self.lpips.update(gt_view, pred_view)
        self.psnr.update(gt_view, pred_view)
        self.ssim.update(gt_view, pred_view)
        
        # Mask: 0 -> no information 
        # NOTE: this is a bit higher since the region mask=0 is all correct
        mask = mask.permute(0, 3, 1, 2)
        self.mask_lpips.update(gt_view * mask, pred_view * mask)
        self.mask_psnr.update(gt_view * mask, pred_view * mask)
        self.mask_ssim.update(gt_view * mask, pred_view * mask)
        self.cnt += ref.shape[0]

        return 
    
    
if __name__ == "__main__":
    pred_path = "evaluation/test_data/example_pairs_pred/"
    gtr_path = "evaluation/test_data/example_pairs/"
    def make_loader():
        for f in os.listdir(gtr_path):
            refimg = torch.from_numpy(
                np.array(Image.open(osp.join(gtr_path, f, "refimg.png")))
            )[None, ...].to(torch.float) / 255. * 2 - 1
            tarimg = torch.from_numpy(
                np.array(Image.open(osp.join(gtr_path, f, "tarimg.png")))
            )[None, ...].to(torch.float) / 255. * 2 - 1
            # mask = torch.from_numpy(
            #     np.load(osp.join(gtr_path, f, "warpedmask.npy"))
            # )[None, ...].to(torch.float) / 255. * 2 - 1
            mask = (torch.rand_like(tarimg) > 0).to(torch.int)
            pred = torch.from_numpy(
                np.array(Image.open(osp.join(pred_path, f, "outimg.png")))
            )[None, ...].to(torch.float) / 255. * 2 - 1
            refimg = refimg.cuda()
            tarimg = tarimg.cuda()
            mask = mask.cuda()
            pred = pred.cuda()
            yield refimg, tarimg, mask, pred
      
    with torch.no_grad(): 
        loader = make_loader()
        metric = ReconMetric(device='cuda')
        for refimg, tarimg, mask, pred in loader:
            out = metric.update(refimg, tarimg, mask, pred)
    result = metric.compute()
    print(result)
            