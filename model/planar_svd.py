from . import planar
from . import base
import warp
from easydict import EasyDict as edict
from util import interp_schedule
import torch
import torch.nn.functional as torch_F
import math
import scipy
import numpy as np
from typing import Tuple

# training cycle is same as planar
class Model(planar.Model):
     def __init__(self, opt):
          super().__init__(opt)

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.svd_image = SVDImageFunction(opt)

    def forward(self,opt,var,mode=None):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,self.warp_param.weight)
        # render images
        var.rgb_warped = self.svd_image.forward(opt,xy_grid_warped) # [B,HW,3]
        var.rgb_warped_map = var.rgb_warped.view(opt.batch_size,opt.H_crop,opt.W_crop,3).permute(0,3,1,2) # [B,3,H,W]
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(opt.batch_size,3,opt.H_crop*opt.W_crop).permute(0,2,1)
            loss.render = self.MSE_loss(var.rgb_warped,image_pert)
        return loss

class SVDImageFunction(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()

        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution
        self.max_ranks = opt.arch.max_ranks
        
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

        self.define_network(opt, self.resolution, self.max_ranks)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

        
    def get_gaussian_kernel(self, t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function
        assert kernel_size %2 ==1 and kernel_size > 0
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel)

    def get_average_kernel(self, t, kernel_size: int):
        # to be consistent with gaussian kernel
        # we should return impulse when t = 0
        t = math.floor(t) 
        kernel = torch.zeros(self.kernel_size)
        kernel[self.kernel_size-t:self.kernel_size+t+1] = 1 / (t*2 + 1) 
        return kernel

    def get_kernel(self) -> torch.tensor :
        # the blurness of kernel depends on scheduled t parameter
        t = interp_schedule(self.progress , self.c2f_kernel)
        if self.kernel_type == "gaussian": 
            kernel = self.get_gaussian_kernel(t, self.kernel_size)
        else: 
            assert self.kernel_type == "average"
            kernel =  self.get_average_kernel(t, self.kernel_size)
        return kernel.to(self.device)
    
    def define_network(self, resolution, max_rank):
        self.rank1 = torch.zeros(3, self.max_ranks, self.resolution)
        self.rank2 = torch.zeros(3, self.max_ranks, self.resolution)

    def forward(self,opt,coord_2D): # [B,...,3]
        cur_rank = interp_schedule(self.progress, self.c2f_rank)
        kernel = self.get_kernel()
        r1_blur  = torch_F.conv1d(self.rank1[:,:cur_rank,:], kernel, 
            bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
        r2_blur  = torch_F.conv1d(self.rank2[:,:cur_rank,:], kernel, 
            bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
        rbg = torch.sum(r1_blur.unqueeze(2) * r2_blur.unsqueeze(3), dim=1, keepdim=False)
        assert rbg.shape == (3,self.resolution, self.resolution)
        return rbg
