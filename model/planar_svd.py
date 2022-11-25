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
from icecream import ic

# training cycle is same as planar


class Model(planar.Model):
    def __init__(self, opt):
        super().__init__(opt)

# ============================ computation graph for forward/backprop ============================


class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)
        self.neural_image = SVDImageFunction(opt)

    def forward(self, opt, var, mode=None):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt, xy_grid, self.warp_param.weight)
        # render images
        var.rgb_warped = self.neural_image.forward(
            opt, xy_grid_warped)  # [B,HW,3]
        var.rgb_warped_map = var.rgb_warped.view(
            opt.batch_size, opt.H_crop, opt.W_crop, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
        return var

    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(
                opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
            #ic(image_pert.shape)
            #ic(var.rgb_warped.shape)
            loss.render = self.MSE_loss(var.rgb_warped, image_pert)

        return loss


class SVDImageFunction(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()

        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution
        self.max_ranks = opt.arch.max_ranks

        self.device = torch.device("cpu" if opt.cpu else f"cuda:{opt.gpu}")
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

        self.define_network(self.resolution, self.max_ranks)
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))

    def get_gaussian_kernel(self, t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function
        assert kernel_size % 2 == 1 and kernel_size > 0
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()

    def get_average_kernel(self, t, kernel_size: int):
        # to be consistent with gaussian kernel
        # we should return impulse when t = 0
        t = math.floor(t)
        kernel = torch.zeros(self.kernel_size)
        kernel[self.kernel_size-t:self.kernel_size+t+1] = 1 / (t*2 + 1)
        return kernel

    def get_kernel(self) -> torch.tensor:
        # the blurness of kernel depends on scheduled t parameter
        t = interp_schedule(self.progress, self.c2f_kernel)
        if self.kernel_type == "gaussian":
            kernel = self.get_gaussian_kernel(t, self.kernel_size)
        else:
            assert self.kernel_type == "average"
            kernel = self.get_average_kernel(t, self.kernel_size)
        return kernel.to(self.device)

    def define_network(self, resolution, max_rank):
        rank1 = torch.zeros(3, self.max_ranks, self.resolution[0])
        rank1 = torch.normal(rank1, 0.1)
        rank2 = torch.zeros(3, self.max_ranks, self.resolution[1])
        rank2 = torch.normal(rank2, 0.1)
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))

    def forward(self, opt, coord_2D):  # [B,...,3]
        cur_rank = int(interp_schedule(self.progress, self.c2f_rank))
        
        kernel = self.get_kernel().expand(cur_rank, 1, -1)
        

        r1_blur = torch_F.conv1d(self.rank1[:, :cur_rank, :], kernel,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
        r2_blur = torch_F.conv1d(self.rank2[:, :cur_rank, :], kernel,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
       
        rbg = torch.sum(r1_blur.unsqueeze(
            2) * r2_blur.unsqueeze(3), dim=1, keepdim=False)
        assert rbg.shape == (
            3, self.resolution[1], self.resolution[0]), f"rbg image has shape {rbg.shape}"
        B = coord_2D.shape[0]
        #ic(coord_2D)
        # coord_2D += 0.5
        # coord_2D[:,:,0] *= self.resolution[0]
        # coord_2D[:,:,1] *= self.resolution[1]

        sampled_rbg = torch_F.grid_sample(rbg.expand(B, -1, -1, -1), coord_2D.unsqueeze(1), align_corners=False).squeeze(2)
        #ic(sampled_rbg)
        #ic(coord_2D[0][0])
        #ic(sampled_rbg.shape)
        return sampled_rbg.permute(0, 2, 1)
