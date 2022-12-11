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
import camera 
from warp import lie
import wandb

# training cycle is same as planar


class Model(planar.Model):
    def __init__(self, opt):
        super().__init__(opt)
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric, step, split)
        if opt.arch.kernel_type in ["gaussian_diff", "combined_diff"]:
            sigma = self.graph.neural_image.gaussian_diff_kernel_sigma
            self.tb.add_scalar(f"{split}/{'gaussian_kernel_std'}", sigma.data , step)
            wandb.log({f"{split}.{'gaussian_kernel_std'}": sigma.data}, step=step)

# ============================ computation graph for forward/backprop ============================


class Graph(planar.Graph):

    def __init__(self, opt):
        super().__init__(opt)
    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(
                opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
            loss.render = self.MSE_loss(var.rgb_warped, image_pert)
        if opt.loss_weight.total_variance is not None:
            loss.total_variance = self.TV_loss(self.neural_image)
        return loss
    
    def TV_loss(self, svdImage):
        # Total Variance Loss
        r1, r2 = svdImage.rank1, svdImage.rank2
        
        N1 = svdImage.resolution[0] * svdImage.max_ranks
        tv1 = (r1[...,1:] - r1[...,:-1])
        tv1 = tv1 * tv1
        tv1 = torch.sum(tv1) / N1
        N2 = svdImage.resolution[1] * svdImage.max_ranks
        tv2 = (r2[...,1:] - r2[...,:-1])
        tv2 = tv2 * tv2
        tv2 = torch.sum(tv2) / N2

        return tv1 + tv2


class NeuralImageFunction(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()

        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution # W, H
        self.max_ranks = opt.arch.max_ranks

        self.device = torch.device("cpu" if opt.cpu else f"cuda:{opt.gpu}")
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

        self.define_network()
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))
    @torch.no_grad()
    def get_gaussian_kernel(self, t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function

        assert kernel_size % 2 == 1 and kernel_size > 0, f"invalid kernel_size={kernel_size}"
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()

    def get_gaussian_diff_kernel(self, t, kernel_size: int, sigma_scale=None):

        if hasattr(self,"gaussian_diff_kernel_sigma"):
            # reuse diff model parameter sigm
            if sigma_scale != None: # scaling parameter for combined kernel
                sigma = sigma_scale * self.gaussian_diff_kernel_sigma
            else: 
                sigma = self.gaussian_diff_kernel_sigma
        else:
            # create and return the parater used as sigma.
            sigma = torch.tensor(t).to(self.device)
        ns =torch.arange(-(kernel_size//2), kernel_size//2+1).to(self.device)
        exponent = - 0.5 * (ns / max(sigma,0.01)) * (ns / max(sigma,0.01))
        kernel = 1/max(sigma*math.sqrt(2*math.pi), 1) * torch.exp(exponent) 
        return kernel.to(torch.float), sigma.to(torch.float)

    def get_combined_diff_kernel(self, t, kernel_size: int): 
        kernels = torch.stack(tuple(
            self.get_gaussian_diff_kernel(t, kernel_size, sigma_scale=2**i)[0] 
            for i in range(5)))
        kernels = torch.mean(kernels, dim=0)
        return kernels

    @torch.no_grad()        
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
        # smaller kernel size for small t, for faster computation 
        kernel_size = min(int(t * 4) , self.kernel_size)
        if kernel_size % 2 == 0 :
            kernel_size += 1

        if self.kernel_type == "gaussian":
            kernel = self.get_gaussian_kernel(t, kernel_size)
        elif self.kernel_type == "average":
            kernel =  self.get_average_kernel(t, kernel_size)
        elif self.kernel_type == "gaussian_diff":
            kernel, _ = self.get_gaussian_diff_kernel(t, kernel_size)
        elif self.kernel_type == "combined_diff":
            kernel = self.get_combined_diff_kernel(t, kernel_size)
        else: 
            raise ValueError(f"invalid kernel type at \"{self.kernel_type}\"")
        return kernel.to(self.device)

    def define_network(self):
        rank1 = torch.zeros(3, self.max_ranks, self.resolution[0])
        rank1 = torch.abs(torch.normal(rank1, 0.1))
        rank2 = torch.zeros(3, self.max_ranks, self.resolution[1])
        rank2 = torch.abs(torch.normal(rank2, 0.1))
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))
        # register kernel if it is differentialble
        if self.kernel_type in ["gaussian_diff", "combined_diff"] :
            t = interp_schedule(0, self.c2f_kernel)
            # register sigma as differentiable parameter
            _ , sigma, = self.get_gaussian_diff_kernel(t, self.kernel_size)
            self.register_parameter(name='gaussian_diff_kernel_sigma', param=torch.nn.Parameter(sigma.to(self.device)))

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
