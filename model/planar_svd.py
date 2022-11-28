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

# training cycle is same as planar


class Model(planar.Model):
    def __init__(self, opt):
        super().__init__(opt)
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric, step, split)
        if opt.arch.kernel_type == "gaussian_diff":
            sigma = self.graph.neural_image.gaussian_diff_kernel_sigma
            self.tb.add_scalar(f"{split}/{'gaussian_kernel_std'}", sigma , step)

# ============================ computation graph for forward/backprop ============================


class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)
        self.neural_image = SVDImageFunction(opt)
        self.device = torch.device("cpu" if opt.cpu else f"cuda:{opt.gpu}")

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
    
    def render_pose2image(self, xy_translation, homography_param, opt=None):
        # render with additional xy_translation or homography parameter
        #xy_translation, homography_param = pose_perterb
        xy_grid = warp.get_normalized_pixel_grid(opt)[:1] # only use 1 grid instead of opt.batch_size
        # translate grid 
        xy_grid += xy_translation
        # convert homo parameter to homo matrix using lie 
        warp_matrix = lie.sl3_to_SL3(homography_param)
        # warp grid using homography matrix
        xy_grid_hom = camera.to_hom(xy_grid)
        warped_grid_hom = xy_grid_hom@warp_matrix.transpose(-2,-1)
        warped_grid = warped_grid_hom[...,:2]/(warped_grid_hom[...,2:]+1e-8) # [B,HW,2]
        # predict rgb image
        rgb_whole = self.neural_image.forward(opt, warped_grid)
        rgb_whole = rgb_whole.view(opt.H, opt.W, 3).permute(2,0,1)
        return rgb_whole

    def pose2image_jacobian(self, opt):
        # function from warp_pert parameters to image
        trans_img, homo_img = torch.autograd.functional.jacobian(
            lambda t,h: self.render_pose2image(t,h, opt=opt), 
            (torch.zeros(2).to(self.device), torch.zeros(8).to(self.device)), # translation + homography params
            create_graph=False,
            strict = False,
            vectorize=True,
            strategy="forward-mode")
        # trans_img now have shape (2, H*W*3)
        assert tuple(trans_img.shape) == (3,opt.H,opt.W,2) , f"trans_img has shape {trans_img.shape}"
        trans_img = trans_img.permute(3, 0, 1, 2)
        homo_img = homo_img.permute(3, 0, 1, 2) 
        return trans_img, homo_img

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
    @torch.no_grad()
    def get_gaussian_kernel(self, t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function
        assert kernel_size % 2 == 1 and kernel_size > 0
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()

    def get_gaussian_diff_kernel(self, t, kernel_size: int):
        if hasattr(self,"gaussian_diff_kernel_sigma"):
            sigma = self.gaussian_diff_kernel_sigma 
        else:
            # create and return the parater used as sigma.
            sigma = torch.tensor(t).to(self.device)
        ns =torch.arange(-(kernel_size//2), kernel_size//2+1).to(self.device)
        exponent = - 0.5 * (ns / max(sigma,0.1)) * (ns / max(sigma,0.1))
        kernel = 1/max(sigma*math.sqrt(2*math.pi), 1) * torch.exp(exponent)
        return kernel.to(torch.float), sigma.to(torch.float)

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
        if self.kernel_type == "gaussian":
            kernel = self.get_gaussian_kernel(t, self.kernel_size)
        elif self.kernel_type == "average":
            kernel =  self.get_average_kernel(t, self.kernel_size)
        elif self.kernel_type == "gaussian_diff":
            kernel, _ = self.get_gaussian_diff_kernel(t, self.kernel_size)
        else: 
            raise ValueError(f"invalid kernel type at \"{self.kernel_type}\"")
        return kernel.to(self.device)

    def define_network(self, resolution, max_rank):
        rank1 = torch.zeros(3, self.max_ranks, self.resolution[0])
        rank1 = torch.normal(rank1, 0.1)
        rank2 = torch.zeros(3, self.max_ranks, self.resolution[1])
        rank2 = torch.normal(rank2, 0.1)
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))
        # register kernel if it is differentialble
        if self.kernel_type == "gaussian_diff":
            t = interp_schedule(0, self.c2f_kernel)
            # register sigma as differentiable parameter
            kernel, sigma, = self.get_gaussian_diff_kernel(t, self.kernel_size)
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
