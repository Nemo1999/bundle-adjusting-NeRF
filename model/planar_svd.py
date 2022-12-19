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
from matplotlib import pyplot as plt

# training cycle is same as planar


class Model(planar.Model):
    def __init__(self, opt):
        super().__init__(opt)
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric, step, split)
        if opt.arch.kernel_type != "none":
            if opt.arch.kernel_type in ["gaussian_diff", "combined_diff"]:
                # log differentialble kernel sigma 
                sigma = self.graph.neural_image.gaussian_diff_kernel_sigma
                self.tb.add_scalar(f"{split}/{'kernel_param'}", sigma.data , step)
                wandb.log({f"{split}.{'kernel_param'}": sigma.data}, step=step)
            else:
                # log scheduled kerenl sigma
                sigma = self.graph.neural_image.get_scheduled_sigma()
                self.tb.add_scalar(f"{split}/{'kernel_param'}", sigma , step)
                wandb.log({f"{split}.{'kernel_param'}": sigma}, step=step)
        # log rank
        rank = self.graph.neural_image.get_scheduled_rank()
        self.tb.add_scalar(f"{split}/{'rank'}", rank , step)
        wandb.log({f"{split}.{'rank'}": rank}, step=step)

    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step,split)
        neural_image = self.graph.neural_image
        max_kernel_size = neural_image.kernel_size
        kernel_sample = neural_image.get_kernel()
        padd_len = (max_kernel_size - kernel_sample.shape[0]) // 2
        kernel_sample = torch.nn.functional.pad(kernel_sample, (padd_len, padd_len))
        kernel_spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft(kernel_sample)))
        kernel_sample = kernel_sample.detach().cpu().numpy()
        kernel_spectrum = kernel_spectrum.detach().cpu().numpy()
        # log kernel
        fig = plt.figure()
        plt.plot(kernel_sample)
        wandb.log({f"{split}.{'kernel'}": wandb.Image(fig)}, step=step)
        plt.close(fig)
        # log fft transform of kernel
        fig = plt.figure()
        plt.plot(kernel_spectrum)
        wandb.log({f"{split}.{'kernel_fft'}": wandb.Image(fig)}, step=step)
        plt.close(fig)

        # log the gradient of sigma w.r.t the reconstruction Loss
        if opt.arch.kernel_type in ["gaussian_diff", "combined_diff", "gaussian"]:
            sigma_scales = [2.0**i for i in range(-3, 8)]
            weights = torch.arange(1,11+1).float().to(opt.device)
            weights = weights / weights.sum()
            for sigma_scale, weight, exponent in zip(sigma_scales, weights, range(-3, 8)):
                sigma = torch.tensor(1.0).to(torch.float).to(opt.device).requires_grad_()
                xy_grid = warp.get_normalized_pixel_grid_crop(opt)
                xy_grid_warped = warp.warp_grid(opt,xy_grid,self.graph.warp_param.weight)
                # render images
                rgb_warped = neural_image.forward(opt,xy_grid_warped, external_sigma=sigma*sigma_scale) # [B,HW,3]
                image_pert = var.image_pert.view(opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
                l2_loss = ((rgb_warped - image_pert)**2).mean(axis=2, keepdim=False).mean(axis=1, keepdim=False)
                
                # log all-patch grad w.r.t sigma
                total_grad_sigma = torch.autograd.grad(l2_loss.mean(), sigma, retain_graph=True)[0]
                #total_grad_sigma *= weight
                self.tb.add_scalar(f"P_all_sigma'_2^{exponent}", total_grad_sigma, step)
                wandb.log({f"P_all_grad_sigma'_2^{exponent}": total_grad_sigma}, step=step)

                # log all-patch grad w.r.t warp parameters
                total_grad_warp = torch.autograd.grad(l2_loss.mean(), self.graph.warp_param.weight, retain_graph=True)[0]
                #total_grad_warp *= weight
                total_grad_warp_norm = torch.norm(total_grad_warp, dim=1)
                total_warp_delta = (self.graph.warp_param.weight - self.warp_pert)  # current warp - GT warp
                total_grad_warp_cosine = torch.nn.functional.cosine_similarity(total_grad_warp, total_warp_delta , dim=1)
                
                self.tb.add_scalar(f"P_all_warp'_norm_2^{exponent}", total_grad_warp_norm.mean(), step)
                wandb.log({f"P_all_warp'_norm_2^{exponent}": total_grad_warp_norm.mean()}, step=step)

                self.tb.add_scalar(f"P_all_warp'_cosine_2^{exponent}", total_grad_warp_cosine.mean(), step)
                wandb.log({f"P_all_warp'_cosine_2^{exponent}": total_grad_warp_cosine.mean()}, step=step)

                # log per-patch loss
                for b in range(opt.batch_size):
                    # log per-patch grad w.r.t sigma
                    retain_graph = b != opt.batch_size - 1
                    patch_grad = torch.autograd.grad(l2_loss[b], sigma, retain_graph=retain_graph)[0]
                    #patch_grad *= weight
                    self.tb.add_scalar(f"P_{b}_sigma'_2^{exponent}", patch_grad, step)
                    wandb.log({f"P_{b}_sigma'_2^{exponent}": patch_grad}, step=step)

                    # log per-patch grad w.r.t warp parameters
                    self.tb.add_scalar(f"P_{b}_warp'_norm_2^{exponent}", total_grad_warp_norm[b], step)
                    wandb.log({f"P_{b}_warp'_norm_2^{exponent}": total_grad_warp_norm[b]}, step=step)
                    self.tb.add_scalar(f"P_{b}_warp'_cosine_2^{exponent}", total_grad_warp_cosine[b], step)
                    wandb.log({f"P_{b}_warp'_cosine_2^{exponent}": total_grad_warp_cosine[b]}, step=step)


           
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
        self.initialize_param(opt)
        self.define_network(opt)
        self.register_diff_kernel()
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.opt = opt

    def initialize_param(self,opt):
        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution # W, H

        self.device = opt.device
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

    @torch.no_grad()
    def get_gaussian_kernel(self, t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function

        assert kernel_size % 2 == 1 and kernel_size > 0, f"invalid kernel_size={kernel_size}"
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    def get_gaussian_diff_kernel(self, t, kernel_size: int, sigma_scale=None, external_sigma=None):
        
        if external_sigma is not None:
            sigma = external_sigma
        elif hasattr(self,"gaussian_diff_kernel_sigma"):
            # use internal sigma for neural image if external_sigma is None
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

    def get_combined_diff_kernel(self, t, kernel_size: int, mode="weighted"):

        combine_levels = self.opt.arch.combine_levels 
        
        kernels = torch.stack(tuple(
            self.get_gaussian_diff_kernel(t, kernel_size, sigma_scale=2**i)[0] 
            for i in range(combine_levels)))
        if mode == "weighted":
            weight = torch.arange(1, combine_levels+1).to(self.device).unsqueeze(1)
            weight = weight / torch.sum(weight)
            kernel = torch.sum(kernels * weight, dim=0)
        else:
            kernel = torch.mean(kernels, dim=0)
        return kernel

    @torch.no_grad()        
    def get_average_kernel(self, t, kernel_size: int):
        # to be consistent with gaussian kernel
        # we should return impulse when t = 0
        t = math.floor(t)
        kernel = torch.zeros(self.kernel_size)
        kernel[self.kernel_size-t:self.kernel_size+t+1] = 1 / (t*2 + 1)
        return kernel
    def get_scheduled_sigma(self):
        return interp_schedule(self.progress, self.c2f_kernel)
    def get_kernel(self, external_sigma=None) -> torch.tensor:
        # the blurness of kernel depends on scheduled t parameter
        t = self.get_scheduled_sigma()
        # smaller kernel size for small t, for faster computation 
        if self.kernel_type != "combined_diff": 
            # in combined_diff, kernel size should be fixed, t doesn't affect kernel size
            kernel_size = min(int(t * 4) , self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if kernel_size % 2 == 0 :
            kernel_size += 1

        if external_sigma is not None:
            kernel , _ = self.get_gaussian_diff_kernel(t, kernel_size, external_sigma=external_sigma)
        elif self.kernel_type == "gaussian":
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

    def define_network(self, opt):
        self.max_ranks = opt.arch.max_ranks
        rank1 = torch.zeros(3, self.max_ranks, self.resolution[0])
        rank1 = torch.abs(torch.normal(rank1, 0.1))
        rank2 = torch.zeros(3, self.max_ranks, self.resolution[1])
        rank2 = torch.abs(torch.normal(rank2, 0.1))
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))
    
    def register_diff_kernel(self):
        # register kernel if it is differentialble
        if self.kernel_type in ["gaussian_diff", "combined_diff"] :
            t = interp_schedule(0, self.c2f_kernel)
            # register sigma as differentiable parameter
            _ , sigma, = self.get_gaussian_diff_kernel(t, self.kernel_size)
            self.register_parameter(name='gaussian_diff_kernel_sigma', param=torch.nn.Parameter(sigma.to(self.device)))

    def get_scheduled_rank(self):
        return int(interp_schedule(self.progress, self.c2f_rank))

    def forward(self, opt, coord_2D, external_kernel=None, external_sigma=None):  # [B,...,3]
        cur_rank = self.get_scheduled_rank()

        if external_kernel is not None:
            kernel = external_kernel
        else:
            kernel = self.get_kernel(external_sigma).expand(cur_rank, 1, -1)

        r1_blur = torch_F.conv1d(self.rank1[:, :cur_rank, :], kernel,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
        r2_blur = torch_F.conv1d(self.rank2[:, :cur_rank, :], kernel,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank)
       
        rbg = torch.sum(r1_blur.unsqueeze(
            2) * r2_blur.unsqueeze(3), dim=1, keepdim=False)
        assert rbg.shape == (
            3, self.resolution[1], self.resolution[0]), f"rbg image has shape {rbg.shape}"
        B = coord_2D.shape[0]

        sampled_rbg = torch_F.grid_sample(rbg.expand(B, -1, -1, -1), coord_2D.unsqueeze(1), align_corners=False, mode=opt.arch.grid_interp).squeeze(2)
        
        return sampled_rbg.permute(0, 2, 1)