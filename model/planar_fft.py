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
        if opt.arch.kernel_type  == "gaussian_diff":
            sigma = self.graph.neural_image.gaussian_diff_kernel_sigma
            self.tb.add_scalar(f"{split}/{'gaussian_kernel_std'}", sigma , step)
            wandb.log({f"{split}.{'gaussian_kernel_std'}": sigma}, step=step)

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
            loss.total_variance = self.Parseval_Loss(self.neural_image)
        return loss
    
    def Parseval_Loss(self, svdImage):
        # ParseVal Loss in PREF (similar to TV loss in spatial domain)
        return svdImage.Parseval_Loss()


class NeuralImageFunction(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()

        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution # H W
        self.H, self.W = self.resolution[0], self.resolution[1]
        
        # PREF setting for 3D
        """
        self.max_ranks = list(map( lambda x: math.floor(math.log2(x))+1 , self.resolution))
        self.freqs_h = torch.tensor([0] + [2**i for i in range(self.max_ranks[0]-1)])
        self.freqs_w = torch.tensor([0] + [2**i for i in range(self.max_ranks[1]-1)])
        """
        # PREF setting for 2D
        self.max_ranks = self.resolution[0]//5, self.resolution[1]//5
        self.freqs_h = torch.arange(0,self.max_ranks[0])
        self.freqs_w = torch.arange(0,self.max_ranks[1])

        self.device = torch.device("cpu" if opt.cpu else f"cuda:{opt.gpu}")
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

        self.basis_h = torch.stack(list(torch.exp(2j*3.141592*f/self.H * torch.arange(0,self.H)) for f in self.freqs_h)).unsqueeze(0).unsqueeze(3).expand(3,-1,-1,self.W) #3, self.max_ranks[0], H, W
        self.basis_w = torch.stack(list(torch.exp(2j*3.141592*f/self.W * torch.arange(0,self.W)) for f in self.freqs_w)).unsqueeze(0).unsqueeze(2).expand(3,-1,self.H,-1) #3, self.max_ranks[1], H, W
        self.basis_h = self.basis_h.to(self.device)
        self.basis_w = self.basis_w.to(self.device)

        self.define_network()
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

    def define_network(self):
        shape1 = (3,self.max_ranks[0], self.resolution[1])
        rank1 = torch.complex(torch.zeros(*shape1), torch.zeros(*shape1))
        
        shape2 = (3, self.max_ranks[1], self.resolution[0])
        rank2 = torch.complex(torch.zeros(*shape2), torch.zeros(*shape2))
        
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
        cur_rank1 = min(cur_rank, self.max_ranks[0]) 
        cur_rank2 = min(cur_rank, self.max_ranks[1])
        
        rank1_ifft = torch.fft.fft(self.rank1[:, :cur_rank1,:], dim=2) # 3, cur_rank1, self.resolution[1]
        rank2_ifft = torch.fft.fft(self.rank2[:, :cur_rank2,:], dim=2) # 3, cur_rank2, self.resolution[0] 

        if self.kernel_type != "none":
            kernel = self.get_kernel()
            complex_kernel = torch.complex(kernel, kernel)

            rank1_ifft = torch_F.conv1d(rank1_ifft[:, :, :], complex_kernel.expand(cur_rank1, 1, -1),
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank1)
            rank2_ifft = torch_F.conv1d(rank2_ifft[:, :, :], complex_kernel.expand(cur_rank1, 1, -1),
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank2)

        img1 = torch.sum(rank1_ifft.unsqueeze(2) * self.basis_h[:,:cur_rank1, ...], dim=1, keepdim=False)
        img2 = torch.sum(rank2_ifft.unsqueeze(3) * self.basis_w[:,:cur_rank2, ...], dim=1, keepdim=False)
       
        rbg = torch.real(img1) + torch.real(img2)
        rbg *= 255
        assert rbg.shape == (
            3, self.resolution[0], self.resolution[1]), f"rbg image has shape {rbg.shape}"
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
    def Parseval_Loss(self):
        r1_v = torch.arange(0,self.W)[None,None,...].to(self.device) * self.rank1
        r1_u = self.freqs_h[None, ... , None].to(self.device) * self.rank1

        r2_v = torch.arange(0,self.H)[None,None,...].to(self.device) * self.rank2
        r2_u = self.freqs_w[None, ... , None].to(self.device) * self.rank2

        return sum(torch.linalg.norm(r) for r in [r1_v, r1_u, r2_v, r2_u])