import torch
import numpy as np
import scipy
import math

@torch.no_grad()
def get_gaussian_kernel(t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function

        assert kernel_size % 2 == 1 and kernel_size > 0, f"invalid kernel_size={kernel_size}"
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

@torch.no_grad()        
def get_average_kernel(self, t, kernel_size: int):
        # to be consistent with gaussian kernel
        # we should return impulse when t = 0
        t = math.floor(t)
        kernel = torch.zeros(self.kernel_size)
        kernel[self.kernel_size-t:self.kernel_size+t+1] = 1 / (t*2 + 1)
        return kernel

def get_gaussian_diff_kernel(kernel_size: int, sigma_scale=None, external_sigma=None):
        assert external_sigma != None , "external_sigma is not defined"

        if external_sigma is not None:
            sigma = external_sigma * (1 if sigma_scale is None else sigma_scale) # scaling parameter for combined kernel
        # elif hasattr(self,"gaussian_diff_kernel_sigma"):
        #     # use internal sigma for neural image if external_sigma is None
        #     # reuse diff model parameter 
        #     sigma = self.gaussian_diff_kernel_sigma * (1 if sigma_scale is None else sigma_scale) # scaling parameter for combined kernel
        else:
            error("gaussian_diff_kernel_sigma is not defined")
        #ic(sigma, kernel_size)
        ns =torch.arange(-(kernel_size//2), kernel_size//2+1).to(self.device)
        exponent = - 0.5 * (ns / max(sigma,0.001)) * (ns / max(sigma,0.001))
        kernel = 1/max(sigma*math.sqrt(2*math.pi), 1) * torch.exp(exponent) 
        return kernel.to(torch.float)

def get_combined_diff_kernel(kernel_size: int, mode="weighted",sigma_scale=1., external_sigma=None, combine_levels=6):
        assert external_sigma != None
        kernels = torch.stack(tuple(
            get_gaussian_diff_kernel(kernel_size, sigma_scale=2**i * sigma_scale, external_sigma=external_sigma) 
            for i in range(combine_levels)))
        if mode == "weighted":
            weight = torch.arange(1, combine_levels+1).to(self.device).unsqueeze(1)
            weight = weight / torch.sum(weight)
            kernel = torch.sum(kernels * weight, dim=0)
        else:
            kernel = torch.mean(kernels, dim=0)
        return kernel