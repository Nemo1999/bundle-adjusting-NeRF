import nerf
import barf
from util import log
import torch
import os, sys, time
import tqdm
import wandb
import camera 
from easydict import EasyDict as edict
import numpy as np
import torch
from icecream import ic 
from typing import List
import tensorf_repr

class Model(nerf.Model):
    def __init__(self, opt):
        super().__init__(opt)
    
    def setup_optimizer(self, opt):
        self.optimizer = self.graph.nerf._get_optimizer(opt)
        # setup hooks for nerf to update optimizer params 
        self.graph.nerf.get_current_optimizer = lambda : self.optimizer
        def register_new_optimizer(optimizer):
            self.optimizer = optimizer
        self.graph.nerf.register_new_optimizer = register_new_optimizer
        
    def summarize_loss(self, opt, var, loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            assert(loss[key].shape==()), f"loss \"{key}\" has shape {loss[key].shape}"
            if key == "L1": 
                first_alpha_update = opt.train_schedule.update_alphamask_iters[0]
                weight = float(opt.loss_weight.L1.rest if self.ip > first_alpha_update else opt.loss_weight.L1.init)
                loss_all += weight * loss["L1"]
            elif opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key]
        loss.update(all=loss_all)
        return loss


class Graph(nerf.Graph):
    def __init__(self.opt):
        super().__init__(opt)
        

class NeRF(nerf.NeRF):
    def __init__(self, opt):
        super().__init__()

        self.bbox = torch.tensor(opt.data.scene_bbox).to(torch.float).view(2,3) # [xyz_min, xyz_max] 
        ic(self.bbox)
        
        # callback for updating optimizer used by up stream tensorf.Model
        self.register_new_optimizer = None
        self.get_current_optimizer = None

        # iterations for upsampling
        self.upsample_list = opt.train_schedule.upsample_iters
        ic(self.upsample_list)
        # upsampling voxels schedule (n_voxel is linear in logrithmic space)
        self.n_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(opt.train_schedule.n_voxel_init), np.log(opt.train_schedule.n_voxel_final), len(self.upsample_list)+1))).long()).tolist()
        ic(self.n_voxel_list)

        # lr_decay_factor
        self.lr_decade_duration = opt.max_iter if opt.optim.lr_decade_iters < 0 else opt.optim.lr_decade_iters
        ic(self.lr_decade_duration)
        self.lr_decade_factor = opt.opt.lr_decade_target_ratio ** (1 / self.lr_decade_duration)
        ic(self.lr_decade_factor)

        # find resolution from n_voxels
        self.n_voxels = self.n_voxel_list.pop(0)
        ic(self.n_voxels)
        self.resolution = self._find_resolution(self.n_voxels)
        ic(self.resolution)

        # alpha mask update
        self.alphamask_resolution = self.resolution
        self.update_alphamask_iters = opt.train_schedule.update_alphamask_iters

        # update ray sampling interval from resolution
        self._update_num_samples(opt, self.resolution)
        ic(opt.nerf.sample_intvs)

        
    def update_schedule(self, opt, it):
        assert self.register_new_optimizer is not None
        assert self.get_current_optimizer is not None
        
        if it in self.upsample_list:
            #  upsample voxels  --> update resolution --> upsample tensorf --> update_num_samples --> get_new_optimizer --> register new optmizer
            self.n_voxels = self.n_voxel_list.pop(0)
            self.resolution = self._find_resolution(self.n_voxels)
            self.tensorf.upsample_volume_grid(self.resolution)
            self._update_num_samples(opt, self.resolution)
            optimizer = self._get_optimizer(opt,it)
            self.register_new_optimizer(optimizer)
        else: 
            # update lr by lr_factor
            optimizer = self.get_current_optimizer()
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.lr_decade_factor
        
        if it in self.update_alphamask_iters:
            self._update_alphamask(it)
        
        if opt.loss_weight.TV_density > 0:
            opt.loss_weight.TV_density *= self.lr_decade_factor
        if opt.loss_weight.TV_color > 0:
            opt.loss_weight.TV_color *= self.lr_decade_factor
    
    def _find_resolution(self, n_voxels: int): 
        # find current resolution given total number of voxels
        xyz_min, xyz_max = self.bbox[0,:], self.bbox[1,:]
        dim = len(xyz_min)
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()

    def _update_num_samples(self, opt,  resolution: List[int]):
        auto_sample_number = int(np.linalg.norm(resolution)/opt.nerf.step_ratio) # auto adjustment sampling step with 
        opt.nerf.sample_intvs = min(opt.nerf.sample_intvs, auto_sample_number)

    def _get_optimizer(self, opt, it=0):
        # reset lr if lr_upsample_reset is true (default) , else, continue exponential lr decay schedule
        lr_scale = 1.0 if opt.optim.lr_upsample_reset or it==0 else opt.optim.lr_decay_target_ratio ** (it / opt.max_iter)
        grad_vars = self.tensorf.get_optparam_groups(opt.optim.lr_index * lr_scale, opt.optim.lr_basis * lr_scale)
        if opt.optim.algo == "Adam":
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99)) # different from default betas(0.9, 0.999)
        else: 
            optimizer = getattr(torch.optim, opt.optim.algo)(grad_vars)
        return optimizer

    def _update_alphamask(self, it):
        if it not in self.update_alphamask_iters:
            return
        if self.resolution[0] * self.resolution[1] * self.resolution[2] <256**3:# update volume resolution
            self.alphamask_resolution = self.resolution
            new_aabb_bbox = self.tensorf.updateAlphaMask(tuple(self.alphamask_resolution))
            if it == self.update_alphamask_iters[0]:
                # update bbox when we first update alpha mask
                self.tensorf.shrink(new_aabb_bbox)
                self.bbox = new_aabb_bbox

    def define_network(self, opt):
        # tensorf module
        self.tensorf = getattr(tensorf_repr, opt.arch.tensorf.model)(
            self.bbox 
            ,self.resolution
            ,opt.device
            ,density_n_comp=opt.arch.tensorf.density_components # number of components in tensor decomposition
            ,appearance_n_comp=opt.arch.tensorf.color_components # number of components in tensor decomposition
            ,app_dim=3 if opt.arch.shading=="RGB" else 27  # input feature dimension to Renderer , 27 is requried for SH (spherical harmonic) , we also use 27 for MLP Renderer for convinience
            ,near_far = opt.nerf.depth.range #near and far range of each camera
            ,shadingMode= opt.arch.shading.model # can be "SH" or "MLP_PE" or "MLP_Fea" or "MLP" or "RGB"
            ,alphaMask_thres=opt.train_schedule.alpha_mask_threshold # threshold for updating alpha mask volume
            ,density_shift=opt.arch.density_shift # shift density in softplus; making density = 0  when feature == 0
            ,distance_scale=opt.arch.distance_scale # scale up distance for sigma to alpha computation
            ,pose_pe=opt.arch.shading.pose_pe # positional encoding for position (MLP shader)
            ,view_pe=opt.arch.shading.view_pe # positional encoding for view direction (MLP shader)
            ,fea_pe=opt.arch.shading.fea_pe # positional encoding for input feature (MLP shader)
            ,featureC=opt.arch.shading.mlp_hidden_dim # hidden dimension for MLP shader
            ,step_ratio=opt.nerf.step_ratio # ratio between resolution and sampling_step_size # this ratio will compute and estimate sampling interval using current resolution,  overwrite nerf.sample_intvs if smaller
            ,fea2denseAct=opt.arch.feature_to_density_activation # activation used to convert raw tensor density value to real densoity
        )

    