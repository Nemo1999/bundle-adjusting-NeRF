import nerf
import barf
from util import log
import torch
import os, sys, time
import tqdm
import wandb
import camera 
from easydict import EasyDict as edict


class Model(barf.Model):
    def __init__(self, opt):
        super().__init__(opt)
    def train(self, opt):
        log.title("Training START")
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        # training
        if self.iter_start==0: self.validate(opt,0)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for self.it in loader:
            if self.it<self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt,var,loader)
            if opt.optim.sched: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
            # update AlphaMask
            if self.it in opt.update_AlphaMask_List:
                if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                    reso_mask = reso_cur
                new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
                if iteration == update_AlphaMask_list[0]:
                    tensorf.shrink(new_aabb)
                    # tensorVM.alphaMask = None
                    L1_reg_weight = args.L1_weight_rest
                    print("continuing L1_reg_weight", L1_reg_weight)

                if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                    # filter rays outside the bbox
                    allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)
        
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

class Graph(barf.Graph):
    