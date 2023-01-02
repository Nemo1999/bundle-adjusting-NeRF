import numpy as np
import os,sys,time
import torch
import importlib
import wandb
import options
from util import log


def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)
    if opt.wandb:
        wandb.init(
            project=f"{opt.group}",
            notes=f"planar run: model={opt.model}, name={opt.name}, yaml={opt.yaml}",
            tags=["planar", opt.name],
            config=opt,
        )
    else:
        wandb.init(mode="disabled")

    wandb.run.name = f"{opt.name}"

    # train model
    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)

        m.train(opt)

    # evaluate model
    with torch.cuda.device(opt.device):
        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        if opt.model == "barf":
            m.generate_videos_pose(opt)

        m.restore_checkpoint(opt)
        if opt.data.dataset in ["blender","llff"]:
            m.evaluate_full(opt)
        m.generate_videos_synthesis(opt)


if __name__=="__main__":
    main()
