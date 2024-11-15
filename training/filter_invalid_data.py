import argparse
import logging
import math
import os
import shutil

import accelerate
import datasets
import torch
import torch.utils.checkpoint
import transformers
from torch.utils.data import RandomSampler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.optim.lr_scheduler import LambdaLR
from dataloaders.load import *
from util.noise import pyramid_noise_like
from util.loss import ScaleAndShiftInvariantLoss, AngularLoss
from util.unet_prep import replace_unet_conv_in
from util.lr_scheduler import IterExponential
from model import IntrinsicUNetSpatioTemporalConditionModel
import time
if is_wandb_available():
    import wandb
import json
from tqdm import tqdm
if __name__ == "__main__":
    # Training datasets
    hypersim_root_dir = "data/hypersim/processed"
    vkitti_root_dir   = "data/virtual_kitti_2"
    train_dataset_hypersim = Hypersim(root_dir=hypersim_root_dir, transform=True, H=16, W=16)
    train_dataset_vkitti   = VirtualKITTI2(root_dir=vkitti_root_dir, transform=True, H=16, W=16)
    mix_dataset = MixDataset([train_dataset_hypersim, train_dataset_vkitti])
    # sampler = RatioMixSampler(mix_dataset, [9, 1])
    sampler = RandomSampler(mix_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        mix_dataset,
        # shuffle=True,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )
    anns = []
    for batch in tqdm(train_dataloader):
        val_mask = batch["val_mask"].bool()
        if val_mask.any():
            continue
        
        rgb_path = batch["rgb_path"][0]
        # def path_to_relative(rgb_path):
        #     rgb_path = "/".join(rgb_path.split("/")[4:])
        anns.append(rgb_path)
    
    with open("123.json", "w") as fw:
        fw.write(json.dumps(anns, indent=2))