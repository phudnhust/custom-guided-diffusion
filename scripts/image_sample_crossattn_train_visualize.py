"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import time

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import random
from tqdm.auto import tqdm

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults()) 
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

from PIL import Image
from noise_refine_model.crossattn import laplacian_kernel, PixelCrossAttentionRefiner
import torch.nn as nn
import torch.optim as optim 

def main():
    start_time = time.perf_counter()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(th.cuda.get_device_properties(dist_util.dev()))
    model.to(dist_util.dev())
    print('device:', dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Loading model time: {elapsed_time:.6f} seconds")

    logger.log("sampling...")
    all_images = []
    all_labels = []

    # repo_folder_path = "/mnt/HDD2/phudoan/my_stuff/custom-guided-diffusion/"  # (server 147)
    repo_folder_path = "/mnt/HDD2/phudh/custom-guided-diffusion/"    # (server 118 or 92)
    
    # Generate codebook
    start_time = time.perf_counter()
    print('Loading codebook...')
    K = 32; img_size = 256; T = 1000


    # SHOULD USE: -------- Using torch ---------
    ## th.manual_seed(100)
    # _codebooks = th.randn((T + 1, K, 3, img_size, img_size), dtype=th.float16, device='cpu')
    # _codebooks = _codebooks.numpy()
    # np.save(repo_folder_path + f'models/codebooks_K_{K}.npy', _codebooks)

    _codebooks = np.load(repo_folder_path + f'models/codebooks_K_{K}.npy')
    print('_codebooks.shape:', _codebooks.shape)

    print('Codebook generated!')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Loading codebooks time: {elapsed_time:.6f} seconds")
    
    start_time = time.perf_counter()

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
    
    #---------------- REFINE NET INITIALIZE -----------------------
    refine_net = PixelCrossAttentionRefiner(feat_dim=3, embed_dim=3, num_heads=3).to(dist_util.dev())
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        refine_net.parameters(), 
        lr=1e-4, weight_decay=1e-5
    )

    ## checkpoint = th.load(repo_folder_path + 'refine_net_950.pth')
    ## refine_net.load_state_dict(checkpoint['model_state_dict'])
    ## optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ## start_epoch = checkpoint['epoch'] + 1
    ## print('start_epoch:', start_epoch)

    # refine_net.train()
    verbose = False

    # load data
    batch_size = 1
    hq_img_folder = '/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama'
    all_hq_img = [os.path.join(hq_img_folder, f) for f in os.listdir(hq_img_folder) if os.path.isfile(os.path.join(hq_img_folder, f))]
    # random.shuffle(all_hq_img)  # Shuffle to ensure randomness before slicing
    hq_img_subset = all_hq_img[:int(0.7 * len(all_hq_img))]
    # hq_img_subset.sort()
    # print('train dataset: ')
    # print(hq_img_subset)
    hq_img_batches = [hq_img_subset[i:i+batch_size] for i in range(0, len(hq_img_subset), batch_size)]

    dummy_count = 0

    epoches_loss_list = []
    n_epoch = 1

    x_0_list = None

    for epoch in range(0, n_epoch):
        print('epoch: ', epoch, end=' ')
        # hq_img_batch = random.sample(hq_img_subset, batch_size)
        hq_img_batch = hq_img_subset[:batch_size]
        epoch_loss_list = []


        batch_size = len(hq_img_batch)
        print('batch size: ', batch_size)

        # timestep = th.randint(1, 200, (1,)).item()
        # timestep = 200
        # timestep = 150
        # timestep = 50
        timestep = 10
        
        for hq_img_path in hq_img_batch:
            noise_candidate_list, hf_info_list, hf_star, r_t, x_0_list = diffusion.get_5_candidates_for_train(
                model,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                codebook=_codebooks[timestep],
                hq_img_path=hq_img_path,
                timestep=timestep,
                verbose=verbose
            )       # (noise_candidate_list, x_0_list)

    for i, x_0 in enumerate(x_0_list):
        x_0 = ((x_0 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        x_0 = x_0.permute(0, 2, 3, 1)
        x_0 = x_0.contiguous()
        imgs = x_0.cpu().numpy()  # shape: (N, H, W, C)
        im = Image.fromarray(imgs[0])
        im.save(f"x_0_t____t_200_{i}.png")
        print('saved image')


    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
