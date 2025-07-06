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
from guided_diffusion.image_datasets import load_data
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
from noise_refine_model.crossattn import laplacian_kernel, PixelCrossAttentionRefiner, AlexNetPerceptualLoss
import torch.nn as nn
import torch.optim as optim 

def main():
    print('branch: main_train_imagenet')
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
    repo_folder_path = "/mnt/HDD2/phu2/custom-guided-diffusion/"  # (server 148)
    # repo_folder_path = "/mnt/HDD2/phudh/custom-guided-diffusion/"    # (server 118 or 92)
    
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
    refine_net = PixelCrossAttentionRefiner(feat_dim=3, embed_dim=32, num_heads=16).to(dist_util.dev())
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        refine_net.parameters(), 
        lr=2e-4, weight_decay=1e-5
    )

    perc_loss = AlexNetPerceptualLoss(device=dist_util.dev()).to(dist_util.dev())

    ## checkpoint = th.load(repo_folder_path + 'refine_net_950.pth')
    ## refine_net.load_state_dict(checkpoint['model_state_dict'])
    ## optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ## start_epoch = checkpoint['epoch'] + 1
    ## print('start_epoch:', start_epoch)

    # refine_net.train()
    verbose = False

    # load data
    batch_size = 32
    data = load_data(
        data_dir='/mnt/HDD2/phu2/custom-guided-diffusion/imagenet1k256',
        batch_size=batch_size,
        image_size=256,
        class_cond=False,
        # deterministic=False,
    )

    # batch = next(data)
    # print('len(batch): ', len(batch))  # 2
    # print('batch[0].shape: ', batch[0].shape)  #torch.Size([32, 3, 256, 256])
    # print('batch[1]: ', batch[1])  # {}  empty dictionary


    # def run_loop(self):
    #     while (
    #         not self.lr_anneal_steps
    #         or self.step + self.resume_step < self.lr_anneal_steps
    #     ):
    #         batch, cond = next(self.data)
    #         self.run_step(batch, cond)
    #         if self.step % self.log_interval == 0:
    #             logger.dumpkvs()
    #         if self.step % self.save_interval == 0:
    #             self.save()
    #             # Run for a finite amount of time in integration tests.
    #             if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
    #                 return
    #         self.step += 1
    #     # Save the last checkpoint if it wasn't already saved.
    #     if (self.step - 1) % self.save_interval != 0:
    #         self.save()


    n_iterations = 2000
    n_save_interval = 50
    for iteration in tqdm(range(n_iterations), desc='Training iterations'):
        timestep = th.randint(1, 200, (1,)).item()      # [1, 199)
        img_batch, _ = next(data)
        img_batch = img_batch.to(dist_util.dev())  # torch.Size([32, 3, 256, 256])

        batch_noise_candidate, batch_hf_info, batch_hf_star, batch_r_t = diffusion.get_5_candidates_for_train(
            model,
            shape=(batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            codebook=_codebooks[timestep],
            img_batch=img_batch,
            timestep=timestep,
            verbose=verbose
        )

        # batch_noise_candidate.shape:  torch.Size([32, 5, 3, 256, 256])
        # batch_hf_info.shape:          torch.Size([32, 5, 3, 256, 256])
        # batch_hf_star.shape:          torch.Size([32, 3, 256, 256])
        # batch_r_t.shape:              torch.Size([32, 3, 256, 256])
    
        z_hat = refine_net(batch_hf_star, batch_hf_info, batch_noise_candidate)  # → [B, 3, H, W]

        loss = criterion(z_hat, batch_r_t) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open('../learning_curve_imagenet.txt', 'a') as f:
            f.write(f'iteration {iteration} loss {loss}\n')

        if (iteration % n_save_interval == 0) or iteration == n_iterations-1:
            refine_net.save_checkpoint(optimizer, iteration, path=repo_folder_path + f'send_more_info_train_imagenet_jun_06/refine_net_epoch_{iteration}.pth')
        
        
    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
