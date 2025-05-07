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

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

# --------- RefineNoiseNet Definition ---------
class RefineNoiseNet(nn.Module):
    def __init__(self, vector_dim, t_embed_dim, hidden_dim=256):
        super().__init__()
        # vector_dim: D (flattened codebook vector size)
        # t_embed_dim: E (timestep embedding size)
        self.score_mlp = nn.Sequential(
            nn.Linear(vector_dim + t_embed_dim, hidden_dim),  # input: D + E
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output: scalar score
        )

    def forward(self, vectors, t_embed):
        # vectors: (B, 5, D) - top-5 codebook vectors flattened
        # t_embed: (B, E) - timestep embeddings
        B, K, D = vectors.shape  # K=5
        # Expand timestep embedding to shape (B, 5, E)
        t_expanded = t_embed.unsqueeze(1).expand(-1, K, -1)
        # Concatenate to shape (B, 5, D+E)
        x = th.cat([vectors, t_expanded], dim=-1)
        # Compute scores: (B, 5)
        scores = self.score_mlp(x).squeeze(-1)
        # Weights via softmax: (B, 5)
        weights = F.softmax(scores, dim=-1)
        # Weighted sum -> refined vector: (B, D)
        refined = th.sum(weights.unsqueeze(-1) * vectors, dim=1)
        return refined, weights
    
    # --------- Timestep Embedding ---------
    def timestep_embedding(t, dim):
        # t: (B,)      - timesteps
        # dim: E       - embedding size
        half_dim = dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = th.exp(-th.arange(half_dim, device=t.device) * emb_factor)
        # emb: (half_dim,)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        # Sinusoidal embedding: (B, E)
        return th.cat([th.sin(emb), th.cos(emb)], dim=-1)



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
    
    sample_fn = diffusion.ddcm_sample_direct
    sample = sample_fn(
        model,
        shape=(args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        codebook=_codebooks[100],
        # hq_img_path="/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/academic_gown/000.jpg",
        # hq_img_path="/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/academic_gown/004.jpg",
        hq_img_path='/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama/4.jpg',
        timestep=100
    )
    print('sample: ', sample) 
    # print('sample[\'retrieved_index\']', sample['retrieved_index'])
    # print('sample[\'top_sim_idx\']', sample['top_sim_idx'])


    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
