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
import random

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

    def train_refine_step(
        self,    # your RefineNoiseNet instance
        optimizer,     # optimizer for refine_net
        top5_vectors,  # Tensor, shape (B, 5, C, H, W)
        timesteps,     # LongTensor, shape (B,)
        residuals,     # Tensor, shape (B, C, H, W)  = (x0 - x0_pred)
        t_embed_fn     # fn: (timesteps: Tensor[B]) -> Tensor (B, E)
    ):
        """
        Performs one training step of RefineNoiseNet.

        Inputs:
        top5_vectors  (B, 5, C, H, W) — the 5 nearest codebook noise maps  
        timesteps     (B,)            — the diffusion step index for each sample  
        residuals     (B, C, H, W)    — the true denoising residual x0 – x0_pred  
        t_embed_fn    fn to get timestep embeddings of shape (B, E)

        Returns:
        loss.item()  — the scalar inner‐product loss
        """
        B, K, C, H, W = top5_vectors.shape
        # print('B, K, C, H, W:', top5_vectors.shape)
        D = C * H * W

        # 1) Flatten spatial dims
        # vectors_flat: (B, 5, D)
        vectors_flat = top5_vectors.contiguous().view(B, K, D)
        # residuals_flat: (B, D)
        residuals_flat = residuals.contiguous().view(B, D)

        # 2) Compute timestep embeddings
        # t_embed: (B, E)
        t_embed = t_embed_fn(timesteps, dim=128)

        # 3) Forward through RefineNoiseNet
        # v_hat: (B, D), weights: (B, 5)
        v_hat, weights = self(vectors_flat, t_embed)

        # 4) Inner‐product loss
        # ⟨residuals_flat, v_hat⟩ summed over D, mean over B
        # loss = - (residuals_flat * v_hat).sum(dim=1).mean()
        loss = F.mse_loss(v_hat, residuals_flat)  # L2 loss

        # 5) Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def save_checkpoint(self, optimizer, epoch, path="refiner_checkpoint.pth"):
        """
        Saves model+optimizer state and current epoch.
        
        refine_net:   your RefineNoiseNet instance
        optimizer:    the optimizer you’re using (e.g. AdamW)
        epoch:        current epoch number (int)
        path:         where to write the .pth file
        """
        th.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")



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
    
    #---------------------------------------
    refine_net = RefineNoiseNet(vector_dim=3*256*256, t_embed_dim=128).to(dist_util.dev())
    optimizer = th.optim.AdamW(
        refine_net.parameters(),
        lr=1e-6,           # a good starting point
        weight_decay=1e-2  # small amount of L2 regularization
    )
    verbose = False

    # load data
    batch_size = 32
    hq_img_folder = '/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama'
    all_hq_img = [os.path.join(hq_img_folder, f) for f in os.listdir(hq_img_folder) if os.path.isfile(os.path.join(hq_img_folder, f))]
    random.shuffle(all_hq_img)  # Shuffle to ensure randomness before slicing
    hq_img_subset = all_hq_img[:int(0.7 * len(all_hq_img))]
    hq_img_batches = [hq_img_subset[i:i+batch_size] for i in range(0, len(hq_img_subset), batch_size)]

    for epoch in range(1000):
        print('epoch: ', epoch, end=' ')
        # hq_img_batch = random.sample(hq_img_subset, batch_size)
        epoch_loss = []
        for hq_img_batch in hq_img_batches:
            batch_size = len(hq_img_batch)

            timestep = th.randint(1, 1000, (1,)).item()

            sample_fn = diffusion.ddcm_sample_direct
            top5_vectors_list = []
            residuals_list = []
            for hq_img_path in hq_img_batch:
                sample = sample_fn(
                    model,
                    shape=(args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    codebook=_codebooks[timestep],
                    hq_img_path=hq_img_path,
                    timestep=timestep,
                    verbose=verbose
                )
                # print('sample: ', sample) 
                if verbose:
                    print('sample[\'retrieved_index\']', sample['retrieved_index'])
                    print('sample[\'top_sim_idx\']', sample['top_sim_idx'])
                    print('-'*20); print()

                top5_vectors_list.append(th.from_numpy(_codebooks[timestep][sample['top_sim_idx']]))
                residuals_list.append(sample['residual'])

            top5_vectors = th.stack(top5_vectors_list, dim=0).to(dist_util.dev())
            residuals = th.stack(residuals_list, dim=0).to(dist_util.dev())
            timesteps = th.tensor([timestep] * batch_size).to(dist_util.dev())

            # if verbose:
                # print('top5_vectors.shape:', top5_vectors.shape)  # torch.Size([16, 5, 3, 256, 256])
                # print('residuals.shape:', residuals.shape)  # torch.Size([16, 1, 3, 256, 256])
                # print('timesteps.shape:', timesteps.shape)  #  torch.Size([16])

            loss = refine_net.train_refine_step(optimizer=optimizer, top5_vectors=top5_vectors, timesteps=timesteps, residuals=residuals, t_embed_fn=RefineNoiseNet.timestep_embedding)
            epoch_loss.append(loss)
        print('epoch_loss:', np.mean(epoch_loss))
        


    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
