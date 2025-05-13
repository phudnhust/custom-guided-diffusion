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
    def __init__(self, vector_dim, t_embed_dim, attn_dim):
        """
        vector_dim:   D = C*H*W, flattened codebook‐vector size
        t_embed_dim:  E, timestep embedding size
        attn_dim:     H, hidden dim for attention
        """
        super().__init__()
        # project anchor_vector to query space
        self.Wq = nn.Linear(vector_dim, attn_dim)
        # project timestep embedding to query space
        self.Wt = nn.Linear(t_embed_dim, attn_dim)
        # project top-5 vectors to key space
        self.Wk = nn.Linear(vector_dim, attn_dim)


    def forward(self, topk_vectors, anchor_vector, t_embed):
        """
        topk_vectors: (B, 5, D)    — 5 vector giống nhất đã flatten
        anchor_vector: (B, D)      — vector retrieved (flattened)
        t_embed: (B, E)        — timestep embedding
        Trả về:
          refined: (B, D)          — weighted sum của top5
          weights: (B, 5)          — softmax attention weights
        """

        # 1) projection key: từ (B,5,D) -> (B,5,H)
        keys = self.Wk(topk_vectors)         # (B,5,H)

        # 2) projection query: từ (B,D) -> (B,H) rồi thêm chiều -> (B,1,H)
        # query from anchor: (B,H)
        q_anchor = self.Wq(anchor_vector)  
        # query from timestep: (B,H)
        q_time   = self.Wt(t_embed)
        # combined query: (B,1,H)
        query = (q_anchor + q_time).unsqueeze(1)

        # 3) tính score dot-product và scale
        #    scores[b,j] = <query[b], keys[b,j]> / sqrt(H)
        scores = th.matmul(query, keys.transpose(1, 2))
        dk = query.size(-1)
        scores = scores / math.sqrt(dk)

        # 4) softmax để ra weights
        weights = F.softmax(scores, dim=-1)        # weights.shape: torch.Size([32, 1, 5])
                                                   # topk_vectors.shape: torch.Size([32, 5, 196608])

        # 5) weighted sum trên giá trị là chính topk_vectors
        refined = th.einsum('b n k, b k d -> b n d', weights, topk_vectors).squeeze(1)  # → [B, 1, 196608]
        #refined.shape: torch.Size([32, 196608]) 
        return refined, weights

    def timestep_embedding(self, timesteps: th.LongTensor, dim: int):
        """
        timesteps: (B,) chứa các giá trị bước t ∈ [0, T)
        dim:       kích thước embedding (E)
        Trả về:    Tensor (B, E)
        """
        half = dim // 2
        exponents = th.arange(half, dtype=th.float32, device=timesteps.device) * (
            -math.log(10000) / (half - 1)
        )
        freqs = th.exp(exponents)                      # (half,)
        args  = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb   = th.cat([th.sin(args), th.cos(args)], dim=1) # (B, dim)
        if dim % 2:  # nếu E là lẻ
            emb = th.cat([emb, th.zeros((emb.size(0), 1), device=emb.device)], dim=1)
        return emb  # (B, E)

    def train_refine_step(
        self,    # your RefineNoiseNet instance
        optimizer,     # optimizer for refine_net
        anchor_vector,  # Tensor, shape (B, C, H, W)
        topk_vectors,  # Tensor, shape (B, 5, C, H, W)
        residuals,     # Tensor, shape (B, 1, C, H, W)  = (x0 - x0_pred),
        t_embed     # (B,E)
    ):
        """
        Performs one training step of RefineNoiseNet.

        Inputs:
        topk_vectors  (B, 5, C, H, W) — the 5 nearest codebook noise maps  
        timesteps     (B,)            — the diffusion step index for each sample  
        residuals     (B, C, H, W)    — the true denoising residual x0 – x0_pred  
        t_embed_fn    fn to get timestep embeddings of shape (B, E)

        Returns:
        loss.item()  — the scalar inner‐product loss
        """
        # print('len(topk_vectors.shape):', )
        if len(topk_vectors.shape) == 5:
            B, K, C, H, W = topk_vectors.shape
        else:
            topk_vectors = topk_vectors.unsqueeze(1)
            B, K, C, H, W = topk_vectors.shape
        D = C*H*W

        vectors_flat = topk_vectors.contiguous().view(B, K, D)    # (B,5,D)
        anchor_flat  = anchor_vector.contiguous().view(B, D)      # (B,D)

           # 3) Forward through RefineNoiseNet
        # v_hat: (B, D), weights: (B, 5)
        with th.cuda.amp.autocast():
            v_hat, weights = self(vectors_flat, anchor_flat, t_embed)
            v_hat = v_hat.view(B, C, H, W)  # (B, C, H, W)
            loss = F.mse_loss(v_hat, residuals.squeeze(1))  # L2 loss

            # 5) Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    
    def save_checkpoint(self, optimizer, epoch, path="refiner_checkpoint.pth"):
        """
        Saves model+optimizer state and current epoch.
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
    
    #---------------- REFINE NET INITIALIZE -----------------------
    checkpoint = th.load(repo_folder_path + 'refine_net_100.pth')
    refine_net = RefineNoiseNet(vector_dim=3*256*256, t_embed_dim=128, attn_dim=256).to(dist_util.dev())   
    refine_net.load_state_dict(checkpoint['model_state_dict'])

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # sample_fn = (
        #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        # )
        sample_fn = diffusion.ddcm_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            codebooks=_codebooks,
            # hq_img_path="/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/academic_gown/000.jpg",
            # hq_img_path="/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/academic_gown/004.jpg",
            # hq_img_path='/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama/4.jpg',
            # hq_img_path='/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Jennifer Lopez/8.jpg',
            hq_img_path='/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Leonardo DiCaprio/20.jpg',
            noise_refine=True,
            noise_refine_model=refine_net
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_filename =  f"samples_{shape_str}"
        out_path = os.path.join(repo_folder_path + "npz_output", out_filename + ".npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        
        # output to png
        data = np.load(out_path)
        images = data['arr_0'][0]
        plt.imshow(images)
        plt.axis('off')  # Remove axes for a cleaner image
        plt.savefig(repo_folder_path + f"png_output/" + out_filename + datetime.now().strftime("_date_%Y%m%d_time_%H%M") + ".png", bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to avoid overlapping

    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
