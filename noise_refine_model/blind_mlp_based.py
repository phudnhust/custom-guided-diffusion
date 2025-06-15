import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


# --------- RefineNoiseNet Definition ---------
class RefineNoiseNet(nn.Module):
    def __init__(self, vector_dim, t_embed_dim, hidden_dim=256, attn_dim=None):
        super().__init__()
        # vector_dim: D (flattened codebook vector size)
        # t_embed_dim: E (timestep embedding size)
        self.score_mlp = nn.Sequential(
            nn.Linear(vector_dim + t_embed_dim, hidden_dim),  # input: D + E
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output: scalar score
        )

    def forward(self, topk_vectors, anchor_vector, t_embed):
        # vectors: (B, 5, D) - top-5 codebook vectors flattened
        # anchor_vector: (B, D)      — vector retrieved (flattened)
        # t_embed: (B, E) - timestep embeddings

        B, K, D = topk_vectors.shape  # K=5
        t_expanded = t_embed.unsqueeze(1).expand(-1, K, -1)
        x = th.cat([topk_vectors, t_expanded], dim=-1)

        scores = self.score_mlp(x).squeeze(-1)
        weights = F.softmax(scores / 0.1, dim=-1) # (B, 5)

        # # Step 1: Expand anchor to match shape of others
        anchor_vector_expanded = anchor_vector.unsqueeze(1)  # (B, 1, D)
        diff = topk_vectors - anchor_vector_expanded     # (B, 5, D)
        weight_expanded = weights.unsqueeze(-1)      # (B, 5, 1)

        weighted_sum = (weight_expanded * diff).sum(dim=1)  # (B, D)
        refined = anchor_vector + weighted_sum  # (B, D)
        
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
        top5_vectors  (B, 5, C, H, W) — the 5 nearest codebook noise maps  
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
        v_hat, weights = self(vectors_flat, anchor_flat, t_embed)

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