import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


# --------- RefineNoiseNet Definition ---------
class RefineNoiseNet(nn.Module):
    def __init__(self, vector_dim=3*256*256, t_embed_dim=128, attn_dim=128, num_heads=8):
        """
        vector_dim:   D = C*H*W, flattened codebook‐vector size
        t_embed_dim:  E, timestep embedding size
        attn_dim:     H, hidden dim for attention
        """
        super().__init__()                       # ← Must be first

        # project D→H
        self.Wq = nn.Linear(vector_dim + t_embed_dim, attn_dim)
        self.Wk = nn.Linear(vector_dim, attn_dim)
        self.Wv = nn.Linear(vector_dim, attn_dim)
        # attention in H-space
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim,
                                         num_heads=num_heads,
                                         batch_first=True)
        # project back H→D
        self.proj_out = nn.Linear(attn_dim, vector_dim)


    def forward(self, topk_vectors, anchor_vector, t_embed):
        """
        topk_vectors: (B, 5, D)    — 5 vector giống nhất đã flatten
        anchor_vector: (B, D)      — vector retrieved (flattened)
        t_embed: (B, E)        — timestep embedding
        Trả về:
          weights: (B, 5)          — softmax attention weights
        """
        # top5: (B,5,D) → keys/values in H
        K = self.Wk(topk_vectors)               # (B,5,H)
        V = self.Wv(topk_vectors)               # (B,5,H)
        # anchor & time → query in H
        Q = self.Wq(th.cat([anchor_vector, t_embed], dim=-1))  # (B,H)
        Q = Q.unsqueeze(1)              # (B,1,H)

        out, attn = self.mha(Q, K, V)   # out: (B,1,H)
        # print('out.shape: ', out.shape)
        out = out.squeeze(1)            # (B,H)
        refined = self.proj_out(out)    # (B,D)
        return refined, attn.squeeze(1) # (B,5)

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
            loss = F.l1_loss(v_hat, residuals.squeeze(1))  # L2 loss

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
