import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


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
        weights = F.softmax(scores / 0.1, dim=-1)        # weights.shape: torch.Size([32, 1, 5])
                                                   # topk_vectors.shape: torch.Size([32, 5, 196608])
        # print('weights: ', weights)

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
