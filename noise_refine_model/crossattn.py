import math
import torch
import torch.nn.functional as F
import torch.nn as nn
# from einops import rearrange
# from torchvision import transforms
# from xformers.ops import memory_efficient_attention, AttentionOpDispatch
import numpy as np

def laplacian_kernel(img_tensor):
    """
    Apply Laplacian high-pass filter to each channel independently.
    Args:
        img_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    Returns:
        high_freq: torch.Tensor of same shape as input
    """
    # Add batch dimension if needed
    squeeze_batch = False
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze_batch = True

    C = img_tensor.shape[1]
    # Create Laplacian kernel for each channel
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=img_tensor.dtype, device=img_tensor.device
    ).expand(C, 1, 3, 3)  # shape (C, 1, 3, 3)

    # Apply per-channel convolution
    high_freq = F.conv2d(img_tensor, laplacian_kernel, padding=1, groups=C)

    # Remove batch dimension if needed
    if squeeze_batch:
        high_freq = high_freq.squeeze(0)

    return high_freq

import torch
import torch.nn as nn

class PixelCrossAttentionRefiner(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_heads):
        """
        feat_dim  = C, channel‐dim of your HF and Z tensors
        embed_dim = desired attention subspace (set = C if you want [B,C,H,W] out)
        num_heads = number of attention heads (embed_dim % num_heads == 0)
        """
        super().__init__()
        self.feat_dim = feat_dim
        # first block: input is (C + 2 coords) → embed_dim
        self.q1 = nn.Linear(feat_dim + 2, embed_dim)
        self.k1 = nn.Linear(feat_dim + 2, embed_dim)
        self.v1 = nn.Linear(feat_dim    , embed_dim)  # values don’t need coords

        # second block: input is (C + embed_dim + 2 coords) → embed_dim
        self.q2 = nn.Linear(feat_dim + embed_dim + 2, embed_dim)
        self.k2 = nn.Linear(feat_dim + embed_dim + 2, embed_dim)
        self.v2 = nn.Linear(feat_dim + embed_dim    , embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def _cross_attn(self, Q_feat, K_feats, V_feats, q_proj, k_proj, v_proj):
        B, _, H, W = Q_feat.shape
        K = K_feats.size(1)
        N = H * W

        # --- build coords for sanity check (not used here) ---
        # flatten Query: [B, Cq, H, W] → [B*N, Cq]
        Query = Q_feat.permute(0,2,3,1).reshape(B*N, -1)
        # sanity‐check
        assert Query.shape[1] == q_proj.in_features, (
            f"Query has {Query.shape[1]} dims but q_proj expects {q_proj.in_features}"
        )
        # project
        Query = q_proj(Query).unsqueeze(0)  # → [1, B*N, E]

        # flatten K: [B, K, Ck, H, W] → [B*N, K, Ck]
        Key = K_feats.permute(0,1,3,4,2).reshape(B*N, K, -1)
        assert Key.shape[2] == k_proj.in_features, (
            f"Key has {Key.shape[2]} dims but k_proj expects {k_proj.in_features}"
        )
        Key = k_proj(Key).permute(1,0,2)  # → [K, B*N, E]

        # flatten Value: [B, K, Cv, H, W] → [B*N, K, Cv]
        # print('type(B, N, K):', type(B), type(N), type(K))  
        Value = V_feats.permute(0,1,3,4,2).reshape(B*N, K, -1)
        assert Value.shape[2] == v_proj.in_features, (
            f"Value has {Value.shape[2]} dims but v_proj expects {v_proj.in_features}"
        )
        Value = v_proj(Value).permute(1,0,2)  # → [K, B*N, E]

        # multi‐head attention
        out, _ = self.attn(Query, Key, Value)     # [1, B*N, E]
        out = out.squeeze(0)            # [B*N, E]
        out = out.reshape(B, H, W, -1).permute(0,3,1,2)
        return out                      # [B, E, H, W]

    def forward(self, HF_star, HF_cands, Z_cands):
        B, C, H, W = HF_star.shape
        K = HF_cands.size(1)
        device = HF_star.device

        # build normalized coord channels
        i = torch.linspace(0, 1, H, device=device)
        j = torch.linspace(0, 1, W, device=device)
        i_grid, j_grid = torch.meshgrid(i, j, indexing='ij')
        i_grid = i_grid.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)
        j_grid = j_grid.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)

        # --- first cross‐attention ---
        Q1 = torch.cat([HF_star, i_grid, j_grid], dim=1)            # [B, C+2, H, W]
        # print("Q1 channels:", Q1.shape[1])    # should equal feat_dim + 2
        # print("self.feat_dim + 2 = ", self.feat_dim + 2)    # should equal feat_dim + 2
        # print()

        i_k = i_grid.unsqueeze(1).expand(-1,K,-1,-1,-1)             # [B,K,1,H,W]
        j_k = j_grid.unsqueeze(1).expand(-1,K,-1,-1,-1)             # [B,K,1,H,W]
        K1 = torch.cat([HF_cands, i_k, j_k], dim=2)                 # [B,K,C+2,H,W]
        V1 = Z_cands                                               # [B,K,C, H, W]
        z_star = self._cross_attn(Q1, K1, V1, self.q1, self.k1, self.v1)
        # return z_star
    
        # --- second cross‐attention ---
        z_star_exp = z_star.unsqueeze(1).expand(-1,K,-1,-1,-1)      # [B,K,embed_dim,H,W]
        Q2 = torch.cat([HF_star, z_star, i_grid, j_grid], dim=1)    # [B,C+E+2,H,W]
        K2 = torch.cat([HF_cands, z_star_exp, i_k, j_k],    dim=2)  # [B,K,C+E+2,H,W]
        V2 = torch.cat([Z_cands,   z_star_exp],            dim=2)  # [B,K,C+E   ,H,W]
        z_hat = self._cross_attn(Q2, K2, V2, self.q2, self.k2, self.v2)

        return z_hat  # [B, embed_dim, H, W]

    def save_checkpoint(self, optimizer, epoch, path="refiner_checkpoint.pth"):
        """
        Saves model+optimizer state and current epoch.
        
        refine_net:   your RefineNoiseNet instance
        optimizer:    the optimizer you’re using (e.g. AdamW)
        epoch:        current epoch number (int)
        path:         where to write the .pth file
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")
