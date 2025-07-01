import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SoftmaxAttention(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, codebook, anchor_noise):
        # codebook: (K, C, H, W)
        # anchor_noise: (C, H, W)

        anchor_noise_flat_norm = nn.functional.normalize(anchor_noise.view(1, -1), dim=1)
        codebook_flat_norm     = nn.functional.normalize(codebook.view(codebook.shape[0], -1), dim=1)

        cos_sim = torch.matmul(anchor_noise_flat_norm, codebook_flat_norm.T).squeeze(0)

        sims_value, sims_index = torch.topk(cos_sim, k=self.top_k, largest=True)
        sims_value = sims_value.view(-1)
        sims_index = sims_index.view(-1)
        # print('Non-learnable softmax attention')
        # print('idxs: ', idxs)
        
        topk_vectors = codebook[sims_index].squeeze(1)                                 # (topk, C, H, W)
        weights = nn.functional.softmax(sims_value / 0.1, dim=0)      # (topk)
        # print('weights: ', weights)
        
        noise = sum(weights[:, None, None, None] * topk_vectors, dim=0, keepdim=True)    # (1, C, H, W)
        # print(f'refine noise: shape {noise.shape} mean {noise.mean()} std {noise.std()} min {noise.min()} max {noise.max()}')
        return noise

       