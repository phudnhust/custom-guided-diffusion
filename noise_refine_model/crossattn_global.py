import torch
import torch.nn as nn

class GlobalCrossAttentionRefiner(nn.Module):
    """
    Global cross-attention refiner handling HF tensors with band dimension.

    Stage1: query=HF_star, key/value=HF_info
    Stage2: query=[z_star, HF_star], key/value=[Z_cands, HF_info]

    Args:
        feat_dim (int): number of channels per HF band (e.g., 3 for RGB).
        band_dim (int): number of HF bands (e.g., 3).
        embed_dim (int): dimension for attention projections.
        num_heads (int): number of attention heads.
    """
    def __init__(self,
                 feat_dim: int,
                 band_dim: int,
                 embed_dim: int,
                 num_heads: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.band_dim = band_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Stage1 input dim = feat_dim * band_dim
        in1 = feat_dim * band_dim
        self.proj_q1 = nn.Linear(in1, embed_dim)
        self.proj_k1 = nn.Linear(in1, embed_dim)
        self.proj_v1 = nn.Linear(in1, embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Stage2 input dim = feat_dim (z_star) + feat_dim*band_dim (HF_star)
        in2 = feat_dim + feat_dim * band_dim
        self.proj_q2 = nn.Linear(in2, embed_dim)
        self.proj_k2 = nn.Linear(in2, embed_dim)
        self.proj_v2 = nn.Linear(in2, embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self,
                batch_hf_star: torch.Tensor,
                batch_hf_info: torch.Tensor,
                batch_noise_candidate: torch.Tensor):
        """
        Args:
            batch_hf_star:         [B, S, C, H, W]
            batch_hf_info:         [B, K, S, C, H, W]
            batch_noise_candidate: [B, K, C, H, W]
        Returns:
            z_star, z_hat, w1, w2
        """
        # Unpack shapes
        B, S, C, H, W = batch_hf_star.shape
        _, K, S2, C2, H2, W2 = batch_hf_info.shape
        assert S == S2 and C == C2 and H == H2 and W == W2
        _, K2, C3, H3, W3 = batch_noise_candidate.shape
        assert K == K2 and C == C3 and H == H3 and W == W3

        # Stage1: pool HF_star and HF_info over spatial
        # flatten band and channel dims
        hf_star_flat = batch_hf_star.view(B, S * C, -1).mean(dim=2)        # [B, S*C]
        hf_info_flat = batch_hf_info.view(B, K, S * C, -1).mean(dim=3)     # [B, K, S*C]

        # Project and apply multi-head attention
        q1 = self.proj_q1(hf_star_flat).unsqueeze(1)                      # [B, 1, E]
        k1 = self.proj_k1(hf_info_flat)                                   # [B, K, E]
        v1 = self.proj_v1(hf_info_flat)                                   # [B, K, E]
        _, attn_w1 = self.attn1(q1, k1, v1)                               # [B, heads, 1, K]
        w1 = attn_w1.mean(dim=1).squeeze(1)                                # [B, K]

        # Intermediate mix z_star
        Z_cands = batch_noise_candidate
        z_star = (w1.view(B, K, 1, 1, 1) * Z_cands).sum(dim=1)            # [B, C, H, W]

        # Stage2: pool z_star and hf_star for query, hf_info & noise cands for key
        z_star_flat = z_star.view(B, C, -1).mean(dim=2)                  # [B, C]
        # reuse hf_star_flat for HF part of query
        q2_input = torch.cat([z_star_flat, hf_star_flat], dim=1)         # [B, C+S*C]

        # key/value: concatenate noise and HF_info pooled features
        zcands_flat = Z_cands.view(B, K, C, -1).mean(dim=3)               # [B, K, C]
        hfcinfo_flat = hf_info_flat                                       # [B, K, S*C]
        kv2_flat = torch.cat([zcands_flat, hfcinfo_flat], dim=2)          # [B, K, C+S*C]

        q2 = self.proj_q2(q2_input).unsqueeze(1)                          # [B, 1, E]
        k2 = self.proj_k2(kv2_flat)                                       # [B, K, E]
        v2 = self.proj_v2(kv2_flat)                                       # [B, K, E]
        _, attn_w2 = self.attn2(q2, k2, v2)                               # [B, heads, 1, K]
        w2 = attn_w2.mean(dim=1).squeeze(1)                                # [B, K]

        # Final refined noise z_hat
        z_hat = (w2.view(B, K, 1, 1, 1) * Z_cands).sum(dim=1)             # [B, C, H, W]

        return z_star, z_hat

    def save_checkpoint(self,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        path: str = "refiner_checkpoint.pth"):
        """
        Save model and optimizer state to a checkpoint file.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
