import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class PixelCrossAttentionRefiner(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_heads):
        """
        Initialize PixelCrossAttentionRefiner.
        
        Args:
            feat_dim (int): Channel dimension of your HF and Z tensors (C)
            embed_dim (int): Desired attention subspace dimension (set = C if you want [B,C,H,W] out)
            num_heads (int): Number of attention heads (must satisfy embed_dim % num_heads == 0)
        """
        super().__init__()
        
        if not isinstance(feat_dim, int) or feat_dim <= 0:
            raise ValueError(f"feat_dim must be a positive integer, got {feat_dim}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        
        # first block: input is (C + 2 coords) → embed_dim
        self.q1 = nn.Linear(feat_dim + 2, embed_dim)
        self.k1 = nn.Linear(feat_dim + 2, embed_dim)
        self.v1 = nn.Linear(feat_dim    , embed_dim)  # values don't need coords

        # second block: input is (C + embed_dim + 2 coords) → embed_dim
        self.q2 = nn.Linear(feat_dim + feat_dim + 2, embed_dim)
        self.k2 = nn.Linear(feat_dim + feat_dim + 2, embed_dim)
        self.v2 = nn.Linear(feat_dim    , embed_dim)  # values don't need coords

        # self.feat_dim = feat_dim * 3
        # self.q1 = nn.Linear(feat_dim * 3 + 2, embed_dim)
        # self.k1 = nn.Linear(feat_dim * 3 + 2, embed_dim)
        # self.v1 = nn.Linear(feat_dim    , embed_dim)  # values don't need coords

        # self.q2 = nn.Linear(feat_dim + feat_dim * 3 + 2, embed_dim)
        # self.k2 = nn.Linear(feat_dim + feat_dim * 3 + 2, embed_dim)
        # self.v2 = nn.Linear(feat_dim    , embed_dim)  # values don't need coords

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def _cross_attn(self, Q_feat, K_feats, V_feats, q_proj, k_proj, v_proj, attn):
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

        """
            CODE CŨ (dùng output của nn.MultiheadAttention)
            (trước khi vào attention: chiếu linear đối với value -> không gian attention,
            sau khi tính xong softmax(QK^T) thì chiếu linear một lần nữa -> để đưa về không gian gốc)
        """
        # flatten Value: [B, K, Cv, H, W] → [B*N, K, Cv]
        # print('type(B, N, K):', type(B), type(N), type(K))  
        # Value = V_feats.permute(0,1,3,4,2).reshape(B*N, K, -1)
        # assert Value.shape[2] == v_proj.in_features, (
            # f"Value has {Value.shape[2]} dims but v_proj expects {v_proj.in_features}"
        # )
        # print('before projection: Value.shape: ', Value.shape)     # ([2097152, 5, 3])
        # Value = v_proj(Value).permute(1,0,2)  # → [K, B*N, E]
        # print('after projection: Value.shape: ', Value.shape)     # ([5, 2097152, 3])

        # multi‐head attention
        # out, out_weights = attn(Query, Key, Value)     # [1, B*N, E]
        # out = out.squeeze(0)            # [B*N, E]
        # out = out.reshape(B, H, W, -1).permute(0,3,1,2)

        # print('attention weights: shape', out_weights.shape)    # [2094152, 1, 5]

        """
            CODE MỚI (chỉ dùng weight của nn.MultiheadAttention)
        """
        Value = V_feats.permute(0,1,3,4,2).reshape(B*N, K, -1)      # [B*H*W, K, Cv] = [2097152, 5, 3]
        
        dummy_value_for_attn = torch.zeros((K,B*N, self.embed_dim), device=Value.device)  # [K, B*N, C] = [5, 2097152, 3]
        _, attention_weights = attn(Query, Key, dummy_value_for_attn)   # [B*N, 1, K] = [2097152, 1, 5]
        attention_weights = nn.functional.normalize(attention_weights, p=2, dim=-1)

        if q_proj.in_features == self.q1.in_features:
            print('q_proj_1.weight: ', q_proj.weight)
            print('k_proj_1.weight: ', k_proj.weight)
            print('attention_weights 1: ', attention_weights)     # check constraint sum(square) = 1 -> OK
        else:
            print('q_proj_2.weight: ', q_proj.weight)
            print('k_proj_2.weight: ', k_proj.weight)
            print('attention_weights 2: ', attention_weights)     # check constraint sum(square) = 1 -> OK
        # print('attention_weights.shape: ', attention_weights.shape)     # check constraint sum(square) = 1 -> OK


        out = torch.bmm(attention_weights, Value).squeeze(1)  # [B*N, Cv] = [2097152, 3]
        out = out.reshape(B, H, W, -1).permute(0,3,1,2)
        # print(f'out: mean {out.mean().item():.4f}, std {out.std().item():.4f}, min {out.min().item():.4f}, max {out.max().item():.4f}')

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
        z_star = self._cross_attn(Q1, K1, V1, self.q1, self.k1, self.v1, self.attn1)
        # return z_star
    
        # --- second cross‐attention ---
        z_star_exp = z_star.unsqueeze(1).expand(-1,K,-1,-1,-1)      # [B,K,embed_dim,H,W]
        Q2 = torch.cat([HF_star, z_star, i_grid, j_grid], dim=1)    # [B,C+E+2,H,W]
        K2 = torch.cat([HF_cands, z_star_exp, i_k, j_k],    dim=2)  # [B,K,C+E+2,H,W]
        # V2 = torch.cat([Z_cands,   z_star_exp],            dim=2)  # [B,K,C+E   ,H,W]
        V2 = Z_cands
        z_hat = self._cross_attn(Q2, K2, V2, self.q2, self.k2, self.v2, self.attn2)

        return z_star, z_hat  # [B, embed_dim, H, W]
    

    # def forward(self, HF_star, HF_cands, Z_cands):
    #     """
    #     HF_star:  [B, 3, 3, H, W]   (3 color channels, 3 wavelet subbands)
    #     HF_cands: [B, K, 3, 3, H, W]
    #     Z_cands:  [B, K, 3, H, W]
    #     """

    #     # print('HF_star.shape ', HF_star.shape)
    #     # print('HF_cands.shape ', HF_cands.shape)
    #     # print('Z_cands.shape ', Z_cands.shape)

    #     B, C, S, H, W = HF_star.shape      # C=3 color channels, S=3 subbands
    #     K = HF_cands.shape[1]              # number of candidates

    #     # flatten color channels and subbands into 9 features
    #     HF_star = HF_star.view(B, C*S, H, W)             # [B, 9, H, W]
    #     HF_cands = HF_cands.view(B, K, C*S, H, W)        # [B, K, 9, H, W]
    #     feat_dim = C * S                                 # 9

    #     device = HF_star.device

    #     # positional coords
    #     i = torch.linspace(0, 1, H, device=device)
    #     j = torch.linspace(0, 1, W, device=device)
    #     i_grid, j_grid = torch.meshgrid(i, j, indexing='ij')
    #     i_grid = i_grid.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)   # [B,1,H,W]
    #     j_grid = j_grid.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)   # [B,1,H,W]

    #     # ------------------------
    #     # FIRST CROSS-ATTENTION
    #     # ------------------------
    #     # Q1: [B, 9+2, H, W]
    #     Q1 = torch.cat([HF_star, i_grid, j_grid], dim=1)

    #     # K1: [B,K,9+2,H,W]
    #     i_k = i_grid.unsqueeze(1).expand(-1,K,-1,-1,-1)     # [B,K,1,H,W]
    #     j_k = j_grid.unsqueeze(1).expand(-1,K,-1,-1,-1)     # [B,K,1,H,W]
    #     K1 = torch.cat([HF_cands, i_k, j_k], dim=2)

    #     # V1: [B,K,3,H,W]  (as before)
    #     V1 = Z_cands

    #     # output z_star: [B, embed_dim, H, W]
    #     # print('Q1.shape ', Q1.shape)
    #     # print('K1.shape ', K1.shape)
    #     # print('V1.shape ', V1.shape)
    #     z_star = self._cross_attn(Q1, K1, V1, self.q1, self.k1, self.v1, self.attn1)
    #     # print('z_star.shape: ', z_star.shape)

    #     # ------------------------
    #     # SECOND CROSS-ATTENTION
    #     # ------------------------
    #     z_star_exp = z_star.unsqueeze(1).expand(-1,K,-1,-1,-1)  # [B,K,embed_dim,H,W]

    #     # Q2: [B, 9 + embed_dim + 2, H, W]
    #     Q2 = torch.cat([HF_star, z_star, i_grid, j_grid], dim=1)

    #     # K2: [B,K,9 + embed_dim + 2, H, W]
    #     K2 = torch.cat([HF_cands, z_star_exp, i_k, j_k], dim=2)

    #     # V2 stays same as V1: [B,K,3,H,W]
    #     V2 = Z_cands

    #     # print('Q2.shape ', Q2.shape)
    #     # print('K2.shape ', K2.shape)
    #     # print('V2.shape ', V2.shape)
    #     z_hat = self._cross_attn(Q2, K2, V2, self.q2, self.k2, self.v2, self.attn2)

    #     return z_star, z_hat  # [B, embed_dim, H, W]


    def save_checkpoint(self, optimizer, epoch, path="refiner_checkpoint.pth"):
        """
        Saves model+optimizer state and current epoch.
        
        Args:
            optimizer: The optimizer being used (e.g. AdamW)
            epoch (int): Current epoch number
            path (str): Path where to write the .pth file
            
        Raises:
            IOError: If saving the checkpoint fails
        """
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f"Successfully saved checkpoint: {path}")
        except Exception as e:
            raise IOError(f"Failed to save checkpoint to {path}: {str(e)}")

import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetPerceptualLoss(nn.Module):
    def __init__(self,
                 layers=('conv1','conv2','conv3','conv4','conv5'),
                 device='cuda'):
        super().__init__()
        # load pretrained AlexNet
        alex_features = models.alexnet(pretrained=True).features.to(device)
        alex_features.eval()
        for p in alex_features.parameters():
            p.requires_grad = False

        # map friendly names to indices in alexnet.features
        layer_idxs = {
            'conv1': 0,   # → after features[0] = Conv2d(3,64,...)
            'conv2': 3,   # → after features[3] = Conv2d(64,192,...)
            'conv3': 6,   # → after features[6] = Conv2d(192,384,...)
            'conv4': 8,   # → after features[8] = Conv2d(384,256,...)
            'conv5': 10,  # → after features[10] = Conv2d(256,256,...)
        }

        # build a ModuleDict of AlexNet‐slices up to each requested layer
        self.blocks = nn.ModuleDict({
            name: nn.Sequential(*list(alex_features.children())[:idx+1])
            for name, idx in layer_idxs.items() if name in layers
        })

        self.criterion = nn.L1Loss()
        # same imagenet normalization
        self.register_buffer('mean', torch.Tensor([0.485,0.456,0.406])[None,:,None,None])
        self.register_buffer('std',  torch.Tensor([0.229,0.224,0.225])[None,:,None,None])

    def forward(self, pred, target):
        """
        pred, target: floats in [0,1], shape (B,3,H,W)
        returns scalar perceptual loss
        """
        # normalize exactly as AlexNet expects
        p = (pred   - self.mean) / self.std
        t = (target - self.mean) / self.std

        loss = 0.0
        for block in self.blocks.values():
            loss += self.criterion(block(p), block(t))
        return loss
