import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t[:, None]
        half = self.embed_dim // 2
        scale = math.log(10000) / (half - 1)
        emb_exp = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t.float() * emb_exp
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.linear(emb)

class RefineINRWithTopK(nn.Module):
    def __init__(self, patch_size=32, in_channels=3, num_neighbors=5, hidden_dim=128, time_embed_dim=64):
        super().__init__()
        total_inputs = (1 + num_neighbors) * in_channels
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(total_inputs, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 2 + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z_patch_anchor, z_patch_neighbors, pos_xy, t):
        """
        z_patch_anchor: (B, C, H, W)
        z_patch_neighbors: (B, 5, C, H, W)
        pos_xy: (B, 2), normalized
        t: (B,)
        """
        B = z_patch_anchor.size(0)
        z_neighbors_flat = z_patch_neighbors.view(B, -1, *z_patch_anchor.shape[2:])
        z_cat = torch.cat([z_patch_anchor, z_neighbors_flat], dim=1)
        patch_feat = self.patch_encoder(z_cat).view(B, -1)
        t_feat = self.time_embedding(t)
        x = torch.cat([patch_feat, pos_xy, t_feat], dim=1)
        return self.mlp(x)  # (B, 3)

    def train_individual_epoch(self, optimizer, batch_patch_z,batch_z_neighbors, batch_rel_pos, batch_t, batch_patch_target):
        with torch.cuda.amp.autocast():
            z_hat = self(batch_patch_z, batch_z_neighbors, batch_rel_pos, batch_t)
            loss = torch.nn.functional.l1_loss(z_hat, batch_patch_target)  # L2 loss

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
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

def get_patch_data(z, neighbors, residual, t, patch_size=32):  # z, residual: (C, H, W); neighbors: (5, C, H, W)
    _, H, W = z.shape

    # (i,j) : top-left corner
    i = random.randint(0, H - patch_size)
    j = random.randint(0, W - patch_size)
    patch_z = z[:, i : i + patch_size, j : j + patch_size]


    cx = (j + patch_size) / W
    cy = (i + patch_size) / H
    rel_pos = torch.tensor([cx, cy], dtype=torch.float)     # relative position

    cy_i = int(i + patch_size - 1)
    cx_j = int(j + patch_size - 1)
    patch_target = residual[:, cy_i, cx_j]

    patch_neighbors = neighbors[:, :, i : i + patch_size, j : j + patch_size]

    return patch_z, patch_neighbors, rel_pos, t, patch_target
     

# # --- Dataset definition for direction refinement ---
# class DDCMNoiseDirectionDataset(Dataset):
#     """
#     Dataset of (z_t patch, rel coords, t) -> target = x0 - xt
#     """
#     def __init__(self, z_images, x0_images, xt_images, ts, patch_size=32, patches_per_image=100):
#         assert len(z_images) == len(x0_images) == len(xt_images) == len(ts)
#         self.z_images = z_images
#         self.x0 = x0_images
#         self.xt = xt_images
#         self.ts = ts
#         self.ps = patch_size
#         self.K = patches_per_image

#     def __len__(self):
#         return len(self.z_images) * self.K

#     def __getitem__(self, idx):
#         img_idx = idx // self.K
#         z_img = self.z_images[img_idx]
#         x0_img = self.x0[img_idx]
#         xt_img = self.xt[img_idx]
#         t = self.ts[img_idx]
#         _, H, W = z_img.shape

#         i = random.randint(0, H - self.ps)
#         j = random.randint(0, W - self.ps)
#         patch_z = z_img[:, i:i+self.ps, j:j+self.ps]

#         cx = (j + self.ps/2) / W
#         cy = (i + self.ps/2) / H
#         rel = torch.tensor([cx, cy], dtype=torch.float)

#         cy_i = int(i + self.ps/2)
#         cx_j = int(j + self.ps/2)
#         target = x0_img[:, cy_i, cx_j] - xt_img[:, cy_i, cx_j]

#         return patch_z, rel, t, target

# # --- Training loop ---
# def train_refine(model, dataset, epochs=10, batch_size=32, lr=1e-4, device='cuda'):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     model.to(device)
#     opt = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for ep in range(epochs):
#         total_loss = 0.0
#         model.train()
#         for patch, rel, t, gt in loader:
#             patch = patch.to(device)
#             rel = rel.to(device)
#             t = t.to(device)
#             gt = gt.to(device)

#             pred = model(patch, rel, t)
#             loss = criterion(pred, gt)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             total_loss += loss.item() * patch.size(0)

#         print(f"Epoch {ep+1}: Loss = {total_loss/len(dataset):.6f}")

# # Example usage
# if __name__ == "__main__":
#     z_images = [torch.randn(3, 256, 256) for _ in range(10)]
#     x0_images = [torch.randn(3, 256, 256) for _ in range(10)]
#     xt_images = [torch.randn(3, 256, 256) for _ in range(10)]
#     ts = list(range(10))

#     dataset = DDCMNoiseDirectionDataset(z_images, x0_images, xt_images, ts)
#     model = RefineINRTimeAware()
#     train_refine(model, dataset, epochs=5, batch_size=16, lr=1e-4, device='cpu')
