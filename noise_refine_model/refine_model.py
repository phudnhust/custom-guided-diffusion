import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

# --------- RefineNoiseNet Definition ---------
class RefineNoiseNet(nn.Module):
    def __init__(self, vector_dim, t_embed_dim, hidden_dim=256):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(vector_dim + t_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, vectors, t_embed):
        B, K, D = vectors.shape
        t_expanded = t_embed.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([vectors, t_expanded], dim=-1)
        scores = self.score_mlp(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        refined = torch.sum(weights.unsqueeze(-1) * vectors, dim=1)
        return refined, weights

# --------- Timestep Embedding ---------
def timestep_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(-torch.arange(half_dim, device=t.device) * emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# --------- Cosine Similarity Search ---------
def batch_top5_similar(queries, codebook):
    queries = F.normalize(queries, dim=1)
    codebook = F.normalize(codebook, dim=1)
    sim = torch.matmul(queries, codebook.T)  # (B, K)
    top5_scores, top5_indices = torch.topk(sim, k=5, dim=1)
    topk_vectors = codebook[top5_indices]  # (B, 5, D)
    return topk_vectors, top5_indices

# --------- DDCM Reverse Step Placeholder ---------
class DummyDDCM(nn.Module):
    def reverse_step(self, x_t_flat, v_hat, t):
        # Placeholder reverse step; replace with actual model logic
        return x_t_flat - v_hat  # Example: simple residual correction

# --------- Training Loop ---------
def train_refiner(dataloader, ddcm_model, codebooks, refine_net, optimizer, alphas, t_embed_dim, device):
    refine_net.train()
    for x0 in dataloader:
        x0 = x0.to(device)
        B, C, H, W = x0.shape
        D = C * H * W
        x0_flat = x0.view(B, -1)

        t = torch.randint(0, len(codebooks), (B,), device=device)
        noise = torch.randn_like(x0)

        alpha_t = alphas[t].view(B, 1, 1, 1)
        x_t = (alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise).detach()
        noise_flat = noise.view(B, -1)

        codebook_t = codebooks[t[0]]  # assuming shared codebook per step
        dists = torch.cdist(noise_flat, codebook_t)  # (B, K)
        anchors = torch.argmin(dists, dim=1)
        anchor_vecs = codebook_t[anchors]  # (B, D)

        topk_vectors, _ = batch_top5_similar(anchor_vecs, codebook_t)
        t_embed = timestep_embedding(t, t_embed_dim)

        v_hat, _ = refine_net(topk_vectors, t_embed)
        x_t_flat = x_t.view(B, -1)
        x0_t = ddcm_model.reverse_step(x_t_flat, v_hat, t).view(B, C, H, W)

        residual = (x0 - x0_t).view(B, -1)
        loss = -torch.sum(residual * v_hat, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

# --------- Example Usage ---------
if __name__ == "__main__":
    # Dummy data and config for demonstration
    vector_dim = 256 * 256 * 3
    t_embed_dim = 128
    batch_size = 4
    codebook_size = 64
    num_timesteps = 1000

    # Dummy codebooks: List of (K, D)
    codebooks = [F.normalize(torch.randn(codebook_size, vector_dim), dim=1) for _ in range(num_timesteps)]
    alphas = torch.linspace(0.0001, 0.9999, num_timesteps)

    # Dummy model and refiner
    ddcm_model = DummyDDCM()
    refine_net = RefineNoiseNet(vector_dim, t_embed_dim).cuda()
    optimizer = torch.optim.Adam(refine_net.parameters(), lr=1e-4)

    # Dummy dataset loader
    dummy_data = torch.randn(20, 3, 256, 256)
    dataloader = DataLoader(dummy_data, batch_size=batch_size, shuffle=True)

    # Train loop
    train_refiner(dataloader, ddcm_model, codebooks, refine_net, optimizer, alphas, t_embed_dim, device="cuda")
