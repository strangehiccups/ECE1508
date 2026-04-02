import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,
                 latent_dim: int=256,
                 ff_dim: int=1024,
                 dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, latent_dim),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)
