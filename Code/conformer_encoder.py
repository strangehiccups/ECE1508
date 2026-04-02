import torch
import torch.nn as nn

from feedforward import FeedForward
from mhsa import MHSA
from convolution import Convolution

class ConformerEncoder(nn.Module):
    def __init__(self,
                 latent_dim: int=256,
                 ff_dim: int=1024,
                 heads: int=4,
                 kernel_size: int=15,
                 dropout=0.1):
        super().__init__()
        # 1. Feed Forward 1
        self.ff1 = FeedForward(latent_dim, ff_dim, dropout)
        # 2. Multi-Head Self Attention
        self.mhsa = MHSA(latent_dim, heads, dropout)
        # 3. Convolution
        self.conv = Convolution(latent_dim, kernel_size, dropout)
        # 4. Feed Forward 2
        self.ff2 = FeedForward(latent_dim, ff_dim, dropout)
        # 5. Layernorm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, key_padding_mask=None):
        # Macaron FFN (half step)
        x = x + 0.5 * self.ff1(x)
        # MHSA
        x = x + self.mhsa(x, key_padding_mask)
        # Convolution
        x = x + self.conv(x)
        # Second FFN
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)
