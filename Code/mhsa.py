import torch
import torch.nn as nn

class MHSA(nn.Module):
    def __init__(self,
                 latent_dim: int=256,
                 heads: int=4,
                 dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = self.norm(x)
        x, _ = self.mhsa(x, x, x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        return x
