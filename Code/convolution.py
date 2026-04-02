import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):
    def __init__(self,
                 latent_dim: int=256,
                 kernel_size=15,
                 dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.pointwise_conv1 = nn.Conv1d(latent_dim, 2 * latent_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            latent_dim, latent_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=latent_dim
        )
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.pointwise_conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x) # (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)