import torch
import torch.nn as nn
import torch.nn.functional as F

class LookAheadConv(nn.Module):
    def __init__(self,
                 in_channels,
                 context=80):
        super().__init__()
        self.in_channels = in_channels
        self.output_size = self.in_channels # convolution over time, preserve no. features 
        self.context = context

        # 1D conv over time (just linear temporal mixing, no activation)
        self.conv = nn.Conv1d(in_channels=self.in_channels,
                              out_channels=self.output_size,
                              kernel_size=context + 1,
                              stride=1,
                              padding=0,
                              groups=self.in_channels,  # depthwise (per feature)
                              bias=False) # batch norm renders bias irrelevant
        self.LN = nn.LayerNorm(self.output_size)

    # TO CONSIDER: transpose-free implementation with learnable weight matrix and simple tensor operations (efficiency gained might not makeup for highly optimised Conv1d)
    def forward(self,
                x): # [batch, time, features]
        # Pad future context on right
        out = F.pad(x, (0, 0, 0, self.context))  # pad time dimension
        out = out.transpose(1, 2)                # Conv1d expects [B, F, T]
        out = self.conv(out)
        out = out.transpose(1, 2)                # LayerNorm operates on last dimension [B, F, T] -> [B, T, F]
        out = self.LN(out)

        return out
