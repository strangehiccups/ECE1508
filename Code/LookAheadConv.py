import torch
import torch.nn as nn
import torch.nn.functional as F

class LookaheadConv(nn.Module):
    def __init__(self,
                 in_channels,
                 context=80):
        super().__init__()
        self.in_channels = in_channels
        self.output_size = self.in_channels # output channels = input channels
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

    def forward(self, x): # x: (batch, time, features)
        # Pad future context on right
        out = F.pad(x, (0, 0, 0, self.context))  # pad time dimension
        out = self.conv(out)
        out = self.LN(out)
        return out
