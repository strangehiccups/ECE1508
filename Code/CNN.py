import torch


# DeepSpeech2 model requires one or more CNN layers
# Below is the examples proposed by the DeepSpeec2 paper:
# Architecture | Channels      | Filter dimension    | Stride        | Regular Dev | Noisy Dev
# 1-layer 1D   | 1280          | 11                  | 2             | 9.52        | 19.36
# 2-layer 1D   | 640, 640      | 5, 5                | 1, 2          | 9.67        | 19.21
# 3-layer 1D   | 512, 512, 512 | 5, 5, 5             | 1, 1, 2       | 9.20        | 20.22
# 1-layer 2D   | 32            | 41x11               | 2x2           | 8.94        | 16.22
# 2-layer 2D   | 32, 32        | 41x11, 21x11        | 2x2, 2x1      | 9.06        | 15.71
# 3-layer 2D   | 32, 32, 96    | 41x11, 21x11, 21x11 | 2x2, 2x1, 2x1 | 8.61        | 14.74
# The overall architecture: Convo2d layer -> BatchNorm2d layer -> Clipped rectified-linear ReLU
# Clipped ReLU is computed as: σ(x) = min{max{x, 0}, 20}
class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.cnn = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.clipped_relu = lambda x: torch.clamp(x, min=0, max=20)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.clipped_relu(x)
        return x
