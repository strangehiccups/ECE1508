import torch
from torch import nn

# DeepSpeech2 model requires one or more CNN layers
# Below is the examples proposed by the DeepSpeec2 paper (1D: time, 2D: time x frequency):
# Architecture | Channels      | Filter dimension    | Stride        | Regular Dev | Noisy Dev
# 1-layer 1D   | 1280          | 11                  | 2             | 9.52        | 19.36
# 2-layer 1D   | 640, 640      | 5, 5                | 1, 2          | 9.67        | 19.21
# 3-layer 1D   | 512, 512, 512 | 5, 5, 5             | 1, 1, 2       | 9.20        | 20.22
# 1-layer 2D   | 32            | 41x11               | 2x2           | 8.94        | 16.22
# 2-layer 2D   | 32, 32        | 41x11, 21x11        | 2x2, 2x1      | 9.06        | 15.71
# 3-layer 2D   | 32, 32, 96    | 41x11, 21x11, 21x11 | 2x2, 2x1, 2x1 | 8.61        | 14.74
# The overall architecture: Convo2d layer -> BatchNorm2d layer -> Clipped rectified-linear ReLU
# Clipped ReLU is computed as: σ(x) = min{max{x, 0}, 20}
class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)):
        super().__init__()
        self.cnn = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.clipped_relu = lambda x: torch.clamp(x, min=0, max=20)

    def forward(self, x, seq_lens):
        # Expected input shape: (batch_size, channels, frequency, time)
        x = self.cnn(x)
        x = self.bn(x)
        x = self.clipped_relu(x)

        # Account for different sequence lengths (time axis)
        seq_lens = (seq_lens + 2 * self.cnn.padding[1] - self.cnn.dilation[1] * (self.cnn.kernel_size[1] - 1) - 1) // self.cnn.stride[1] + 1
        
        return x, seq_lens

# Contains multiple CNN layers stacked together to extract features from the input
# TODO: If necessary, add more tunnable hyperparameters: number of layers, kernel, stride, etc.
class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, in_feat_dim=80):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 

        self.conv1 = CNNLayer(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        self.conv2 = CNNLayer(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))

        # Compute output_size based on how the frequency dimension changes
        feat_dim = in_feat_dim
        for layer in [self.conv1, self.conv2]:
            pad = layer.cnn.padding[0]
            kernel = layer.cnn.kernel_size[0]
            stride = layer.cnn.stride[0]
            dilation = layer.cnn.dilation[0]
            feat_dim = (feat_dim + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            
        self.output_size = feat_dim * self.out_channels
    
    def forward(self, x, seq_lens):
        x, seq_lens = self.conv1(x, seq_lens)
        x, seq_lens = self.conv2(x, seq_lens)

        # Output shape is currently: (batch_size, out_channels, frequency, time)
        batch_size, out_channels, freq, time = x.size()
        # The GRU expects: (batch_size, time, feature = out_channels * frequency)
        # Permute to (batch_size, time, out_channels, frequency)
        x = x.permute(0, 3, 1, 2).contiguous()
        # Flatten the frequency and channels to create the feature vector
        x = x.view(batch_size, time, out_channels * freq)

        return x, seq_lens
