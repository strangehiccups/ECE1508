import transformers
import torch
import torch.nn as nn
import CNN
import GRU
import LookAheadConv

# DeepSpeech2 paper: "best English model has 2 layers of 2D convolution, 
#                     followed by 3 layers of unidirectional recurrent layers with 2560 GRU cells each,
#                     followed by a lookahead convolution layer with tau = 80,
#                     trained with BatchNorm and SortaGrad"
class DeepSpeech2(nn.Module):
    # defaults are based on the 2-layer 2D architecture
    def __init__(self,
                 tokenizer:transformers.PreTrainedTokenizerBase,
                 conv_in_channels=1,
                 conv_out_channels=32,
                 GRU_hidden_size=2560,
                 GRU_depth=3,
                 device=None):
        super().__init__()
        # 0. device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1. feature extractor: time (x frequency) tensor -> feature maps
        self.feature_extractor = CNN(in_channels=conv_in_channels,
                                     out_channels=conv_out_channels) # more parameters to be configurable
        # 2. GRU block: features -> hidden state sequences (time-sequential information)
        self.gru = GRU(input_size=self.feature_extractor.output_size, # TO DO: add output_size in CNN class
                       hidden_size=GRU_hidden_size,
                       num_layers=GRU_depth,
                       birectional=False,
                       device=device)
        # 3. look ahead convolution block: hidden state sequences -> hidden state sequences with future context
        self.lookAheadConv = LookAheadConv(in_channels=self.gru.output_size,
                                           context=80)
        # 4. output layer: hidden state sequences with future context -> character probabilities
        self.head = nn.Linear(self.lookAheadConvs.output_size, tokenizer.vocab_size)
        # 4a. log softmax activation for CTC loss (only during training)
        self.logSoftmax = nn.LogSoftmax()
        # 4b. softmax activation (only during inference)
        self.softmax = nn.Softmax()
    
    def forward(self,
                x, # x: (batch, time, frequency)
                seq_lens):
        out, final_seq_lens = self.feature_extractor(x, seq_lens)
        out = self.gru(out, seq_lens)
        out = self.lookAheadConv(out)
        out = self.head(out)
        if self.training:
            out = self.logSoftmax(out)
        else:
            out = self.softmax(out)
        return out, final_seq_lens
