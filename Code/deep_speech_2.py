import transformers
from transformers import Wav2Vec2CTCTokenizer
import torch
import torch.nn as nn
from cnn import ConvolutionFeatureExtractor
from gru import GRU
from look_ahead_conv import LookAheadConv

# DeepSpeech2 paper: "best English model has 2 layers of 2D convolution, 
#                     followed by 3 layers of unidirectional recurrent layers with 2560 GRU cells each,
#                     followed by a lookahead convolution layer with tau = 80,
#                     trained with BatchNorm and SortaGrad"
# DeepSpeech2's GRU (Gated Recurrent Units) Layers
# Hyperparameters proposed by the DeepSpeec2 paper:
#     hidden units: 2560
#     RNN direction: unidirectional
#     no. of GRU->BN layers: 3
#     dropout: not mentioned
# Hyperparameters proposed by https://openspeech-team.github.io/openspeech/architectures/DeepSpeech2.html (fewer learnable parameters but slower due to bidrectional layers):
#     hidden units: 1024
#     RNN direction: bidirectional
#     no. of GRU->BN layers: 5
#     dropout: 0.3
class DeepSpeech2(nn.Module):
    # defaults are based on the 2-layer 2D architecture
    def __init__(self,
                 tokenizer:transformers.PreTrainedTokenizerBase=None,
                 conv_in_channels: int=1,
                 conv_out_channels: int=32,
                 in_feat_dim: int=80,
                 GRU_hidden_size: int=512,   # DeepSpeech configuration (2560) is overkill for LJSpeech
                 GRU_depth: int=3,
                 GRU_bidirectional: bool=False, # unidirectional as we have look ahead convolution to add future context (cheaper but weaker approach)
                 GRU_dropout: float=0.3,
                 look_ahead_context: int=40):   # DeepSpeech configuration (80) is overkill for LJSpeech
        super().__init__()
        # 0. tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
        self.blank_token_id = self.tokenizer.pad_token_id
        # 1. feature extractor: time (x frequency) tensor -> feature maps
        self.feature_extractor = ConvolutionFeatureExtractor(in_channels=conv_in_channels,
                                                             out_channels=conv_out_channels,
                                                             in_feat_dim=in_feat_dim,
                                                             device=device)
        # 2. GRU block: features -> hidden state sequences (time-sequential information)
        self.gru = GRU(input_size=self.feature_extractor.output_size,
                       hidden_size=GRU_hidden_size,
                       num_layers=GRU_depth,
                       bidirectional=GRU_bidirectional,
                       dropout=GRU_dropout)
        # 3. look ahead convolution block: hidden state sequences -> hidden state sequences with future context
        self.lookAheadConv = LookAheadConv(in_channels=self.gru.output_size,
                                           context=look_ahead_context)
        # 4. output layer: hidden state sequences with future context -> character logits
        self.head = nn.Linear(self.lookAheadConv.output_size, self.tokenizer.vocab_size)
        # 4a. log softmax activation for CTC loss (only during training): character logits -> log character probabilities
        self.logSoftmax = nn.LogSoftmax(dim=2) # head output shape: [batch, time, logits], log softmax on logits
        # 4b. softmax activation (only during inference): character logits -> character probabilities
        self.softmax = nn.Softmax(dim=2)       # head output shape: [batch, time, logits], softmax on logits
    
    def forward(
        self,
        x, # [batch, channel, frequency, time]
        seq_lens
    ):
        out, final_seq_lens = self.feature_extractor(x, seq_lens)
        out = self.gru(out, final_seq_lens)
        out = self.lookAheadConv(out)
        out = self.head(out)
        out = self.logSoftmax(out)
        return out, final_seq_lens

    @staticmethod
    def loss_fn(
        blank,
        log_probs,    # [batch, time, log character probability]
        seq_lens,     # [sequence length]
        targets,      # [all targets over all batches]
        target_lens,
        reduction='mean'
    ): # [target length]
        # nn.CTCLoss expects log_probs of shape [input (time), batch, class]
        log_probs = log_probs.transpose(0,1)
        loss = nn.CTCLoss(
            blank=blank, reduction=reduction
        )
        return loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=seq_lens,
            target_lengths=target_lens
        )

