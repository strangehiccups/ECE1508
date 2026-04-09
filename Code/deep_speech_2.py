from typing import Optional

import torch.nn as nn
import transformers
from transformers import Wav2Vec2CTCTokenizer

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
    """DeepSpeech2-inspired model with a GRU temporal block.

    Set ``look_ahead_context`` to add a LookAheadConv after the GRU (used with
    unidirectional GRU to approximate future context cheaply).  Leave it as
    ``None`` for a plain GRU variant (typically bidirectional) without the
    look-ahead convolution.

    Architecture:
        ConvolutionFeatureExtractor → GRU → [LookAheadConv] → Linear → LogSoftmax
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase = None,
        conv_in_channels: int = 1,
        conv_out_channels: int = 32,
        in_feat_dim: int = 80,
        GRU_hidden_size: int = 512,
        GRU_depth: int = 3,
        GRU_bidirectional: bool = False,
        GRU_dropout: float = 0.3,
        look_ahead_context: Optional[int] = None,  # None → no look-ahead conv
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
        self.blank_token_id = self.tokenizer.pad_token_id

        # 1. 2D CNN feature extractor
        self.feature_extractor = ConvolutionFeatureExtractor(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            in_feat_dim=in_feat_dim,
        )
        # 2. GRU temporal block
        self.gru = GRU(
            input_size=self.feature_extractor.output_size,
            hidden_size=GRU_hidden_size,
            num_layers=GRU_depth,
            bidirectional=GRU_bidirectional,
            dropout=GRU_dropout,
        )
        # 3. Optional look-ahead convolution (adds future context for unidirectional models)
        if look_ahead_context is not None:
            self.lookAheadConv = LookAheadConv(
                in_channels=self.gru.output_size,
                context=look_ahead_context,
            )
            head_input_size = self.lookAheadConv.output_size
        else:
            self.lookAheadConv = None
            head_input_size = self.gru.output_size

        # 4. Output projection
        self.head = nn.Linear(head_input_size, self.tokenizer.vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=2)
        self._ctc_loss = nn.CTCLoss(blank=self.blank_token_id, reduction="mean")

    def forward(self, x, seq_lens):
        out, final_seq_lens = self.feature_extractor(x, seq_lens)
        out = self.gru(out, final_seq_lens)
        if self.lookAheadConv is not None:
            out = self.lookAheadConv(out)
        out = self.head(out)
        out = self.logSoftmax(out)
        return out, final_seq_lens

    def loss_fn(self, blank, log_probs, seq_lens, targets, target_lens, reduction="mean"):
        log_probs = log_probs.transpose(0, 1)
        return self._ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=seq_lens,
            target_lengths=target_lens,
        )
