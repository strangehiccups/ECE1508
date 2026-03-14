import transformers
from transformers import Wav2Vec2CTCTokenizer
import torch
import torch.nn as nn
from cnn import ConvolutionFeatureExtractor
from lstm import LSTM
from look_ahead_conv import LookAheadConv

# DeepSpeech2 variant using LSTM instead of GRU.
# Architecture is otherwise identical to deep_speech_2.py:
#   2-layer 2D CNN -> N x (LSTM + LayerNorm) -> LookAheadConv -> Linear + LogSoftmax
#
# LSTM vs GRU notes:
#   - LSTM adds an explicit cell state (c_t) alongside the hidden state (h_t),
#     which can improve gradient flow for long sequences.
#   - In practice, LSTM ≈ GRU for ASR at this scale; the main benefit comes from
#     combining with bidirectionality (set LSTM_bidirectional=True).
#   - If bidirectional=True, halve LSTM_hidden_size to 256 to keep parameter count
#     comparable to the unidirectional GRU baseline.
#
# LSTM layers are stacked manually (one per ModuleList slot) with a LayerNorm after
# each layer, mirroring gru.py's design for a fair A/B comparison.
class DeepSpeech2LSTM(nn.Module):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase=None,
                 conv_in_channels: int=1,
                 conv_out_channels: int=32,
                 LSTM_hidden_size: int=512,
                 LSTM_depth: int=3,
                 LSTM_bidirectional: bool=False,
                 LSTM_dropout: float=0.3,
                 look_ahead_context: int=40,
                 device: torch.device=None):
        super().__init__()
        # 0. tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
        self.blank_token_id = self.tokenizer.pad_token_id

        # 1. feature extractor: time (x frequency) tensor -> feature maps
        self.feature_extractor = ConvolutionFeatureExtractor(in_channels=conv_in_channels,
                                                             out_channels=conv_out_channels,
                                                             device=device)

        # 2. LSTM block: stacked manually so LayerNorm can be applied between layers,
        #    matching gru.py's design.
        self.lstm_output_size = 2 * LSTM_hidden_size if LSTM_bidirectional else LSTM_hidden_size
        self.lstms = nn.ModuleList([
            LSTM(input_size=self.feature_extractor.output_size if i == 0 else self.lstm_output_size,
                 hidden_size=LSTM_hidden_size,
                 num_layers=1,
                 bidirectional=LSTM_bidirectional,
                 dropout=LSTM_dropout)
            for i in range(LSTM_depth)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(self.lstm_output_size)
            for _ in range(LSTM_depth)
        ])

        # 3. look ahead convolution block: hidden state sequences -> sequences with future context
        self.lookAheadConv = LookAheadConv(in_channels=self.lstm_output_size,
                                           context=look_ahead_context)

        # 4. output layer: hidden state sequences with future context -> character logits
        self.head = nn.Linear(self.lookAheadConv.output_size, self.tokenizer.vocab_size)
        # 4a. log softmax for CTC loss during training
        self.logSoftmax = nn.LogSoftmax(dim=2)
        # 4b. softmax for inference
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        x,        # [batch, channel, frequency, time]
        seq_lens
    ):
        out, final_seq_lens = self.feature_extractor(x, seq_lens)
        for lstm, ln in zip(self.lstms, self.lns):
            out = lstm(out, final_seq_lens)
            out = ln(out)
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
    ):
        # nn.CTCLoss expects log_probs of shape [input (time), batch, class]
        log_probs = log_probs.transpose(0, 1)
        loss = nn.CTCLoss(blank=blank, reduction=reduction)
        return loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=seq_lens,
            target_lengths=target_lens
        )
