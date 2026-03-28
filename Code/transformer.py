import transformers
from transformers import Wav2Vec2CTCTokenizer
import torch
import torch.nn as nn
from cnn import ConvolutionFeatureExtractor
from positional_encoding import PositionalEncoding

# based on "Attention Is All You Need" (no need for decoder as CTC already aligns & decodes)
class Transformer(nn.Module):
    def __init__(self,
                 max_seq_len: int=2000,
                 conv_in_channels: int=1,
                 conv_out_channels: int=32,
                 enc_latent_dim: int=256,       # number of expected features in the input
                 enc_nhead: int=4,              # number of heads in the multiheadattention models
                 enc_feedforward_dim: int=1024, # dimension of the feedforward network model
                 enc_dropout: float=0.1,
                 tokenizer: transformers.PreTrainedTokenizerBase=None):
        super().__init__()
        # 0. tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
        self.blank_token_id = self.tokenizer.pad_token_id
        # 1. feature extractor: time (x frequency) tensor -> feature maps
        self.feature_extractor = ConvolutionFeatureExtractor(in_channels=conv_in_channels,
                                                             out_channels=conv_out_channels)
        # 2. projection to feature space
        self.input_proj = nn.Linear(self.feature_extractor.output_size, enc_latent_dim)
        # 3. positional encoding: input -> input + positional encoding
        self.positional_encoder = PositionalEncoding(
          d_model=enc_latent_dim,
          max_len=max_seq_len
        )
        # 4. encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_latent_dim,
            nhead=enc_nhead,
            dim_feedforward=enc_feedforward_dim,
            dropout=enc_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        # 5. output layer: features -> character logits
        self.output_layer = nn.linear(enc_latent_dim, self.tokenizer.vocab_size)
        # 6. log softmax activation for CTC loss: character logits -> log character probabilities
        self.logSoftmax = nn.LogSoftmax(dim=2) # output layer's output shape: [batch, time, logits], log softmax on logits
    
    def forward(self,
                x, # [batch, channel, frequency, time]
                seq_lens):
        # 1. CNN feature extraction
        x, final_seq_lens = self.feature_extractor(x, seq_lens)
        # 2. encoder projection + positional encoding
        x = self.input_proj(x)
        x = self.positional_encoder(x)
        # 3. mask future positions (True = pad)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(len(final_seq_lens), max_len)
        src_key_padding_mask = mask >= final_seq_lens.unsqueeze(1)
        # 4. encode
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # 5. map to logits
        x = self.output_layer(x)
        # 6. compute log probabilities
        x = self.logSoftmax(x)
        return x, final_seq_lens

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
