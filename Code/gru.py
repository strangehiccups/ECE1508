import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class GRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 dropout: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        if self.bidirectional:
            self.output_size = 2*self.hidden_size
        else:
            self.output_size = self.hidden_size

        self.grus = nn.ModuleList([
            nn.GRU(input_size=self.input_size if i == 0 else self.output_size,
                   hidden_size=self.hidden_size,
                   num_layers=1,  # we stack multiple GRU layers instead of using the built-in multi-layer functionality to allow for layer norm in between
                   bias=True,
                   batch_first=True, # (batch, time, features)
                   bidirectional=self.bidirectional)
            for i in range(self.num_layers)
        ])
        self.lns = nn.ModuleList([
            # LayerNorm is identical to BatchNorm1d except that it normalises over each sample instead of over the entire batch (no transpose required)
            nn.LayerNorm(self.output_size)
            for i in range(self.num_layers)
        ])
        self.drop = nn.Dropout(p=self.dropout)

    def layer_parameters(self, input_size: int):
        return 3*self.hidden_size*input_size \
             + 3*self.hidden_size*self.hidden_size \
             + 6*self.hidden_size \
             + 2*self.output_size

    def forward(self,
                x, # [batch, time, features]
                seq_lens): # expected to be sorted (in decreasing order)
        seq_lens = seq_lens.detach().to(dtype=torch.int64).cpu() # pack_padded_sequence requires lengths to be on CPU
        _, seq_len, _ = x.shape
        # Pack once before the layer loop — eliminates redundant pack/unpack overhead on every layer
        out = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=True)
        for l in range(self.num_layers):
            out, _ = self.grus[l](out)   # PackedSequence in, PackedSequence out
            # Apply LayerNorm directly to packed data tensor [total_tokens, hidden_size] —
            # semantically identical to unpacking first since LayerNorm operates per-token
            normed = self.lns[l](out.data)
            # Apply dropout between layers only (not after the final layer)
            if l < self.num_layers - 1:
                normed = self.drop(normed)
            out = PackedSequence(normed, out.batch_sizes, out.sorted_indices, out.unsorted_indices)
        # Unpack once after all layers
        out, _ = pad_packed_sequence(out, total_length=seq_len, batch_first=True)
        return out # [batch, time, hidden state sequences]
