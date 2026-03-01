import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 dropout: float,
                 device: torch.device=None):
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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grus = nn.ModuleList([
            nn.GRU(input_size=self.input_size if i == 0 else self.output_size,
                   hidden_size=self.hidden_size,
                   num_layers=self.num_layers,
                   bias=True,
                   batch_first=True, # (batch, time, features)
                   dropout=self.dropout,
                   bidirectional=self.bidirectional,
                   device=device)
            for i in range(self.num_layers)
        ])
        self.lns = nn.ModuleList([
            # LayerNorm is identical to BatchNorm1d except that it applies per-element scale and bias (no transpose required)
            nn.LayerNorm(self.output_size)
            for i in range(self.num_layers)
        ])

        self.learnable_parameters = 0
        input_size = self.input_size
        for _ in range(self.num_layers):
            self.learnable_parameters += self.layer_parameters(input_size)
            input_size = self.output_size

    def layer_parameters(self, input_size: int):
        return 3*self.hidden_size*input_size \
             + 3*self.hidden_size*self.hidden_size \
             + 6*self.hidden_size \
             + 2*self.output_size

    def forward(self,
                x, # [batch, time, features]
                seq_lens): # expected to be sorted (in decreasing order)
        batch, seq_len, _ = x.shape
        out = x
        for l in range(self.num_layers):
            # pack padded sequence to eliminate unnecessary convolution on pad cells
            out = nn.utils.rnn.pack_padded_sequence(input=out,
                                                    lengths=seq_lens,
                                                    batch_first=True,
                                                    enforce_sorted=True)
            out, _ = self.grus[l](out)   # [batch, time, hidden_size]
            # unpack to match expected layer norm input size
            out, _ = nn.utils.rnn.pad_packed_sequence(sequence=out,
                                                      total_length=seq_len,
                                                      batch_first=True)
            out = self.lns[l](out)
        return out # [batch, time, hidden state sequences]