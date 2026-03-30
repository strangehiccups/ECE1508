import torch
import torch.nn as nn

class LSTM(nn.Module):
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
            self.output_size = 2 * self.hidden_size
        else:
            self.output_size = self.hidden_size

        # Stacked manually so LayerNorm can be applied between layers.
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=self.input_size if i == 0 else self.output_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional)
            for i in range(self.num_layers)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(self.output_size)
            for _ in range(self.num_layers)
        ])
        self.drop = nn.Dropout(p=self.dropout)

    def forward(self,
                x,        # [batch, time, features]
                seq_lens): # expected to be sorted (in decreasing order)
        seq_lens = seq_lens.detach().to(dtype=torch.int64).cpu()
        batch, seq_len, _ = x.shape
        out = x
        for l in range(self.num_layers):
            # Pack the padded sequence for efficient processing
            out = nn.utils.rnn.pack_padded_sequence(input=out,
                                                    lengths=seq_lens,
                                                    batch_first=True,
                                                    enforce_sorted=True)
            out, _ = self.lstms[l](out)
            # Unpack the output back to padded sequence format
            out, _ = nn.utils.rnn.pad_packed_sequence(sequence=out,
                                                      total_length=seq_len,
                                                      batch_first=True)
            out = self.lns[l](out)
            # Apply dropout for each stacked layer
            if l < self.num_layers - 1:
                out = self.drop(out)
        return out  # [batch, time, hidden state sequences]
