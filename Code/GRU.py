import torch
import torch.nn as nn

# DeepSpeech2's GRU (Gated Recurrent Units) Layers
# Hyperparameters proposed by the DeepSpeec2 paper:
#     hidden units: 2560
#     RNN direction: unidirectional
#     no. of GRU->BN layers: 3
class GRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=2560,
                 num_layers=3,
                 bidirectional=False,
                 device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.output_size = 2.0*self.hidden_size
        else:
            self.output_size = self.hidden_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grus = nn.ModuleList([
            nn.GRU(input_size=input_size if i == 0 else self.output_size,
                   hidden_size=self.hidden_size,
                   num_layers=1,
                   bias=False,       # batch norm renders bias irrelevant
                   batch_first=True, # (batch, time, features)
                   dropout=0.0,
                   bidirectional=self.birectional,
                   device=device)
            for i in range(num_layers)
        ])
        self.lns = nn.ModuleList([
            # LayerNorm is identical to BatchNorm1d except that it applies per-element scale and bias (no transpose required)
            nn.LayerNorm(self.output_size)
            for i in range(num_layers)
        ])

    def forward(self,
                x, # x: (batch, time, features)
                seq_lens): # expected to be sorted (in decreasing order)
        batch, seq_len, _ = x.shape
        out = x
        for l in range(self.num_layers):
            out = nn.utils.rnn.pack_padded_sequence(input=out,
                                                    lengths=seq_lens,
                                                    batch_first=True,
                                                    enforce_sorted=True) # NOTE: lengths are expected to be sorted (in decreasing order)
            out, _ = self.grus[l](out)   # (batch, time, hidden_size)
            out, _ = nn.utils.rnn.pad_packed_sequence(sequence=out,
                                                      total_length=seq_len,
                                                      batch_first=True)
            out = self.lns[l](out)
        return out # (batch, hidden state sequences