from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, x, seq_lens):
        # Pack the padded sequence for efficient processing
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        # Unpack the output back to padded sequence format
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output