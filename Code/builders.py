from conformer import Conformer
from deep_speech_2 import DeepSpeech2
from deep_speech_2_lstm import DeepSpeech2LSTM


def build_gru_model(in_channels: int, in_feat_dim: int):
    """Unidirectional GRU, no look-ahead convolution, depth 3."""
    print("Building unidirectional GRU, depth 3, without look-ahead convolution...")
    return DeepSpeech2(
        conv_in_channels=in_channels,
        in_feat_dim=in_feat_dim,
        GRU_hidden_size=512,
        GRU_depth=3,
        GRU_bidirectional=False,
        look_ahead_context=None,
    )


def build_gru_look_ahead_model(in_channels: int, in_feat_dim: int):
    """Unidirectional GRU with look-ahead convolution (context=40), depth 3."""
    print("Building unidirectional GRU, depth 3, with look-ahead convolution (context=40)...")
    return DeepSpeech2(
        conv_in_channels=in_channels,
        in_feat_dim=in_feat_dim,
        GRU_hidden_size=512,
        GRU_depth=3,
        GRU_bidirectional=False,
        look_ahead_context=40,
    )


def build_gru_bidirectional_model(in_channels: int, in_feat_dim: int):
    """Bidirectional GRU, depth 3, hidden 256×2."""
    print("Building bidirectional GRU, depth 3, hidden 256×2...")
    return DeepSpeech2(
        conv_in_channels=in_channels,
        in_feat_dim=in_feat_dim,
        GRU_hidden_size=512,
        GRU_depth=3,
        GRU_bidirectional=True,
        look_ahead_context=None,
    )


def build_lstm_model(in_channels: int, in_feat_dim: int):
    """Unidirectional LSTM, depth 3, hidden 512."""
    print("Building unidirectional LSTM, depth 3, hidden 512...")
    return DeepSpeech2LSTM(
        conv_in_channels=in_channels,
        in_feat_dim=in_feat_dim,
        LSTM_hidden_size=512,
        LSTM_depth=3,
        LSTM_bidirectional=False,
        LSTM_dropout=0.3,
    )


def build_lstm_look_ahead_model(in_channels: int, in_feat_dim: int):
    """Bidirectional LSTM with look-ahead convolution (context=40), depth 3."""
    print("Building bidirectional LSTM, depth 3, with look-ahead convolution (context=40)...")
    return DeepSpeech2LSTM(
        conv_in_channels=in_channels,
        in_feat_dim=in_feat_dim,
        LSTM_hidden_size=256,
        LSTM_depth=3,
        LSTM_bidirectional=True,
        LSTM_dropout=0.3,
    )


def build_transformer_model(in_channels: int, in_feat_dim: int):
    """Conformer (CNN + encoder), ~5M parameters."""
    return Conformer(
        conv_in_channels=in_channels,
        enc_latent_dim=144,
        enc_feedforward_dim=576,
        enc_layers=10,
    )
