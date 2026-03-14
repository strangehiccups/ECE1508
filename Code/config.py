from transformers import Wav2Vec2CTCTokenizer

HOP_LENGTH = 256
N_FFT = 512
N_MELS = 80
SAMPLE_RATE = 16000

SAVE_MODEL_PATH = "model.pth"
SAVE_HISTORY_PATH = "history.h5"
HISTORY_KEYS = ["train_loss", "val_loss", "train_cer", "val_cer", "train_wer", "val_wer"]

TOKENIZER = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
BLANK_TOKEN_ID = TOKENIZER.pad_token_id
