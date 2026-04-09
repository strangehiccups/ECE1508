from enum import Enum
from transformers import Wav2Vec2CTCTokenizer

# DATASET URLS
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

# Language model URLs
ARPA_GZ_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
PHONEME_LEXICON_URL = "https://www.openslr.org/resources/11/librispeech-lexicon.txt"

# TRAINING CONFIG
# TEMPORAL_NETWORK = TemporalNetwork.CONFORMER
BATCH_SIZE  = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20

# DATALOADER CONFIG
NUM_WORKERS = 0
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# AUDIO FEATURE CONFIG
HOP_LENGTH = 256
N_FFT = 512
N_MELS = 80
SAMPLE_RATE = 16000

# SAVE PATHS
SAVE_MODEL_PATH = "model.pth"
SAVE_BEST_MODEL_PATH = "model_best.pth"
SAVE_HISTORY_PATH = "history.h5"
HISTORY_KEYS = [
    "train_loss",
    "val_loss",
    "train_cer",
    "val_cer",
    "train_wer",
    "val_wer",
    "train_time",
    "val_time"
]

# TOKENIZER CONFIG
TOKENIZER = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
BLANK_TOKEN_ID = TOKENIZER.pad_token_id

# -- Language-model-augmented beam search -------------------------------------
LM_WEIGHT      = 1.5    # How much to weight the language model vs. the acoustic model
WORD_SCORE     = -0.5   # Value added to the score of each word to encourage or discourage the decoder from outputting more words. 
LM_ARPA_PATH   = "../data/lm/kenlm_3gram.arpa"
LEXICON_PATH   = "../data/lm/librispeech-lexicon-grapheme.txt"