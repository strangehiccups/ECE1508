import logging
from pathlib import Path

import torchaudio
import torch
from torch.utils.data import Dataset

from utils import AudioSample, get_audio_mel_spectrogram, spec_augment
from config import TOKENIZER

# AI-generated code for logging setup. Not part of the original codebase, but included here for completeness.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for a single split (default: *test-clean*).

    Expected layout after extraction::

        data_dir/
            LibriSpeech/
                <split>/
                    <speaker>/
                        <chapter>/
                            <speaker>-<chapter>-<utt>.flac
                            <speaker>-<chapter>.trans.txt

    Each line of a ``.trans.txt`` file has the form::

        <SPEAKER>-<CHAPTER>-<UTT_ID> <TRANSCRIPT>

    Parameters
    ----------
    data_dir:
        Root data directory (the one that contains the ``LibriSpeech/`` folder
        after extraction).
    split:
        Dataset split to load, e.g. ``"test-clean"``.
    """

    def __init__(self, data_dir: str, split: str = "test-clean", download: bool = False, augment: bool = False):
        self.split_dir = Path(data_dir) / "LibriSpeech" / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"LibriSpeech split directory not found: {self.split_dir}. "
                f"Please download and extract the dataset using data_loader.download_librispeech()."
            )

        self.augment = augment
        logger.info(f"Loading LibriSpeech {split} from {self.split_dir}...")
        self.file_paths, self.labels = self._index()
        logger.info(f"Found {len(self.file_paths)} utterances.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self):
        """Walk the split directory, collect .flac paths and their transcripts."""
        labels: dict[str, str] = {}

        # Collect transcripts first (one .trans.txt per chapter directory)
        for trans_file in self.split_dir.rglob("*.trans.txt"):
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, _, transcript = line.partition(" ")
                    labels[utt_id] = transcript

        file_paths = sorted(
            p for p in self.split_dir.rglob("*.flac")
            if p.stem in labels
        )
        return file_paths, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> AudioSample:
        file_path = self.file_paths[idx]
        # LibriSpeech is recorded at 16 kHz — preserve native sample rate
        audio, sr = torchaudio.load(file_path)
        mel_audio = get_audio_mel_spectrogram(audio, sr)
        mel_audio_spec_augment = spec_augment(mel_audio) if self.augment else None

        raw_text = self.labels.get(file_path.stem, "")
        try:
            raw_text = raw_text.upper()
            tokenized_text = TOKENIZER.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing transcript for {file_path.name}. Skipping sample.")
            return None

        return AudioSample(
            raw_audio=audio,
            raw_mel_audio=mel_audio,
            mel_audio_spec_augment=mel_audio_spec_augment,
            sample_rate=sr,
            file_path=str(file_path),
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long),
        )

def load_librispeech(data_dir: str, split: str = "test-clean", augment: bool = False) -> LibriSpeechDataset:
    """Helper function to load LibriSpeech dataset."""
    return LibriSpeechDataset(data_dir, split=split, augment=augment)