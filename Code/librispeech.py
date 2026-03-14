import logging
import os
from pathlib import Path

import librosa
import requests
import tarfile
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2CTCTokenizer

from utils import AudioSample, get_audio_mel_spectrogram

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")


# AI-generated code for logging setup. Not part of the original codebase, but included here for completeness.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def download_librispeech(data_dir: str, split: str, url: str) -> None:
    """Download and extract a LibriSpeech split into *data_dir* if not already present.

    After extraction the split lives at::

        data_dir/LibriSpeech/<split>/
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / "LibriSpeech" / split
    if split_dir.exists():
        logger.info(f"LibriSpeech {split} already present at {split_dir}. Skipping download.")
        return

    url = url
    archive_path = data_dir / f"LibriSpeech-{split}.tar.gz"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading LibriSpeech {split} from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info("Download complete. Extracting...")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    logger.info(f"Extraction complete. Dataset available at {split_dir}")

    archive_path.unlink()


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
    download:
        If ``True``, download the split when not already present.
    """

    def __init__(self, data_dir: str, split: str = "test-clean", download: bool = False):
        self.split_dir = Path(data_dir) / "LibriSpeech" / split

        if download:
            download_librispeech(data_dir, split)

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"LibriSpeech split directory not found: {self.split_dir}. "
                "Pass download=True to download it automatically."
            )

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
        audio, sr = librosa.load(file_path, sr=None)
        mel_audio = get_audio_mel_spectrogram(audio, sr)

        raw_text = self.labels.get(file_path.stem, "")
        try:
            raw_text = raw_text.upper()
            tokenized_text = tokenizer.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing transcript for {file_path.name}. Skipping sample.")
            return None

        return AudioSample(
            raw_audio=audio,
            mel_audio=torch.tensor(mel_audio, dtype=torch.float32),
            sample_rate=sr,
            file_path=str(file_path),
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long),
        )

def load_librispeech(data_dir: str, split: str = "test-clean", download: bool = False) -> LibriSpeechDataset:
    """Helper function to load LibriSpeech dataset."""
    return LibriSpeechDataset(data_dir, split=split, download=download)