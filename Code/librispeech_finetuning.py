import logging
from pathlib import Path

import torchaudio
import torch
from torch.utils.data import Dataset

from utils import AudioSample, get_audio_mel_spectrogram, spec_augment
from config import TOKENIZER

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class LibriSpeechFinetuningDataset(Dataset):
    """Limited-supervision finetuning splits built on LibriSpeech.

    Expected layout after extraction::

        data_dir/
            1h/
                0/clean/   # 2 speakers, clean  ─┐
                0/other/   # 2 speakers, other   │ fold 0 (10 min)
                ...                              │
                5/clean/                         ├─ 1h split (6 folds)
                5/other/                         │
            9h/                                  │
                clean/     # 12 speakers, clean  ┘
                other/     # 12 speakers, other    9h remainder
            phones/        # frame-wise phoneme transcriptions

    Each subdirectory follows the standard LibriSpeech structure::

        <speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
        <speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    Parameters
    ----------
    data_dir:
        Root directory that contains the ``1h/``, ``9h/``, and ``phones/``
        folders (i.e. the extracted archive root).
    split:
        ``"10min"``  – first 10-minute fold only (``1h/0``)
        ``"1h"``     – all six 10-minute folds combined (``1h/``)
        ``"10h"``    – full dataset: ``1h/`` + ``9h/`` (``10h = 1h + 9h``)
    condition:
        ``"clean"``, ``"other"``, or ``"both"`` (default).
    augment:
        If ``True``, SpecAugment is applied and stored in
        ``AudioSample.mel_audio_spec_augment`` for use by
        ``collate_fn_train``.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "10h",
        condition: str = "both",
        augment: bool = False,
    ):
        self.base = Path(data_dir)
        self.augment = augment
        self.subdirs = self._resolve_subdirs(split, condition)

        logger.info(
            f"Loading LibriSpeech finetuning (split={split}, condition={condition}) "
            f"from {self.base}..."
        )
        self.file_paths, self.labels = self._index()
        logger.info(f"Found {len(self.file_paths)} utterances.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_subdirs(self, split: str, condition: str) -> list[Path]:
        conditions = ["clean", "other"] if condition == "both" else [condition]
        dirs: list[Path] = []


        if split in ("10min", "1h", "10h"):
            logger.info(f"Resolving subdirectories for split='{split}', condition='{condition}'...")
            folds = range(6) if split in ("1h", "10h") else range(1)
            # 
            for fold in folds:
                for cond in conditions:
                    d = self.base / "1h" / str(fold) / cond
                    if d.exists():
                        dirs.append(d)
            
            logger.info(f"Resolved 1h subdirectories: {len(dirs)}")

        if split == "10h":
            logger.info(f"Resolving subdirectories for split='{split}', condition='{condition}'...")
            for cond in conditions:
                d = self.base / "9h" / cond
                if d.exists():
                    dirs.append(d)

            logger.info(f"+Resolved 9h subdirectories: {len(dirs)}")

        if not dirs:
            raise FileNotFoundError(
                f"No data found for split='{split}', condition='{condition}' "
                f"under {self.base}. "
                f"Please download the dataset using "
                f"data_loader.download_librispeech_finetuning()."
            )
        return dirs

    def _index(self):
        """Collect .flac paths and their transcripts across all subdirs."""
        labels: dict[str, str] = {}
        for subdir in self.subdirs:
            for trans_file in subdir.rglob("*.trans.txt"):
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        utt_id, _, transcript = line.partition(" ")
                        labels[utt_id] = transcript

        file_paths: list[Path] = []
        for subdir in self.subdirs:
            file_paths.extend(
                p for p in subdir.rglob("*.flac") if p.stem in labels
            )
        return sorted(file_paths), labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> AudioSample:
        file_path = self.file_paths[idx]
        audio, sr = torchaudio.load(file_path)
        mel_audio = get_audio_mel_spectrogram(audio, sr)

        raw_text = self.labels.get(file_path.stem, "").upper()
        try:
            tokenized_text = TOKENIZER.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing transcript for {file_path.name}. Skipping sample.")
            return None

        mel_augmented = spec_augment(mel_audio) if self.augment else None

        return AudioSample(
            raw_audio=audio,
            raw_mel_audio=mel_audio,
            mel_audio_spec_augment=mel_augmented,
            sample_rate=sr,
            file_path=str(file_path),
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long),
        )


def load_librispeech_finetuning(
    data_dir: str,
    split: str = "10h",
    condition: str = "both",
    augment: bool = False,
) -> LibriSpeechFinetuningDataset:
    """Helper function to load the LibriSpeech finetuning dataset."""
    return LibriSpeechFinetuningDataset(data_dir, split=split, condition=condition, augment=augment)
