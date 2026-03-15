import requests
import tarfile
import gzip
import shutil
import os

from config import TOKENIZER


def download_ljspeech(url: str, path: str = "../data/LJSpeech-1.1"):
    if not os.path.exists(path):
        print("Downloading LJSpeech dataset...")
        lj_archive = f"{path}.tar.bz2"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(lj_archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. Extracting...")
        with tarfile.open(lj_archive, "r:bz2") as tar:
            tar.extractall(path=path)
        os.remove(lj_archive)
        print("LJSpeech ready.")
    else:
        print("LJSpeech already present, skipping download.")


def download_librispeech(url: str, path: str = "../data/LibriSpeech", split: str = "test-clean"):
    if not os.path.exists(f"{path}/{split}"):
        print(f"Downloading LibriSpeech {split}...")
        ls_archive = f"{path}/LibriSpeech-{split}.tar.gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(ls_archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. Extracting...")
        with tarfile.open(ls_archive, "r:gz") as tar:
            tar.extractall(path=path)
        os.remove(ls_archive)
        print(f"LibriSpeech {split} ready.")
    else:
        print(f"LibriSpeech {split} already present, skipping download.")


def download_lm(url: str, path: str = "../data/lm", filename: str = "kenlm_3gram.arpa"):
    arpa_path = os.path.join(path, filename)
    lm_gz_path = os.path.join(path, f"{filename}.gz")

    if not os.path.exists(arpa_path):
        os.makedirs(path, exist_ok=True)

        print(f"Downloading language model {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(lm_gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Download complete. Decompressing...")

        with gzip.open(lm_gz_path, "rb") as f_in, open(arpa_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(lm_gz_path)

        print(f"Language model ready: {arpa_path}")
    else:
        print(f"Language model already present, skipping download.")


# The CTC model outputs characters, NOT phonemes. The beam-search decoder
# needs a lexicon (pronunciation ) that maps each word to its character-level spelling using
# the model's token vocabulary.
#   Format:  WORD\tW O R D |\n
# where each letter is a token the acoustic model can emit, and | is the
# word-boundary / silence token.
def download_lexicon(
    url: str,
    path: str = "../data/lm",
    phoneme_filename: str = "librispeech-lexicon-phoneme.txt",  # What a word sounds like
    lexicon_filename: str = "librispeech-lexicon-grapheme.txt", # How a word is spelled
):
    phoneme_lexicon_path = os.path.join(path, phoneme_filename)
    lexicon_path = os.path.join(path, lexicon_filename)

    os.makedirs(path, exist_ok=True)

    ### ---- Download phoneme lexicon ----
    if not os.path.exists(phoneme_lexicon_path):
        print("Downloading phoneme lexicon...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(phoneme_lexicon_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Phoneme lexicon ready.")
    else:
        print("Phoneme lexicon already present, skipping download.")

    ### ---- Build graphemic lexicon ----
    if not os.path.exists(lexicon_path):
        vocab_set = set(TOKENIZER.get_vocab().keys())
        single_chars = {tok for tok in vocab_set if len(tok) == 1}

        words = set()
        with open(phoneme_lexicon_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    words.add(parts[0])

        written = 0
        with open(lexicon_path, "w") as f:
            for word in sorted(words):
                mapped = list(word)
                if all(c in single_chars or c == "'" for c in mapped):
                    spelled = " ".join(mapped)
                    f.write(f"{word}\t{spelled} |\n")
                    written += 1

        print(f"Graphemic lexicon written: {lexicon_path} ({written} words)")
    else:
        print(f"Graphemic lexicon already exists: {lexicon_path}")